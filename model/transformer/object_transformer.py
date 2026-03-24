from typing import Dict, Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn
from cutie.model.group_modules import GConv2d
from cutie.utils.tensor_utils import aggregate
from cutie.model.transformer.positional_encoding import PositionalEncoding
from cutie.model.transformer.transformer_layers import *
from cutie.model.transformer.object_summarizer import *


def topk_features_from_mask(aux_logits, pixel, k):
    """
    Args:
        aux_logits: (bs, num_objects, h, w) - 예측된 마스크 (로짓)
        pixel: (bs, num_objects, c, h, w) - 각 픽셀의 특징 텐서
        k: int - Top-k 픽셀을 선택

    Returns:
        topk_features: (bs, num_objects, k, c) - 선택된 픽셀의 특징
    """
    bs, num_objects, c, h, w = pixel.shape
    # Flatten aux_logits: (bs, num_objects, h * w)
    aux_logits_flat = aux_logits.view(bs, num_objects, -1)  # (bs, num_objects, h * w)

    # Get top-k indices and values: (bs, num_objects, k)
    topk_values, topk_indices = torch.topk(aux_logits_flat, k, dim=-1)  # (bs, num_objects, k)
    
    # Flatten pixel: (bs, num_objects, c, h * w)
    pixel_flat = pixel.view(bs, num_objects, c, -1)  # (bs, num_objects, c, h * w)

    # Gather features for top-k indices
    topk_indices_expanded = topk_indices.unsqueeze(2).expand(-1, -1, c, -1)  # (bs, num_objects, c, k)
    topk_features = torch.gather(pixel_flat, 3, topk_indices_expanded)  # (bs, num_objects, c, k)

    # Transpose to (bs, num_objects, k, c)
    topk_features = topk_features.permute(0, 1, 3, 2).view(bs * num_objects, k, c)  # (bs, num_objects, k, c)

    return topk_features


class QueryTransformerBlock(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_transformer
        self.embed_dim = this_cfg.embed_dim
        self.num_heads = this_cfg.num_heads
        self.num_queries = this_cfg.num_queries
        self.ff_dim = this_cfg.ff_dim

        self.read_from_pixel = CrossAttention(self.embed_dim * 2,
                                              self.num_heads,
                                              add_pe_to_qkv=this_cfg.read_from_pixel.add_pe_to_qkv)


        self.pixel_ffn3 = FFN(self.embed_dim * 2, self.ff_dim)


        self.read_from_query = CrossAttention(self.embed_dim,
                                              self.num_heads,
                                              add_pe_to_qkv=this_cfg.read_from_query.add_pe_to_qkv,
                                              norm=this_cfg.read_from_query.output_norm)
        self.pixel_ffn = PixelFFN(self.embed_dim)
        self.read_from_query2 = CrossAttention(self.embed_dim,
                                               self.num_heads,
                                               add_pe_to_qkv=this_cfg.read_from_query.add_pe_to_qkv,
                                               norm=this_cfg.read_from_query.output_norm)
        self.pixel_ffn2 = PixelFFN(self.embed_dim)

        # self.fusion_conv = nn.Conv2d(self.embed_dim * 2, self.embed_dim, 3, 1, 1)

    def forward(
            self,
            x: torch.Tensor,
            event_token: torch.Tensor,
            pixel: torch.Tensor,
            event: torch.Tensor,
            query_pe: torch.Tensor,
            event_pe: torch.Tensor,
            pixel_pe: torch.Tensor,
            event_pe_f: torch.Tensor,
            attn_mask: torch.Tensor,
            need_weights: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        # x: (bs*num_objects)*num_queries*embed_dim
        # pixel: bs*num_objects*C*H*W
        # event: bs*num_obejcts*C*H*W
        # query_pe: (bs*num_objects)*num_queries*embed_dim
        # pixel_pe: (bs*num_objects)*(H*W)*C
        # event_pe: (bs*num_objects)*(H*W)*C
        # attn_mask: (bs*num_objects*num_heads)*num_queries*(H*W)
        # bs*num_objects*C*H*W -> (bs*num_objects)*(H*W)*C

        bs, num_objects, c, h, w = pixel.shape

        pixel_flat = pixel.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        event_flat = event.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()



        flat_c = torch.cat((pixel_flat, event_flat), dim=-1)
        pe_c = torch.cat((pixel_pe, event_pe_f), dim=-1)

        pixel_flat_c, p_weights = self.read_from_pixel(flat_c,
                                                       flat_c,
                                                       pe_c,
                                                       pe_c,
                                                       need_weights=need_weights)

        pixel_flat_c = self.pixel_ffn3(pixel_flat_c)

        pixel_flat_t = pixel_flat_c[:,:,:c] + pixel_flat
        event_flat_t = pixel_flat_c[:,:,c:] + event_flat



        pixel_flat, p_weights = self.read_from_query(pixel_flat_t,
                                                     x,
                                                     pixel_pe,
                                                     query_pe,
                                                     need_weights=need_weights)
        pixel = self.pixel_ffn(pixel, pixel_flat)

        event_flat, t_weights = self.read_from_query2(event_flat_t,
                                                      event_token,
                                                      event_pe_f,
                                                      event_pe,
                                                      need_weights=need_weights)
        event_flat = self.pixel_ffn2(event, event_flat)
        if need_weights:
            bs, num_objects, _, h, w = pixel.shape
            p_weights = p_weights.transpose(2, 3).view(bs, num_objects, self.num_heads,
                                                       self.num_queries, h, w)

        return x, event_token, pixel, event_flat, t_weights, p_weights


class QueryTransformer(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        this_cfg = model_cfg.object_transformer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_heads = this_cfg.num_heads
        self.num_queries = this_cfg.num_queries

        # query initialization and embedding
        self.query_init = nn.Embedding(self.num_queries, self.embed_dim)
        self.query_emb = nn.Embedding(self.num_queries, self.embed_dim)

        self.event_token = nn.Embedding(self.num_queries, self.embed_dim)
        self.event_emb = nn.Embedding(self.num_queries, self.embed_dim)

        # projection from object summaries to query initialization and embedding
        self.summary_to_query_init = nn.Linear(self.embed_dim, self.embed_dim)
        self.summary_to_query_emb = nn.Linear(self.embed_dim, self.embed_dim)

        self.summary_to_e_query_init = nn.Linear(self.embed_dim, self.embed_dim)
        self.summary_to_e_query_emb = nn.Linear(self.embed_dim, self.embed_dim)

        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature
        self.pixel_init_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.pixel_emb_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.spatial_pe = PositionalEncoding(self.embed_dim,
                                             scale=self.pixel_pe_scale,
                                             temperature=self.pixel_pe_temperature,
                                             channel_last=False,
                                             transpose_output=True)

        self.event_emb_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)

        self.current_summarizer = ObjectSummarizer2(model_cfg)

        # transformer blocks
        self.num_blocks = this_cfg.num_blocks
        self.blocks = nn.ModuleList(
            QueryTransformerBlock(model_cfg) for _ in range(self.num_blocks))
        self.mask_pred = nn.ModuleList(
            nn.Sequential(nn.ReLU(), GConv2d(self.embed_dim, 1, kernel_size=1))
            for _ in range(self.num_blocks + 1))

        self.act = nn.ReLU(inplace=True)

    def forward(self,
                pixel: torch.Tensor,
                event: torch.Tensor,
                obj_summaries: torch.Tensor,
                e_obj_summaries: torch.Tensor,
                selector: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> (torch.Tensor, Dict[str, torch.Tensor]):
        # pixel: B*num_objects*embed_dim*H*W
        # obj_summaries: B*num_objects*T*num_queries*embed_dim
        T = obj_summaries.shape[2]
        bs, num_objects, _, H, W = pixel.shape

        # normalize object values
        # the last channel is the cumulative area of the object
        obj_summaries = obj_summaries.view(bs * num_objects, T, self.num_queries,
                                           self.embed_dim + 1)
        e_obj_summaries = e_obj_summaries.view(bs * num_objects, T, self.num_queries,
                                               self.embed_dim + 1)
        # sum over time
        # during inference, T=1 as we already did streaming average in memory_manager
        obj_sums = obj_summaries[:, :, :, :-1].sum(dim=1)
        obj_area = obj_summaries[:, :, :, -1:].sum(dim=1)
        obj_values = obj_sums / (obj_area + 1e-4)
        obj_init = self.summary_to_query_init(obj_values)
        obj_emb = self.summary_to_query_emb(obj_values)

        # positional embeddings for object queries
        query = self.query_init.weight.unsqueeze(0).expand(bs * num_objects, -1, -1) + obj_init
        query_emb = self.query_emb.weight.unsqueeze(0).expand(bs * num_objects, -1, -1) + obj_emb

        # positional embeddings for pixel features
        pixel_init = self.pixel_init_proj(pixel)
        pixel_emb = self.pixel_emb_proj(pixel)
        pixel_pe = self.spatial_pe(pixel.flatten(0, 1))
        pixel_emb = pixel_emb.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        pixel_pe = pixel_pe.flatten(1, 2) + pixel_emb

        e_obj_sums = e_obj_summaries[:, :, :, :-1].sum(dim=1)
        e_obj_area = e_obj_summaries[:, :, :, -1:].sum(dim=1)
        e_obj_values = e_obj_sums / (e_obj_area + 1e-4)
        e_obj_init = self.summary_to_e_query_init(e_obj_values)
        e_obj_emb = self.summary_to_e_query_emb(e_obj_values)

        event_token = self.event_token.weight.unsqueeze(0).expand(bs * num_objects, -1, -1) + e_obj_init
        event_token_emb = self.event_emb.weight.unsqueeze(0).expand(bs * num_objects, -1, -1) + e_obj_emb

        event_emb = self.event_emb_proj(event)
        event_pe = self.spatial_pe(event.flatten(0, 1))
        event_emb = event_emb.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        event_pe = event_pe.flatten(1, 2) + event_emb

        pixel = pixel_init

        # run the transformer
        aux_features = {'logits': []}

        # first aux output
        aux_logits = self.mask_pred[0](pixel).squeeze(2)
        attn_mask = self._get_aux_mask(aux_logits, selector)
        aux_features['logits'].append(aux_logits)
        for i in range(self.num_blocks):
            query, event_token, pixel, event, q_weights, p_weights = self.blocks[i](query,
                                                                             event_token,
                                                                             pixel,
                                                                             event,
                                                                             query_emb,
                                                                             event_token_emb,
                                                                             pixel_pe,
                                                                             event_pe,
                                                                             attn_mask,
                                                                             need_weights=need_weights)

            # query = event_token + query

            if self.training or i <= self.num_blocks - 1 or need_weights:
                aux_logits = self.mask_pred[i + 1](pixel).squeeze(2)
                obj_upd, _ = self.current_summarizer(aux_logits.sigmoid(), pixel)
                b, n, q, c = obj_upd.shape
                obj_upd = obj_upd.view(b * n, q, c)
                obj_sums_t = obj_upd[:, :, :-1]
                obj_area_t = obj_upd[:, :, -1:]
                # print(obj_area_t[0])
                obj_upd_t = obj_sums_t / (obj_area_t + 1e-4)
                each_mask = torch.einsum('bnchw,bnqc->bnqhw', pixel,obj_upd_t.view(b,n,q,c-1))
                iou_score = self.compute_iou(each_mask, aux_logits.sigmoid())

                weighted_query, weight_matrix = self.weight_topk_queries(obj_upd_t.view(b, n, q, c-1), iou_score, topk=8, alpha=1.5, beta=0.8,
                                                                    ignore_others=False)

                query = query + weighted_query.view(b*n, q, c-1)
                event_token = event_token + weighted_query.view(b*n, q, c-1)
                aux_features['logits'].append(aux_logits)

        aux_features['q_weights'] = q_weights  # last layer only
        aux_features['p_weights'] = p_weights  # last layer only
        if self.training:
            # no need to save all heads
            aux_features['attn_mask'] = attn_mask.view(bs, num_objects, self.num_heads,
                                                       self.num_queries, H, W)[:, :, 0]

        return pixel + event, aux_features

    def weight_topk_queries(self,
                            query: torch.Tensor,
                            iou: torch.Tensor,
                            topk: int = 8,
                            alpha: float = 2.0,
                            beta: float = 1.0,
                            ignore_others: bool = False) -> (torch.Tensor, torch.Tensor):
        """
        IoU 스코어가 높은 상위 topk 쿼리에만 가중치를 부여합니다.

        Args:
            query: (B, N, Q, C) - 쿼리 텐서
            iou: (B, N, Q) - 각 쿼리의 IoU 스코어 (0~1 범위)
            topk: 상위 몇 개의 쿼리를 선택할 것인지 (예: 8)
            alpha: 선택된 쿼리에 IoU에 곱할 스케일 팩터
            beta: 기본 가중치 (선택되지 않은 쿼리에도 부여할 기본값)
            ignore_others: True이면 topk 이외의 쿼리는 0으로 처리 (즉, 무시)
                          False이면 topk 이외의 쿼리는 beta를 사용

        Returns:
            weighted_query: (B, N, Q, C) - 각 쿼리에 가중치를 적용한 텐서
            weight_matrix: (B, N, Q) - 각 쿼리에 부여된 가중치
        """
        # 상위 topk IoU 스코어와 해당 인덱스 구하기
        topk_values, topk_indices = torch.topk(iou, k=topk, dim=-1)  # (B, N, topk)

        # weight_matrix 초기화:
        # ignore_others가 True이면 나머지는 0, False이면 beta로 채움
        if ignore_others:
            weight_matrix = torch.zeros_like(iou)
        else:
            weight_matrix = torch.full_like(iou, fill_value=beta)

        # topk 쿼리에 대해 weight = beta + alpha * IoU (상위 쿼리에 부여할 가중치)
        weight_for_topk = beta + alpha * topk_values  # (B, N, topk)

        # torch.scatter_를 이용해 topk 인덱스에 해당하는 위치에 weight_for_topk 값을 넣어줌
        # topk_indices의 shape: (B, N, topk), weight_for_topk도 (B, N, topk)
        weight_matrix.scatter_(dim=2, index=topk_indices, src=weight_for_topk)

        # 가중치를 query에 적용: query의 마지막 차원(C)에 대해 broadcast
        weighted_query = query * weight_matrix.unsqueeze(-1)

        return weighted_query, weight_matrix

    def compute_iou(self,pred_masks: torch.Tensor, target_masks: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        예측 마스크(pred_masks)와 GT 마스크(target_masks) 사이의 IoU를 계산합니다.

        Args:
            pred_masks: (B, N, Q, H, W) - 예측 마스크 (보통 sigmoid 등으로 확률값이거나 이진 마스크)
            target_masks: (B, N, H, W) - GT(ground truth) 마스크 (보통 이진값)
            threshold: 예측 마스크를 이진화할 때 사용할 임계값 (default: 0.5)

        Returns:
            iou: (B, N, Q) - 각 query 마스크에 대해 계산한 IoU 값
        """
        # 예측 마스크 이진화: (B, N, Q, H, W)
        pred_bin = (pred_masks > threshold).float()

        # GT 마스크가 이진값이 아닐 경우 threshold 적용 (이미 이진화된 경우엔 생략 가능)
        target_bin = (target_masks > threshold).float() if target_masks.dtype != torch.bool else target_masks.float()
        # GT 마스크의 shape을 (B, N, 1, H, W)로 만들어서 Q 차원과 broadcast가 되도록 함
        target_bin = target_bin.unsqueeze(2)  # now: (B, N, 1, H, W)

        # Intersection: 두 마스크의 논리적 AND 후 픽셀 수 합산 (H, W 차원 합)
        intersection = (pred_bin * target_bin).sum(dim=[-2, -1])  # (B, N, Q)

        # Union: 두 마스크의 논리적 OR (혹은 합에서 intersection을 빼는 방식)
        union = (pred_bin + target_bin - pred_bin * target_bin).sum(dim=[-2, -1])  # (B, N, Q)

        # IoU 계산 (분모가 0인 경우를 방지하기 위해 작은 상수 추가)
        iou = intersection / (union + 1e-6)

        return iou

    def _get_aux_mask(self, logits: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        # logits: batch_size*num_objects*H*W
        # selector: batch_size*num_objects*1*1
        # returns a mask of shape (batch_size*num_objects*num_heads)*num_queries*(H*W)
        # where True means the attention is blocked

        if selector is None:
            prob = logits.sigmoid()
        else:
            prob = logits.sigmoid() * selector
        logits = aggregate(prob, dim=1)

        is_foreground = (logits[:, 1:] >= logits.max(dim=1, keepdim=True)[0])
        foreground_mask = is_foreground.bool().flatten(start_dim=2)
        inv_foreground_mask = ~foreground_mask
        inv_background_mask = foreground_mask

        aux_foreground_mask = inv_foreground_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)
        aux_background_mask = inv_background_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)

        aux_mask = torch.cat([aux_foreground_mask, aux_background_mask], dim=1)

        aux_mask[torch.where(aux_mask.sum(-1) == aux_mask.shape[-1])] = False

        return aux_mask
