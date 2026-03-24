from typing import List, Dict, Optional
from omegaconf import DictConfig

from cutie.model.transformer.positional_encoding import PositionalEncoding

import torch
import torch.nn as nn
import torch.nn.functional as F

# class SlotAttention(nn.Module):
#     def __init__(self, n_slots, slot_dim, iters=3, n_heads=8):
#         """
#         Args:
#             n_slots: 생성할 슬롯의 개수
#             slot_dim: 슬롯 임베딩의 차원
#             iters: 슬롯 갱신 반복 횟수
#             n_heads: Attention head의 개수
#         """
#         super().__init__()
#         self.n_slots = n_slots
#         self.slot_dim = slot_dim
#         self.iters = iters
#         self.n_heads = n_heads

#         # 슬롯 초기화를 위한 learnable parameters
#         self.slot_mu = nn.Parameter(torch.randn(1, n_slots, slot_dim))
#         self.slot_sigma = nn.Parameter(torch.randn(1, n_slots, slot_dim))

#         # Linear projections for multi-head Q, K, V
#         self.q_proj = nn.Linear(slot_dim, slot_dim)
#         self.k_proj = nn.Linear(slot_dim, slot_dim)
#         self.v_proj = nn.Linear(slot_dim, slot_dim)

#         # Simple MLP for slot updates
#         self.mlp = nn.Sequential(
#             nn.Linear(slot_dim, slot_dim),
#             nn.ReLU(),
#             nn.Linear(slot_dim, slot_dim),
#         )

#         # LayerNorms and GRU for update steps
#         self.norm_inputs = nn.LayerNorm(slot_dim)
#         self.norm_slots = nn.LayerNorm(slot_dim)
#         self.norm_mlp = nn.LayerNorm(slot_dim)
#         self.gru = nn.GRUCell(slot_dim, slot_dim)

#     def forward(self, feats: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             feats: (B * num_objects, H * W, C)
#         Returns:
#             slots: (B * num_objects, n_slots, C)
#         """
#         Bn, F_dim, C = feats.shape  # F_dim: H*W or num_features

#         # Slot Initialization
#         mu = self.slot_mu.expand(Bn, -1, -1)
#         sigma = torch.nn.functional.softplus(self.slot_sigma).expand_as(mu)
#         slots = mu + sigma * torch.randn_like(mu)

#         for _ in range(self.iters):
#             # 이전 슬롯 보관
#             slots_prev = slots

#             # 입력과 슬롯에 LayerNorm 적용
#             feats_norm = self.norm_inputs(feats)
#             slots_norm = self.norm_slots(slots)

#             # Multi-head Q, K, V computation
#             q = self.q_proj(slots_norm).view(Bn, self.n_slots, self.n_heads, C // self.n_heads)
#             q = q.transpose(2, 3)  # (Bn, n_slots, n_heads, head_dim)
#             k = self.k_proj(feats_norm).view(Bn, F_dim, self.n_heads, C // self.n_heads)
#             k = k.transpose(2, 3)  # (Bn, F_dim, n_heads, head_dim)
#             v = self.v_proj(feats_norm).view(Bn, F_dim, self.n_heads, C // self.n_heads)
#             v = v.transpose(2, 3)  # (Bn, F_dim, n_heads, head_dim)

#             # Attention computation
#             attn_logits = torch.einsum('bqhc,bfhc->bhqf', q, k) / (C // self.n_heads) ** 0.5
#             attn = F.softmax(attn_logits, dim=2)
#             attn = attn / attn.sum(dim=-1, keepdim)

#             # Weighted sum for new slot updates
#             updates = torch.einsum('bhqf,bfhc->bqhc', attn, v)
#             updates = updates.transpose(2, 3).reshape(Bn, self.n_slots, C)

#             # GRU-based slot update
#             # GRUCell expects inputs of shape (batch_size, input_size)
#             # 여기서 batch_size = Bn * n_slots, input_size = slot_dim(C)
#             gru_input = updates.reshape(Bn * self.n_slots, C)
#             gru_hidden = slots_prev.reshape(Bn * self.n_slots, C)
#             updated_slots = self.gru(gru_input, gru_hidden)
#             slots = updated_slots.reshape(Bn, self.n_slots, C)

#             # MLP 업데이트 전에 LayerNorm 적용
#             slots = slots + self.mlp(self.norm_mlp(slots))

#         return slots

# @torch.jit.script
def _weighted_pooling(masks: torch.Tensor, value: torch.Tensor,
                      logits: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    # value: B*num_objects*H*W*value_dim
    # logits: B*num_objects*H*W*num_summaries
    # masks: B*num_objects*H*W*num_summaries: 1 if allowed
    weights = logits.sigmoid() * masks
    # B*num_objects*num_summaries*value_dim
    sums = torch.einsum('bkhwq,bkhwc->bkqc', weights, value)
    # B*num_objects*H*W*num_summaries -> B*num_objects*num_summaries*1
    area = weights.flatten(start_dim=2, end_dim=3).sum(2).unsqueeze(-1)

    # B*num_objects*num_summaries*value_dim
    return sums, area



# class ObjectSummarizer(nn.Module):
#     def __init__(self, model_cfg: DictConfig):
#         super().__init__()

#         this_cfg = model_cfg.object_summarizer
#         self.value_dim = model_cfg.value_dim
#         self.embed_dim = this_cfg.embed_dim
#         self.num_summaries = this_cfg.num_summaries
#         self.add_pe = this_cfg.add_pe
#         self.pixel_pe_scale = model_cfg.pixel_pe_scale
#         self.pixel_pe_temperature = model_cfg.pixel_pe_temperature

#         if self.add_pe:
#             self.pos_enc = PositionalEncoding(self.embed_dim,
#                                               scale=self.pixel_pe_scale,
#                                               temperature=self.pixel_pe_temperature)

#         self.input_proj = nn.Linear(self.value_dim, self.embed_dim)
#         self.feature_pred = nn.Sequential(
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.embed_dim, self.embed_dim),
#         )
#         # self.weights_pred = nn.Sequential(
#         #     nn.Linear(self.embed_dim, self.embed_dim),
#         #     nn.ReLU(inplace=True),
#         #     nn.Linear(self.embed_dim, self.num_summaries),
#         # )

#         self.slot = SlotAttention(n_slots=16, n_heads=8, slot_dim=256)

#     def forward(self,
#                 masks: torch.Tensor,
#                 value: torch.Tensor,
#                 need_weights: bool = False) -> (torch.Tensor, Optional[torch.Tensor]):
#         # masks: B*num_objects*(H0)*(W0)
#         # value: B*num_objects*value_dim*H*W
#         # -> B*num_objects*H*W*value_dim
#         h, w = value.shape[-2:]
#         masks = F.interpolate(masks, size=(h, w), mode='area')
#         masks = masks.unsqueeze(-1)
#         inv_masks = 1 - masks
#         repeated_masks = torch.cat([
#             masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
#             inv_masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
#         ],
#                                    dim=-1)

#         value = value.permute(0, 1, 3, 4, 2)
#         value = self.input_proj(value)
#         if self.add_pe:
#             pe = self.pos_enc(value)
#             value = value + pe

#         with torch.cuda.amp.autocast(enabled=False):
#             value = value.float()
#             feature = self.feature_pred(value) * masks
#             b, n, h, w, c = feature.shape
#             feature = feature.view(b * n, h*w, c)
#             summaries = self.slot(feature)

#         bn, q, c = summaries.shape
#         summaries = summaries.view(b, n, q, c)

#         if need_weights:
#             return summaries, None
#         else:
#             return summaries, None


class ObjectSummarizer(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_summarizer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_summaries = this_cfg.num_summaries
        self.add_pe = this_cfg.add_pe
        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature

        if self.add_pe:
            self.pos_enc = PositionalEncoding(self.embed_dim,
                                              scale=self.pixel_pe_scale,
                                              temperature=self.pixel_pe_temperature)

        self.input_proj = nn.Linear(self.value_dim, self.embed_dim)
        self.feature_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.weights_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.num_summaries),
        )

    def forward(self,
                masks: torch.Tensor,
                value: torch.Tensor,
                need_weights: bool = False) -> (torch.Tensor, Optional[torch.Tensor]):
        # masks: B*num_objects*(H0)*(W0)
        # value: B*num_objects*value_dim*H*W
        # -> B*num_objects*H*W*value_dim
        h, w = value.shape[-2:]
        masks = F.interpolate(masks, size=(h, w), mode='area')
        masks = masks.unsqueeze(-1)
        inv_masks = 1 - masks
        repeated_masks = torch.cat([
            masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
            masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
        ],
                                   dim=-1)

        value = value.permute(0, 1, 3, 4, 2)
        value = self.input_proj(value)
        if self.add_pe:
            pe = self.pos_enc(value)
            value = value + pe

        with torch.cuda.amp.autocast(enabled=False):
            value = value.float()
            feature = self.feature_pred(value)
            logits = self.weights_pred(value)
            sums, area = _weighted_pooling(repeated_masks, feature, logits)

        summaries = torch.cat([sums, area], dim=-1)

        if need_weights:
            return summaries, logits
        else:
            return summaries, None

class ObjectSummarizer2(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_summarizer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_summaries = this_cfg.num_summaries
        self.add_pe = this_cfg.add_pe
        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature

        if self.add_pe:
            self.pos_enc = PositionalEncoding(self.embed_dim,
                                              scale=self.pixel_pe_scale,
                                              temperature=self.pixel_pe_temperature)

        self.input_proj = nn.Linear(self.value_dim, self.embed_dim)
        self.feature_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.weights_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.Linear(self.embed_dim, self.num_summaries),
        )

    def forward(self,
                masks: torch.Tensor,
                value: torch.Tensor,
                need_weights: bool = False) -> (torch.Tensor, Optional[torch.Tensor]):
        # masks: B*num_objects*(H0)*(W0)
        # value: B*num_objects*value_dim*H*W
        # -> B*num_objects*H*W*value_dim
        h, w = value.shape[-2:]
        masks = F.interpolate(masks, size=(h, w), mode='area')
        masks = masks.unsqueeze(-1)
        inv_masks = 1 - masks
        repeated_masks = torch.cat([
            masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
            masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
        ],
                                   dim=-1)

        value = value.permute(0, 1, 3, 4, 2)
        value = self.input_proj(value)
        if self.add_pe:
            pe = self.pos_enc(value)
            value = value + pe

        with torch.cuda.amp.autocast(enabled=False):
            value = value.float()
            feature = self.feature_pred(value)
            logits = self.weights_pred(value)
            sums, area = _weighted_pooling(repeated_masks, feature, logits)

        summaries = torch.cat([sums, area], dim=-1)

        if need_weights:
            return summaries, logits
        else:
            return summaries, None
class EventSummarizer(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_summarizer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_summaries = this_cfg.num_summaries
        self.add_pe = this_cfg.add_pe
        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature

        if self.add_pe:
            self.pos_enc = PositionalEncoding(self.embed_dim,
                                              scale=self.pixel_pe_scale,
                                              temperature=self.pixel_pe_temperature)

        self.input_proj = nn.Linear(self.value_dim, self.embed_dim)
        self.feature_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.weights_pred = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.num_summaries),
        )

    def forward(self,
                masks: torch.Tensor,
                value: torch.Tensor,
                need_weights: bool = False) -> (torch.Tensor, Optional[torch.Tensor]):
        # masks: B*num_objects*(H0)*(W0)
        # value: B*num_objects*value_dim*H*W
        # -> B*num_objects*H*W*value_dim
        h, w = value.shape[-2:]
        masks = F.interpolate(masks, size=(h, w), mode='area')
        masks = masks.unsqueeze(-1)
        masks = masks.expand(-1, -1, -1, -1, self.num_summaries)
        value = value.permute(0, 1, 3, 4, 2)
        value = self.input_proj(value)
        if self.add_pe:
            pe = self.pos_enc(value)
            value = value + pe

        with torch.cuda.amp.autocast(enabled=False):
            value = value.float()
            feature = self.feature_pred(value)
            logits = self.weights_pred(value)
            sums, area = _weighted_pooling(masks, feature, logits)

        summaries = torch.cat([sums, area], dim=-1)

        if need_weights:
            return summaries, logits
        else:
            return summaries, None