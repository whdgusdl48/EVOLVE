# Modified from PyTorch nn.Transformer

from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from cutie.model.channel_attn import CAResBlock


import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import torchvision.transforms as transforms
from einops import rearrange
import numbers

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class EVFModule(nn.Module):
    def __init__(self, dim, num_heads):
        super(EVFModule, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.norm1 = LayerNorm(dim,'With')
        self.norm2 = LayerNorm(dim,'With')
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.pixel_d_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)
        self.event_d_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.event_d_conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.pixel_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.event_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.event_conv2 = nn.Conv2d(dim, dim, kernel_size=1)

        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1)

        self.ch_p = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

        self.final_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
    def forward(self, pixel, event):
        bs, num_objects, _, h, w = pixel.shape
        pixel_flat = pixel.view(bs * num_objects,self.dim, h, w)
        event_flat = event.view(bs * num_objects,self.dim, h, w)

        pixel_flat = self.norm1(pixel_flat)
        event_flat = self.norm2(event_flat)

        q = self.pixel_conv(self.pixel_d_conv(pixel_flat))
        k = self.event_conv(self.event_d_conv(event_flat))
        v = self.event_conv2(self.event_d_conv2(event_flat))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q,p=2, dim=-1)
        k = torch.nn.functional.normalize(k,p=2, dim=-1)

        # hw 방향으로 평균 계산

        attn_feature = (q @ k.transpose(-2, -1)) * self.temperature
        attn_feature = attn_feature.softmax(dim=-1)
        out = (attn_feature @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.out_conv(out)
        
        # spatial attention

        ch_p = self.ch_p(pixel_flat)  # [B*num_objects, C, H, W]
        out_p = ch_p + out  # [B*num_objects, C, H, W]
        
        # Final Convolution and Activation
        out = F.gelu(self.final_conv(out_p))  # [B*num_objects, C, H, W]
        
        return out


class SelfAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 nhead: int,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 add_pe_to_qkv: List[bool] = [True, True, False]):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv

    def forward(self,
                x: torch.Tensor,
                pe: torch.Tensor,
                attn_mask: bool = None,
                key_padding_mask: bool = None) -> torch.Tensor:
        x = self.norm(x)
        if any(self.add_pe_to_qkv):
            x_with_pe = x + pe
            q = x_with_pe if self.add_pe_to_qkv[0] else x
            k = x_with_pe if self.add_pe_to_qkv[1] else x
            v = x_with_pe if self.add_pe_to_qkv[2] else x
        else:
            q = k = v = x

        r = x
        x = self.self_attn(q, k, v, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        return r + self.dropout(x)


# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
class CrossAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 nhead: int,
                 dropout: float = 0.0,
                 batch_first: bool = True,
                 add_pe_to_qkv: List[bool] = [True, True, False],
                 residual: bool = True,
                 norm: bool = True):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim,
                                                nhead,
                                                dropout=dropout,
                                                batch_first=batch_first)
        if norm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.add_pe_to_qkv = add_pe_to_qkv
        self.residual = residual

    def forward(self,
                x: torch.Tensor,
                mem: torch.Tensor,
                x_pe: torch.Tensor,
                mem_pe: torch.Tensor,
                attn_mask: bool = None,
                *,
                need_weights: bool = False) -> (torch.Tensor, torch.Tensor):
        x = self.norm(x)
        if self.add_pe_to_qkv[0]:
            q = x + x_pe
        else:
            q = x

        if any(self.add_pe_to_qkv[1:]):
            mem_with_pe = mem + mem_pe
            k = mem_with_pe if self.add_pe_to_qkv[1] else mem
            v = mem_with_pe if self.add_pe_to_qkv[2] else mem
        else:
            k = v = mem
        r = x
        x, weights = self.cross_attn(q,
                                     k,
                                     v,
                                     attn_mask=attn_mask,
                                     need_weights=need_weights,
                                     average_attn_weights=False)

        if self.residual:
            return r + self.dropout(x), weights
        else:
            return self.dropout(x), weights

class FFN(nn.Module):
    def __init__(self, dim_in: int, dim_ff: int, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_ff)
        self.linear2 = nn.Linear(dim_ff, dim_in)
        self.norm = nn.LayerNorm(dim_in)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm(x)
        x = self.linear2(self.activation(self.linear1(x)))
        x = r + x
        return x


class PixelFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.conv = CAResBlock(dim, dim)

    def forward(self, pixel: torch.Tensor, pixel_flat: torch.Tensor) -> torch.Tensor:
        # pixel: batch_size * num_objects * dim * H * W
        # pixel_flat: (batch_size*num_objects) * (H*W) * dim
        bs, num_objects, _, h, w = pixel.shape
        pixel_flat = pixel_flat.view(bs * num_objects, h, w, self.dim)
        pixel_flat = pixel_flat.permute(0, 3, 1, 2).contiguous()

        x = self.conv(pixel_flat)
        x = x.view(bs, num_objects, self.dim, h, w)
        return x


class OutputFFN(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation=F.relu):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_out, dim_out)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
