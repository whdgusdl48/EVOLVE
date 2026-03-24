import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def get_emb(sin_inp):
    return torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self,
                 dim: int,
                 scale: float = math.pi * 2,
                 temperature: float = 10000,
                 normalize: bool = True,
                 channel_last: bool = False,
                 transpose_output: bool = False):
        super().__init__()
        dim = int(np.ceil(dim / 4) * 2)
        self.dim = dim
        inv_freq = 1.0 / (temperature ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.normalize = normalize
        self.scale = scale
        self.eps = 1e-6
        self.channel_last = channel_last
        self.transpose_output = transpose_output

        self.cached_penc = None  # the cache is irrespective of the number of objects

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: A 4/5d tensor of size
            channel_last=True: (batch_size, h, w, c) or (batch_size, k, h, w, c)
            channel_last=False: (batch_size, c, h, w) or (batch_size, k, c, h, w)
        :return: positional encoding tensor that has the same shape as the input if the input is 4d
                 if the input is 5d, the output is broadcastable along the k-dimension
        """
        if len(tensor.shape) != 4 and len(tensor.shape) != 5:
            raise RuntimeError(f'The input tensor has to be 4/5d, got {tensor.shape}!')

        if len(tensor.shape) == 5:
            # take a sample from the k dimension
            num_objects = tensor.shape[1]
            tensor = tensor[:, 0]
        else:
            num_objects = None

        if self.channel_last:
            batch_size, h, w, c = tensor.shape
        else:
            batch_size, c, h, w = tensor.shape

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            if num_objects is None:
                return self.cached_penc
            else:
                return self.cached_penc.unsqueeze(1)

        self.cached_penc = None

        pos_y = torch.arange(h, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_x = torch.arange(w, device=tensor.device, dtype=self.inv_freq.dtype)
        if self.normalize:
            pos_y = pos_y / (pos_y[-1] + self.eps) * self.scale
            pos_x = pos_x / (pos_x[-1] + self.eps) * self.scale

        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_x = get_emb(sin_inp_x)

        emb = torch.zeros((h, w, self.dim * 2), device=tensor.device, dtype=tensor.dtype)
        emb[:, :, :self.dim] = emb_x
        emb[:, :, self.dim:] = emb_y

        if not self.channel_last and self.transpose_output:
            # cancelled out
            pass
        elif (not self.channel_last) or (self.transpose_output):
            emb = emb.permute(2, 0, 1)

        self.cached_penc = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        if num_objects is None:
            return self.cached_penc
        else:
            return self.cached_penc.unsqueeze(1)


class CrossAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_heads=8):
        super(CrossAttentionModule, self).__init__()

        # 채널을 256으로 맞추기 위한 Conv 레이어
        self.img_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.event_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)

        # LayerNorm 추가
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.SiLU(),
            nn.Linear(out_channels * 2, out_channels * 2),
            nn.SiLU(),
            nn.Linear(out_channels * 2, out_channels)
        )

        # 채널을 원래 크기로 되돌리는 Conv 레이어
        self.img_conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=1)

        # Positional Encoding 추가
        self.pos_encoding = PositionalEncoding(dim=out_channels)

    def forward(self, img_feat, event_feat):
        B, C, H, W = img_feat.shape

        # Step 1: 채널을 256으로 맞추기
        img_feat = self.img_conv1(img_feat)  # B x 256 x H x W
        event_feat = self.event_conv1(event_feat)  # B x 256 x H x W

        # Step 2: Positional Encoding 추가
        pos_enc = self.pos_encoding(img_feat)
        img_feat = img_feat + pos_enc
        event_feat = event_feat + pos_enc

        # Step 3: Cross Attention
        img_feat_flatten = img_feat.flatten(2).permute(0, 2, 1)  # B x HW x 256
        event_feat_flatten = event_feat.flatten(2).permute(0, 2, 1)  # B x HW x 256

        attn_output, _ = self.attention(img_feat_flatten,
                                        event_feat_flatten,
                                        event_feat_flatten) 
        attn_output = self.norm1(attn_output +img_feat_flatten)   # LayerNorm 적용

        # Reshape back to spatial dimensions
          # B x 256 x H x W
        # Step 4: FFN 적용
        attn_output2 = self.ffn(attn_output)
        attn_output = self.norm2(attn_output + attn_output2)  # LayerNorm 적용
        attn_output = attn_output.permute(0, 2, 1).reshape(B, -1, H, W)
        # Step 5: 원래 채널 차원으로 복원
        output = self.img_conv2(attn_output)  # B x C x H x W

        return output


# 테스트 예제
if __name__ == "__main__":
    img_feat = torch.randn(4, 128, 32, 32)  # Batch=4, Channels=128, H=32, W=32
    event_feat = torch.randn(4, 128, 32, 32)

    model = CrossAttentionModule(in_channels=128, out_channels=256)
    output = model(img_feat, event_feat)
    print(output.shape)  # 예상: torch.Size([4, 128, 32, 32])
