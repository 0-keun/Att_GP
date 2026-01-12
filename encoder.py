import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn

# GP (gpytorch) 필요
import gpytorch

# ---------------------------
# Positional Encoding
# ---------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (T, D)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        x = x + self.pe[:, :t, :].to(dtype=x.dtype, device=x.device)
        return self.dropout(x)


# ---------------------------
# ReZero MHA-only Encoder Layer
# ---------------------------
class ReZeroMHAOnlyEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, attn_dropout: float = 0.0, batch_first: bool = True):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=attn_dropout,
            batch_first=batch_first,
        )
        self.alpha = nn.Parameter(torch.zeros(1))  # ReZero gate (init 0)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_w = self.mha(
            query=x, key=x, value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False if return_attn else True,
        )
        x = x + self.alpha * attn_out
        return (x, attn_w) if return_attn else (x, None)


# ---------------------------
# Encoder + CLS head
#   - forward(..., return_cls=True)로 CLS 임베딩을 직접 반환 가능
# ---------------------------
class ReZeroTransformerEncoder_CLS(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 6,
        out_dim: int = 4,          # 기존 head 출력 차원(원하면 사용), GP용 CLS만 뽑을 수도 있음
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        max_len: int = 10000,
    ):
        super().__init__()
        self.d_model = d_model

        self.embed = nn.Linear(in_dim, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            ReZeroMHAOnlyEncoderLayer(d_model=d_model, nhead=nhead, attn_dropout=attn_dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    @staticmethod
    def build_causal_mask(t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((t, t), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, in_dim)
        causal: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,   # (B, T) True=PAD
        return_attn: bool = False,
        return_cls: bool = False,                 # ✅ GP용: CLS 임베딩 (B, D) 반환 옵션
    ):
        z = self.embed(x)  # (B, T, D)
        b, t, d = z.shape
        cls = self.cls_token.expand(b, 1, d)      # (B, 1, D)
        z = torch.cat([cls, z], dim=1)            # (B, T+1, D)
        z = self.posenc(z)

        attn_mask = self.build_causal_mask(z.size(1), z.device, z.dtype) if causal else None

        kpm = None
        if key_padding_mask is not None:
            cls_pad = torch.zeros((b, 1), dtype=torch.bool, device=key_padding_mask.device)
            kpm = torch.cat([cls_pad, key_padding_mask], dim=1)  # (B, T+1)

        attn_maps = [] if return_attn else None
        for layer in self.layers:
            z, attn_w = layer(z, attn_mask=attn_mask, key_padding_mask=kpm, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn_w)

        z_cls = z[:, 0, :]  # (B, D)

        if return_cls:
            # GP 입력으로 바로 쓰기 좋은 CLS 임베딩
            return (z_cls, attn_maps) if return_attn else (z_cls, None)

        y = self.head(z_cls)  # (B, out_dim)
        return (y, attn_maps) if return_attn else (y, None)


def build_rezero_transformer_encoder_cls(
    in_dim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    out_dim: int,
    dropout: float = 0.1,
    attn_dropout: float = 0.0,
    max_len: int = 10000,
) -> ReZeroTransformerEncoder_CLS:
    return ReZeroTransformerEncoder_CLS(
        in_dim=in_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        out_dim=out_dim,
        dropout=dropout,
        attn_dropout=attn_dropout,
        max_len=max_len,
    )