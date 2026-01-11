import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    입력:  x (B, T, D)
    출력:  x + PE (B, T, D)
    """
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


class ReZeroMHAOnlyEncoderLayer(nn.Module):
    """
    ReZero 방식(α=0 초기화) residual gate:
      x <- x + α * MHA(x)

    - 이 레이어는 MHA만 포함(FFN 없음)
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        attn_dropout: float = 0.0,
        batch_first: bool = True,
    ):
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
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False if return_attn else True,
        )
        x = x + self.alpha * attn_out
        return (x, attn_w) if return_attn else (x, None)


class ReZeroFinalFFN(nn.Module):
    """
    마지막에만 1회 적용되는 FFN + ReZero residual gate:
      x <- x + alpha * FFN(x)
    """
    def __init__(
        self,
        d_model: int,
        dim_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        ):
        super().__init__()
        act = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        self.alpha = nn.Parameter(torch.zeros(1))  # ReZero gate (init 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.ffn(x)


class ReZeroTransformerEncoder_LastFFN(nn.Module):
    """
    전체 구조:
      Linear embedding -> Positional Encoding -> (MHA-only ReZero layer) x L -> Final FFN(ReZero) 1회

    입력:
      x: (B, T, in_dim)

    출력:
      y: (B, T, d_model)
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 6,
        dim_ff: int = 128,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        max_len: int = 10000,
        activation: str = "gelu",
    ):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        self.layers = nn.ModuleList([
            ReZeroMHAOnlyEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                attn_dropout=attn_dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.final_ffn = ReZeroFinalFFN(
            d_model=d_model,
            dim_ff=dim_ff,
            dropout=dropout,
            activation=activation,
        )

    @staticmethod
    def build_causal_mask(t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # (T,T)에서 미래 토큰을 -inf로 막는 causal mask
        mask = torch.full((t, t), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x = self.embed(x)      # (B, T, d_model)
        x = self.posenc(x)     # (B, T, d_model)

        attn_mask = None
        if causal:
            attn_mask = self.build_causal_mask(x.size(1), x.device, x.dtype)

        attn_maps = [] if return_attn else None
        for layer in self.layers:
            x, attn_w = layer(
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                return_attn=return_attn,
            )
            if return_attn:
                attn_maps.append(attn_w)

        x = self.final_ffn(x)  # 마지막에만 FFN 1회
        return (x, attn_maps) if return_attn else (x, None)


def build_rezero_transformer_encoder_last_ffn(
    in_dim: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_ff: int,
    dropout: float = 0.1,
    attn_dropout: float = 0.0,
    max_len: int = 10000,
    activation: str = "gelu",):

    return ReZeroTransformerEncoder_LastFFN(
        in_dim=in_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        attn_dropout=attn_dropout,
        max_len=max_len,
        activation=activation,
    )


if __name__ == "__main__":
    # 예시: 4채널 * 250 timestep
    B, T, C = 8, 250, 4
    model = build_rezero_transformer_encoder_last_ffn(
        in_dim=C,
        d_model=32,
        nhead=4,
        num_layers=6,
        dim_ff=128,
        dropout=0.1,
        attn_dropout=0.0,
        max_len=1000,
        activation="gelu",
    )
    x = torch.randn(B, T, C)
    y, _ = model(x, causal=False, return_attn=False)
    print(y.shape)  # (8, 250, 32)
