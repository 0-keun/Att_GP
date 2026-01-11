import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn


# ---------------------------
# Positional Encoding
# ---------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """
    Input : x (B, T, D)
    Output: x + PE (B, T, D)
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


# ---------------------------
# ReZero MHA-only Encoder Layer
# ---------------------------
class ReZeroMHAOnlyEncoderLayer(nn.Module):
    """
    ReZero residual gate:
      x <- x + alpha * MHA(x)
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
        self.alpha = nn.Parameter(torch.zeros(1))  # init 0

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
# Encoder + CLS token head (D -> out_dim)
# ---------------------------
class ReZeroTransformerEncoder_CLS(nn.Module):
    """
    전체 구조:
      Linear embedding -> (prepend CLS) -> PosEnc -> (ReZero MHA-only layer)*L -> take CLS -> Linear -> out_dim

    입력:
      x: (B, T, in_dim)

    출력:
      y: (B, out_dim)
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 6,
        out_dim: int = 4,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        max_len: int = 10000,
    ):
        super().__init__()
        self.d_model = d_model

        self.embed = nn.Linear(in_dim, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)

        # learnable CLS token (1, 1, D)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            ReZeroMHAOnlyEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                attn_dropout=attn_dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        # final projection: CLS(D) -> out_dim
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
        x: torch.Tensor,
        causal: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        key_padding_mask: (B, T)  True=PAD (원본 시계열 길이 기준)
        CLS 토큰이 앞에 붙기 때문에 내부에서는 (B, T+1)로 확장해서 사용.
        """
        # 1) embed
        z = self.embed(x)  # (B, T, D)

        # 2) prepend CLS
        b, t, d = z.shape
        cls = self.cls_token.expand(b, 1, d)  # (B, 1, D)
        z = torch.cat([cls, z], dim=1)        # (B, T+1, D)

        # 3) positional encoding
        z = self.posenc(z)

        # 4) masks
        attn_mask = None
        if causal:
            attn_mask = self.build_causal_mask(z.size(1), z.device, z.dtype)  # (T+1, T+1)

        kpm = None
        if key_padding_mask is not None:
            # CLS는 pad가 아니므로 False를 앞에 붙임
            cls_pad = torch.zeros((b, 1), dtype=torch.bool, device=key_padding_mask.device)
            kpm = torch.cat([cls_pad, key_padding_mask], dim=1)  # (B, T+1)

        # 5) encoder layers
        attn_maps = [] if return_attn else None
        for layer in self.layers:
            z, attn_w = layer(z, attn_mask=attn_mask, key_padding_mask=kpm, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn_w)

        # 6) take CLS and head
        z_cls = z[:, 0, :]           # (B, D)
        y = self.head(z_cls)         # (B, out_dim)

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


# ---------------------------
# (Optional) Train function (regression / classification 둘 다 대응 가능하게)
# ---------------------------
def train_cls_encoder(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    task: str = "regression",   # "regression" or "classification"
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip_norm: float = 1.0,
    causal: bool = False,
    device: Optional[str] = None,
    verbose: bool = True,
):
    """
    train_loader batch formats:
      - (x, y)
      - (x, y, key_padding_mask)

    regression:
      y shape: (B, out_dim), dtype float
      loss: MSELoss

    classification:
      y shape: (B,), dtype long
      loss: CrossEntropyLoss
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    model = model.to(device_t)

    if task not in ("regression", "classification"):
        raise ValueError("task must be 'regression' or 'classification'")

    criterion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    logs = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0

        for batch in train_loader:
            if len(batch) == 2:
                x, y = batch
                kpm = None
            else:
                x, y, kpm = batch

            x = x.to(device_t)
            y = y.to(device_t)
            if kpm is not None:
                kpm = kpm.to(device_t)

            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(x, causal=causal, key_padding_mask=kpm, return_attn=False)

            loss = criterion(pred, y)
            loss.backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

            total_loss += float(loss.item()) * x.size(0)
            total_n += int(x.size(0))

        train_loss = total_loss / max(total_n, 1)
        logs["train_loss"].append(train_loss)

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vloss, vn = 0.0, 0
                for batch in val_loader:
                    if len(batch) == 2:
                        x, y = batch
                        kpm = None
                    else:
                        x, y, kpm = batch
                    x = x.to(device_t)
                    y = y.to(device_t)
                    if kpm is not None:
                        kpm = kpm.to(device_t)

                    pred, _ = model(x, causal=causal, key_padding_mask=kpm, return_attn=False)
                    loss = criterion(pred, y)
                    vloss += float(loss.item()) * x.size(0)
                    vn += int(x.size(0))
                val_loss = vloss / max(vn, 1)
            logs["val_loss"].append(val_loss)

            if verbose:
                print(f"[{ep:03d}/{epochs}] train loss={train_loss:.6f} | val loss={val_loss:.6f}")
        else:
            if verbose:
                print(f"[{ep:03d}/{epochs}] train loss={train_loss:.6f}")

    return model, logs


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Example: 4채널 * 250 timestep -> out_dim=4 (회귀라고 가정)
    B, T, C = 128, 250, 4
    out_dim = 4

    model = build_rezero_transformer_encoder_cls(
        in_dim=C,
        d_model=32,
        nhead=4,
        num_layers=6,
        out_dim=out_dim,
        dropout=0.1,
        attn_dropout=0.0,
        max_len=1000,  # T+1 포함해서 충분히 크게
    )

    # dummy dataset (regression)
    x = torch.randn(B, T, C)
    y = torch.randn(B, out_dim)

    ds = torch.utils.data.TensorDataset(x, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    trained_model, logs = train_cls_encoder(
        model=model,
        train_loader=dl,
        val_loader=None,
        task="regression",
        epochs=3,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        causal=False,
        device=None,
        verbose=True,
    )

    # inference
    x_test = torch.randn(8, T, C)
    pred, _ = trained_model(x_test)
    print("pred shape:", pred.shape)  # (8, 4)
