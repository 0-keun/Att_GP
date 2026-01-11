# ============================================================
# 1) (이미 만든) ReZeroTransformerEncoder_CLS 모델을 그대로 사용
#    - 단, "CLS 임베딩 (B, d_model)"을 뽑는 함수/옵션을 추가
# 2) CLS 임베딩을 입력으로 받는 GP Binary Classifier 추가 (gpytorch)
#    - Variational GP + Bernoulli likelihood (inducing points 사용)
# 3) 바로 붙여넣기 가능한 "전체 기능" 코드 (Encoder + CLS feature + GP 학습/추론)
# ============================================================

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


# ============================================================
# GP Binary Classification (CLS 임베딩 -> Bernoulli GP)
# ============================================================

class VariationalGPBinaryClassifier(gpytorch.models.ApproximateGP):
    """
    입력:  X (N, D)  (여기서 D = d_model, 즉 CLS 임베딩 차원)
    출력:  latent f(X) ~ GP
    likelihood(Bernoulli)로 binary classification 수행
    """
    def __init__(self, inducing_points: torch.Tensor):
        # inducing_points: (M, D)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@torch.no_grad()
def extract_cls_features(
    encoder: nn.Module,
    x: torch.Tensor,                               # (N,T,C)
    key_padding_mask: Optional[torch.Tensor] = None,# (N,T) True=PAD
    batch_size: int = 256,
    causal: bool = False,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    encoder에서 CLS 임베딩만 뽑아서 (N, d_model) 반환
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    encoder.eval().to(device_t)

    feats = []
    n = x.size(0)
    for i in range(0, n, batch_size):
        xb = x[i:i+batch_size].to(device_t)
        kpm = None
        if key_padding_mask is not None:
            kpm = key_padding_mask[i:i+batch_size].to(device_t)

        cls_b, _ = encoder(xb, causal=causal, key_padding_mask=kpm, return_attn=False, return_cls=True)
        feats.append(cls_b.detach().cpu())

    return torch.cat(feats, dim=0)  # (N, d_model)


def train_gp_binary_on_cls(
    cls_train: torch.Tensor,        # (N, D) float
    y_train: torch.Tensor,          # (N,) 0/1 long or float
    num_inducing: int = 64,
    epochs: int = 200,
    lr: float = 0.01,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[VariationalGPBinaryClassifier, gpytorch.likelihoods.BernoulliLikelihood, Dict[str, list]]:
    """
    CLS 임베딩 기반 GP binary classifier 학습.
    - inducing point는 train feature에서 랜덤으로 뽑아 초기화
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    X = cls_train.to(device_t).float()
    y = y_train.to(device_t)
    if y.dtype != torch.float32:
        y = y.float()  # Bernoulli likelihood에서 float도 OK (0/1)

    n = X.size(0)
    m = min(num_inducing, n)
    idx = torch.randperm(n, device=device_t)[:m]
    inducing = X[idx].clone()

    model = VariationalGPBinaryClassifier(inducing_points=inducing).to(device_t)
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device_t)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ], lr=lr)

    # Variational ELBO objective
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)

    logs = {"loss": []}

    for ep in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(X)                 # latent f(X)
        loss = -mll(output, y)            # maximize ELBO -> minimize negative
        loss.backward()
        optimizer.step()

        l = float(loss.item())
        logs["loss"].append(l)
        if verbose and (ep == 1 or ep % 20 == 0 or ep == epochs):
            print(f"[GP {ep:03d}/{epochs}] loss={l:.6f}")

    return model, likelihood, logs


@torch.no_grad()
def gp_binary_predict_proba(
    gp_model: VariationalGPBinaryClassifier,
    likelihood: gpytorch.likelihoods.BernoulliLikelihood,
    cls_x: torch.Tensor,          # (N, D)
    batch_size: int = 512,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    반환: p(y=1|x) (N,)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    gp_model.eval().to(device_t)
    likelihood.eval().to(device_t)

    probs = []
    n = cls_x.size(0)
    for i in range(0, n, batch_size):
        xb = cls_x[i:i+batch_size].to(device_t).float()
        with gpytorch.settings.fast_pred_var(True):
            pred = likelihood(gp_model(xb))  # Bernoulli distribution
            # gpytorch BernoulliLikelihood returns Bernoulli distribution with probs
            probs.append(pred.probs.detach().cpu().view(-1))
    return torch.cat(probs, dim=0)


# ============================================================
# 예시: "Class1 vs Class2" binary GP on CLS
# ============================================================
if __name__ == "__main__":
    # (1) Encoder 만들기
    B, T, C = 2000, 250, 4
    d_model = 32

    encoder = build_rezero_transformer_encoder_cls(
        in_dim=C,
        d_model=d_model,
        nhead=4,
        num_layers=6,
        out_dim=4,        # head는 여기선 안 써도 됨
        dropout=0.1,
        attn_dropout=0.0,
        max_len=2000,
    )

    # (2) 더미 데이터: 3-class 중 class 1 vs class 2만 골라 binary로 만든다고 가정
    x_all = torch.randn(B, T, C)
    y_all = torch.randint(0, 3, (B,), dtype=torch.long)

    mask_12 = (y_all == 1) | (y_all == 2)
    x_12 = x_all[mask_12]
    y_12 = y_all[mask_12]
    y_bin = (y_12 == 2).long()  # class2를 1, class1을 0

    # (3) CLS 임베딩 추출
    cls_feat = extract_cls_features(encoder, x_12, batch_size=128, device=None)  # (N, 32)

    # (4) GP binary classifier 학습
    gp_model, gp_lik, gp_logs = train_gp_binary_on_cls(
        cls_train=cls_feat,
        y_train=y_bin,
        num_inducing=64,
        epochs=200,
        lr=0.01,
        device=None,
        verbose=True,
    )

    # (5) 확률 예측
    p1 = gp_binary_predict_proba(gp_model, gp_lik, cls_feat[:10])
    print("p(y=1) first 10:", p1.numpy())
