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

import gpytorch

from encoder import *
from GaussianProcess import *

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
