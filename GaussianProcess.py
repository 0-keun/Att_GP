import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn

# GP (gpytorch) 필요
import gpytorch

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