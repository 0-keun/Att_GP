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


