import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn

# GP (gpytorch) 필요
import gpytorch

from encoder import *
from GaussianProcess import *
from train import *
from test import *

from preprocessing import DataFromCSV



# ============================================================
# relabel -> binary
# ============================================================

def relabel_one_vs_rest(
    dataset: DatasetTensors,
    target_label_idx: int,
) -> DatasetTensors:
    """
    특정 클래스(target_label_idx)만 label=0,
    나머지는 전부 label=1 로 binary relabel.

    입력:
        dataset.y : (N,)  기존 label (0~7)
        target_label_idx : int (예: open1에 해당하는 index)

    출력:
        DatasetTensors (binary label)
          y == 0 : target class
          y == 1 : all others
    """
    y_old = dataset.y
    y_new = torch.ones_like(y_old)          # default = 1
    y_new[y_old == target_label_idx] = 0    # target만 0

    return DatasetTensors(
        X=dataset.X,
        y=y_new,
        file_ids=dataset.file_ids,
        end_rows=dataset.end_rows,
    )

# ============================================================
# 예시: "Class1 vs Class2" binary GP on CLS
# ============================================================
if __name__ == "__main__":
    # (1) Encoder 만들기
    T, C = 2000, 250, 4
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
    data = DataFromCSV(csv_dir="./dataset_OBC/dataset_train_251023", device="cpu")
    data.load_all_csv_sliding(window=250,stride=10,drop_last_incomplete=True,label_source="output_cols",verbose=True,)
    all_data = data.concat_all_datasets()
    all_data_bin = relabel_one_vs_rest(all_data,0)

    x_all = all_data_bin.X
    y_all = all_data_bin.y

    # (3) CLS 임베딩 추출
    cls_feat = extract_cls_features(encoder, x_all, batch_size=128, device=None)  # (N, 32)

    # (4) GP binary classifier 학습
    gp_model, gp_lik, gp_logs = train_gp_binary_on_cls(
        cls_train=cls_feat,
        y_train=y_all,
        num_inducing=64,
        epochs=200,
        lr=0.01,
        device=None,
        verbose=True,
    )

    # (5) 확률 예측
    p1 = gp_binary_predict_proba(gp_model, gp_lik, cls_feat[:10])
    print("p(y=1) first 10:", p1.numpy())
