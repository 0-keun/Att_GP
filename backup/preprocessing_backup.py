import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch


output_list = ["open1","open2","open3","open4","short1","short2","short3","short4"]
input_list  = ["Vo:Measured voltage","IL:Measured current"]


@dataclass
class DatasetTensors:
    X: torch.Tensor           # (N, T, C)
    y: torch.Tensor           # (N,)  (class index)
    file_ids: List[str]       # (N,)  (원본 파일명)
    end_rows: torch.Tensor    # (N,)  (각 window의 마지막 row index)


class DataFromCSV:
    """
    폴더의 모든 csv를 읽고, 각 파일에서 sliding window로 (T,C) 샘플들을 생성.
    라벨 y는 "각 window의 마지막 row"에서 결정.
    - input: input_list 컬럼 -> X window
    - label: output_list 컬럼(또는 label_col)을 window 마지막 row에서 읽어 y 생성

    접근:
      data.class_name["open1"].X  # (N_open1, T, C)
      data.class_name["open1"].y  # (N_open1,)
    """
    def __init__(
        self,
        csv_dir: str,
        input_list: List[str] = input_list,
        output_list: List[str] = output_list,
        device: str = "cpu",
        x_dtype: torch.dtype = torch.float32,
    ):
        self.csv_dir = csv_dir
        self.input_list = input_list
        self.output_list = output_list
        self.device = torch.device(device)
        self.x_dtype = x_dtype

        self.label2idx: Dict[str, int] = {name: i for i, name in enumerate(self.output_list)}
        self.idx2label: Dict[int, str] = {i: name for name, i in self.label2idx.items()}

        # 최종 저장소
        self.class_name: Dict[str, DatasetTensors] = {}

    # -------------------------
    # 핵심: 폴더 전체 로딩 (슬라이딩 윈도우)
    # -------------------------
    def load_all_csv_sliding(
        self,
        window: int = 250,
        stride: int = 1,
        drop_last_incomplete: bool = True,
        label_source: str = "output_cols",     # "output_cols" | "label_col"
        label_col: str = "label",
        file_filter: Optional[str] = None,     # 예: "open" 넣으면 open* 파일만
        sort_files: bool = True,
        verbose: bool = True,
    ) -> None:
        files = [f for f in os.listdir(self.csv_dir) if f.lower().endswith(".csv")]
        if file_filter:
            files = [f for f in files if file_filter in f]
        if sort_files:
            files.sort()

        if verbose:
            print(f"[DataFromCSV] csv_dir={self.csv_dir} | files={len(files)} | window={window}, stride={stride}")

        # 누적 버퍼 (클래스별로 모았다가 마지막에 concat)
        buf: Dict[str, List[Tuple[torch.Tensor, int, str, int]]] = {k: [] for k in self.output_list}

        for fname in files:
            fpath = os.path.join(self.csv_dir, fname)
            df = pd.read_csv(fpath)

            Xw, yw, end_rows = self._make_sliding_windows_from_df(
                df=df,
                window=window,
                stride=stride,
                drop_last_incomplete=drop_last_incomplete,
                label_source=label_source,
                label_col=label_col,
            )
            # Xw: (Nw, T, C), yw: (Nw,), end_rows: (Nw,)

            for i in range(Xw.shape[0]):
                y_idx = int(yw[i].item())
                if y_idx < 0:
                    continue
                label = self.idx2label[y_idx]
                buf[label].append((Xw[i], y_idx, fname, int(end_rows[i].item())))

        # buf -> class_name
        self.class_name = {}
        for label, items in buf.items():
            if len(items) == 0:
                continue
            X_list, y_list, fid_list, end_list = zip(*items)

            X_cat = torch.stack(X_list, dim=0).to(self.device, dtype=self.x_dtype)       # (N,T,C)
            y_cat = torch.tensor(y_list, dtype=torch.long, device=self.device)           # (N,)
            end_rows = torch.tensor(end_list, dtype=torch.long, device=self.device)      # (N,)

            self.class_name[label] = DatasetTensors(
                X=X_cat,
                y=y_cat,
                file_ids=list(fid_list),
                end_rows=end_rows,
            )

        if verbose:
            for k in self.class_name:
                ds = self.class_name[k]
                print(f"  - {k}: X={tuple(ds.X.shape)}, y={tuple(ds.y.shape)}")

    # -------------------------
    # 유저가 말한 형태의 접근용
    # -------------------------
    def get(self, class_key: str) -> DatasetTensors:
        return self.class_name[class_key]

    def keys(self) -> List[str]:
        return list(self.class_name.keys())

    # -------------------------
    # 내부: 한 DF에서 sliding window 생성
    # -------------------------
    def _make_sliding_windows_from_df(
        self,
        df: pd.DataFrame,
        window: int,
        stride: int,
        drop_last_incomplete: bool,
        label_source: str,
        label_col: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 입력 컬럼 체크
        missing_in = [c for c in self.input_list if c not in df.columns]
        if missing_in:
            raise KeyError(f"Missing input columns: {missing_in}")

        # 라벨 소스 체크
        label_source = label_source.lower()
        if label_source == "output_cols":
            missing_out = [c for c in self.output_list if c not in df.columns]
            if missing_out:
                raise KeyError(f"Missing output(label) columns: {missing_out}")
        elif label_source == "label_col":
            if label_col not in df.columns:
                raise KeyError(f"Missing label column: {label_col}")
        else:
            raise ValueError("label_source must be 'output_cols' or 'label_col'")

        x_np = df[self.input_list].to_numpy(dtype=np.float32)  # (L, C)
        L, C = x_np.shape

        if L < window:
            if drop_last_incomplete:
                return (
                    torch.empty((0, window, C), dtype=self.x_dtype),
                    torch.empty((0,), dtype=torch.long),
                    torch.empty((0,), dtype=torch.long),
                )
            # 패딩 등은 여기서는 구현하지 않음(원하면 추가해줄게)
            raise ValueError(f"Sequence length L={L} is smaller than window={window}")

        # 윈도우 시작 인덱스들
        starts = list(range(0, L - window + 1, stride))
        if not drop_last_incomplete:
            # 남는 tail도 포함하고 싶으면 padding 전략이 필요해서 여기선 제외(필요하면 구현 가능)
            pass

        X_list = []
        y_list = []
        end_list = []

        for s in starts:
            e = s + window
            xw = x_np[s:e, :]  # (window, C)

            # 라벨은 window "마지막 row"에서 결정
            last_row = df.iloc[e - 1]

            if label_source == "output_cols":
                # 1) output_list 값이 원-핫/indicator이면: 1인 클래스를 고르거나
                # 2) 점수/확률이면: argmax로 클래스를 고름
                vals = last_row[self.output_list].to_numpy(dtype=np.float32)  # (K,)
                y_idx = self._label_index_from_output_vals(vals)
            else:
                # label_col에 문자열 라벨이 들어있다고 가정 (예: "open1")
                lab = str(last_row[label_col])
                y_idx = self.label2idx.get(lab, -1)

            if y_idx < 0:
                continue

            X_list.append(torch.from_numpy(xw).to(dtype=self.x_dtype))
            y_list.append(y_idx)
            end_list.append(e - 1)

        if len(X_list) == 0:
            return (
                torch.empty((0, window, C), dtype=self.x_dtype),
                torch.empty((0,), dtype=torch.long),
                torch.empty((0,), dtype=torch.long),
            )

        X = torch.stack(X_list, dim=0)              # (Nw, T, C)
        y = torch.tensor(y_list, dtype=torch.long)  # (Nw,)
        end_rows = torch.tensor(end_list, dtype=torch.long)
        return X, y, end_rows

    def _label_index_from_output_vals(self, vals: np.ndarray) -> int:
        """
        output_list 컬럼을 마지막 row에서 읽은 값(vals)로부터 class index 결정.
        - 원핫(0/1) 형태면: 가장 큰 값(대개 1)을 가진 클래스 선택
        - 여러 개가 1이면 argmax로 하나 선택
        - 전부 0이면 -1 반환
        """
        if vals.ndim != 1 or vals.shape[0] != len(self.output_list):
            return -1
        m = float(vals.max())
        if m <= 0.0:
            return -1
        return int(vals.argmax())

    def concat_all_datasets(self) -> DatasetTensors:
        """
        self.class_name 에 저장된 모든 클래스의 DatasetTensors를
        하나의 DatasetTensors로 병합한다.

        반환:
            DatasetTensors(
                X:        (N_total, T, C),
                y:        (N_total,),
                file_ids: (N_total,),
                end_rows: (N_total,)
            )
        """
        if not self.class_name:
            raise RuntimeError("class_name is empty. load_all_csv_sliding()을 먼저 호출하세요.")

        X_list = []
        y_list = []
        file_id_list = []
        end_row_list = []

        for label, ds in self.class_name.items():
            X_list.append(ds.X)
            y_list.append(ds.y)
            file_id_list.extend(ds.file_ids)
            end_row_list.append(ds.end_rows)

        X_all = torch.cat(X_list, dim=0)          # (N_total, T, C)
        y_all = torch.cat(y_list, dim=0)          # (N_total,)
        end_rows_all = torch.cat(end_row_list, 0) # (N_total,)

        return DatasetTensors(
            X=X_all,
            y=y_all,
            file_ids=file_id_list,
            end_rows=end_rows_all,
        )

# -------------------------
# 사용 예시
# -------------------------
if __name__ == "__main__":
    data = DataFromCSV(csv_dir="./dataset_OBC/dataset_train_251023", device="cpu")

    data.load_all_csv_sliding(
        window=250,
        stride=10,                 
        drop_last_incomplete=True,
        label_source="output_cols",
        verbose=True,
    )

    all_data = data.concat_all_datasets()

    print(f'X ({type(all_data.X)}): {all_data.X}')
    print(f'y ({type(all_data.y)}): {all_data.y}')
    # for _class in data.class_name:
    #     ds = data.class_name[_class]
    #     print(f'ds.X ({type(ds.X)}): {ds.X}')
    #     print(f'ds.y ({type(ds.y)}): {ds.y}')

    # if "open1" in data.class_name:
    #     ds = data.class_name["open1"]
    #     print(ds.X.shape)          # (N_open1, 250, 2)
    #     print(ds.y.shape)          # (N_open1,)
    #     print(ds.file_ids[0], ds.end_rows[0].item())
