import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ✅ normal 추가 (중요)
output_list = ["normal","open1","open2","open3","open4","short1","short2","short3","short4"]
input_list  = ["Vo:Measured voltage","IL:Measured current"]


@dataclass
class DatasetTensors:
    X: torch.Tensor           # (N, T, C)
    y: torch.Tensor           # (N,)  (class index)
    file_ids: List[str]       # (N,)  (원본 파일명)
    end_rows: torch.Tensor    # (N,)  (각 window의 마지막 row index)


class DataFromCSV:
    def __init__(
        self,
        csv_dir: str,
        input_list: List[str] = input_list,
        output_list: List[str] = output_list,
        device: str = "cpu",
        x_dtype: torch.dtype = torch.float32,
        # ✅ all-zero일 때 넣을 클래스 이름(기본 normal)
        zero_as_label: str = "normal",
    ):
        self.csv_dir = csv_dir
        self.input_list = input_list
        self.output_list = output_list
        self.device = torch.device(device)
        self.x_dtype = x_dtype

        self.label2idx: Dict[str, int] = {name: i for i, name in enumerate(self.output_list)}
        self.idx2label: Dict[int, str] = {i: name for name, i in self.label2idx.items()}

        self.zero_as_label = zero_as_label
        if self.zero_as_label not in self.label2idx:
            raise ValueError(
                f"zero_as_label='{self.zero_as_label}' is not in output_list. "
                f"Add it to output_list. (current: {self.output_list})"
            )

        self.class_name: Dict[str, DatasetTensors] = {}

    def load_all_csv_sliding(
        self,
        window: int = 250,
        stride: int = 1,
        drop_last_incomplete: bool = True,
        label_source: str = "output_cols",
        label_col: str = "label",
        file_filter: Optional[str] = None,
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

            for i in range(Xw.shape[0]):
                y_idx = int(yw[i].item())
                if y_idx < 0:
                    continue
                label = self.idx2label[y_idx]
                buf[label].append((Xw[i], y_idx, fname, int(end_rows[i].item())))

        self.class_name = {}
        for label, items in buf.items():
            if len(items) == 0:
                continue
            X_list, y_list, fid_list, end_list = zip(*items)

            X_cat = torch.stack(X_list, dim=0).to(self.device, dtype=self.x_dtype)
            y_cat = torch.tensor(y_list, dtype=torch.long, device=self.device)
            end_rows = torch.tensor(end_list, dtype=torch.long, device=self.device)

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

    def get(self, class_key: str) -> DatasetTensors:
        return self.class_name[class_key]

    def keys(self) -> List[str]:
        return list(self.class_name.keys())

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

        label_source = label_source.lower()

        # ✅ (핵심) output_cols 모드일 때 'normal'은 DF에 없어도 OK 처리
        if label_source == "output_cols":
            required_out = [c for c in self.output_list if c != self.zero_as_label]
            missing_out = [c for c in required_out if c not in df.columns]
            if missing_out:
                raise KeyError(f"Missing output(label) columns: {missing_out}")

            # DF에서 실제로 읽을 output 컬럼들 (normal 제외)
            df_output_cols = required_out

        elif label_source == "label_col":
            if label_col not in df.columns:
                raise KeyError(f"Missing label column: {label_col}")
            df_output_cols = None
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
            raise ValueError(f"Sequence length L={L} is smaller than window={window}")

        starts = list(range(0, L - window + 1, stride))

        X_list, y_list, end_list = [], [], []

        for s in starts:
            e = s + window
            xw = x_np[s:e, :]
            last_row = df.iloc[e - 1]

            if label_source == "output_cols":
                # ✅ normal 컬럼 없이 open/short만 읽어서 판단
                vals = last_row[df_output_cols].to_numpy(dtype=np.float32)  # (K_no_normal,)

                m = float(vals.max()) if vals.size > 0 else 0.0
                if m <= 0.0:
                    # all-zero -> normal
                    y_idx = self.label2idx[self.zero_as_label]
                else:
                    # argmax -> 해당 라벨명을 전체 output_list의 index로 매핑
                    best_label = df_output_cols[int(vals.argmax())]
                    y_idx = self.label2idx.get(best_label, -1)

            else:
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

        X = torch.stack(X_list, dim=0)
        y = torch.tensor(y_list, dtype=torch.long)
        end_rows = torch.tensor(end_list, dtype=torch.long)
        return X, y, end_rows

    def _label_index_from_output_vals(self, vals: np.ndarray) -> int:
        """
        ✅ 변경점:
        - output_list 값이 전부 0이면 self.zero_as_label(기본 normal)로 분류
        """
        if vals.ndim != 1 or vals.shape[0] != len(self.output_list):
            return -1

        m = float(vals.max())
        if m <= 0.0:
            # ✅ all-zero -> normal
            return int(self.label2idx[self.zero_as_label])

        return int(vals.argmax())

    def concat_all_datasets(self) -> DatasetTensors:
        if not self.class_name:
            raise RuntimeError("class_name is empty. load_all_csv_sliding()을 먼저 호출하세요.")

        X_list = []
        y_list = []
        file_id_list = []
        end_row_list = []

        for _, ds in self.class_name.items():
            X_list.append(ds.X)
            y_list.append(ds.y)
            file_id_list.extend(ds.file_ids)
            end_row_list.append(ds.end_rows)

        X_all = torch.cat(X_list, dim=0)
        y_all = torch.cat(y_list, dim=0)
        end_rows_all = torch.cat(end_row_list, 0)

        return DatasetTensors(
            X=X_all,
            y=y_all,
            file_ids=file_id_list,
            end_rows=end_rows_all,
        )

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