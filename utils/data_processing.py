from pandas import read_csv
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
import os
from sklearn.preprocessing import StandardScaler
from utils.utils import name_date, name_time, name_to_dir
from sklearn.metrics import mean_absolute_error, r2_score
from evaluate_result import summarize_metrics

##########################################
##             NORMALIZATION            ##
##########################################

def instance_normalize(insatnce: np.ndarray,
                      method: str = "standard",
                      eps: float = 1e-6) -> np.ndarray:
    """
    Returns a Instance normalized data.

    Parameters
    ----------
    insatnce : np.ndarray, shape (time_steps, num_features)
        The sequence data to be normalized.
    method : str, default="standard"
        - "minmax": Scales values within insatnce to [0, 1] based on min/max values.
        - "standard": Standardizes insatnce using mean and standard deviation (mean 0, variance 1).
    eps : float, default=1e-6
        A small value added to the denominator to prevent division by zero.

    Returns
    -------
    insatnce_norm : np.ndarray, same shape as insatnce
        The normalized sequence.
    """
    if method == "minmax":
        insatnce_min = np.min(insatnce, axis=0)           #  minimum value of features
        insatnce_max = np.max(insatnce, axis=0)           #  maximum value of features
        insatnce_norm = (insatnce - insatnce_min) / (insatnce_max - insatnce_min + eps)
    
    elif method == "standard":
        insatnce_mean = np.mean(insatnce, axis=0)         # mean value of all value
        insatnce_std  = np.std(insatnce, axis=0)          # standard deviation for each value
        insatnce_norm = (insatnce - insatnce_mean) / (insatnce_std + eps)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return insatnce_norm

def normalize_and_save(data,time_flag=False):
    scaler = StandardScaler()
    trans_data = scaler.fit_transform(data)

    dirname = name_to_dir(name='scaler',time_flag=time_flag)

    mean_name = dirname+name_time(default_name='mean.npy')
    scale_name = dirname+name_time(default_name='scale.npy')
    
    np.save(mean_name, scaler.mean_)
    np.save(scale_name, scaler.scale_)

    return scaler

def normalize_std_scaler(data,scaler):
    data = scaler.transform(data)

    return data

def load_and_normalize(data,mean_file,std_file):
    scaler = StandardScaler()
    scaler.mean_ = np.load(mean_file)
    scaler.scale_ = np.load(std_file)

    data = scaler.transform(data)

    return data

##########################################
##            define output             ##
##########################################

def add_normal_class(labels_2d):
    """
    labels_2d: (batch, num_classes)
    return: (batch, num_classes + 1)
    """
    batch, num_classes = labels_2d.shape
    # 마지막 차원 1개 확장
    new_labels = np.zeros((batch, num_classes + 1), dtype=labels_2d.dtype)
    # 기존 값 복사
    new_labels[..., :num_classes] = labels_2d

    # 모두 0인 경우 normal state 인덱스에 1
    mask = np.all(labels_2d == 0, axis=-1)   # (batch, time_steps)
    new_labels[mask, num_classes] = 1        # num_classes == 마지막 인덱스

    return new_labels


##########################################
##           csv to np.array            ##
##########################################

def load_serial_data_from_csv(input_file,feature_list,classes_list,bulk_size,time_steps):
    '''
    Load sequential data from an input file that contains multiple stacked episodes
    '''
    df = read_csv(input_file, index_col=None)
    data = df[feature_list].values
    cls = np.full(len(df), 0, dtype=int)  # 0=정상으로 초기화

    for i in range(len(classes_list)):
        cls[df[classes_list[i]] > 0] = i+1

    labels = to_categorical(cls, num_classes=len(classes_list)+1)  # shape: (N, 9)

    # 나머지 시계열 준비 및 정규화 등은 동일하게 유지
    Xs, Ys = [], []
    ep_num = len(data) // bulk_size
    for ep in range(ep_num):
        start = ep * bulk_size
        for i in range(bulk_size - time_steps):
            instance_start = start + i
            instance_end = instance_start + time_steps
            Xs.append(data[instance_start:instance_end])
            Ys.append(labels[instance_end])

    X_input = np.array(Xs)
    y_output = np.array(Ys)

    # # 각 시퀀스별 정규화
    # norm_data = [instance_normalize(instance, method="minmax") for instance in X_input]
    # X_input = np.array(norm_data)

    # list = [0]*(len(classes_list)+1)
    # for output in y_output:
    #     list += output
    # print("sample distribution by class:", list)

    return X_input, y_output

def load_features_data_from_csv(input_file,feature_list):
    '''
    This function return a ndarray including values of features
    '''
    df = read_csv(input_file, index_col=None)
    data = df[feature_list].values

    return(data)

def load_classes_data_from_csv(input_file,classes_list):
    '''
    This function return a ndarray including values of classes
    '''
    df = read_csv(input_file, index_col=None)
    data = df[classes_list].values

    return(data)

def read_all_csv_to_np_list(dir_path,feature_list,classes_list,dim_reduction=False):
    features_list = []
    class_list = []

    if not dim_reduction:
        for fname in os.listdir(dir_path):
            if fname.endswith('.csv'):
                file_path = os.path.join(dir_path, fname)
                features = df[feature_list].values
                classes = df[classes_list].values
                features_list.append(features)
                class_list.append(classes)
    else:
        for fname in os.listdir(dir_path):
            if fname.endswith('.csv'):
                file_path = os.path.join(dir_path, fname)
                df = pd.read_csv(file_path)

                features = df[feature_list].values  
                classes = df[classes_list].values
                for row in features:
                    features_list.append(row)
                for row in classes:
                    class_list.append(row)
                # data_list.append(data)
    
    return features_list, class_list

def make_sequence_dataset(dir_path, time_steps, feature_list, classes_list, scaler=None):
    X_list = []   # feature 시퀀스 리스트
    y_list = []   # class(레이블) 시퀀스 리스트

    if scaler == None:
        for fname in os.listdir(dir_path):
            if fname.endswith('.csv'):
                file_path = os.path.join(dir_path, fname)
                df = pd.read_csv(file_path)
                
                features = df[feature_list].values  # (N, F)
                classes = df[classes_list].values   # (N, C) 또는 (N,) 형태
                
                N = len(df)
                for i in range(N - time_steps):
                    X_seq = features[i:i+time_steps]     # (time_steps, F)
                    y_seq = classes[i+time_steps-1]      # (time_steps, C) 또는 (time_steps,)
                                        
                    X_list.append(X_seq)
                    y_list.append(y_seq)
        
        X = np.array(X_list)   # (전체시퀀스수, time_steps, F)
        y = np.array(y_list)   # (전체시퀀스수, time_steps, C) 또는 (전체시퀀스수, time_steps)

    else:
        for fname in os.listdir(dir_path):
            if fname.endswith('.csv'):
                file_path = os.path.join(dir_path, fname)
                df = pd.read_csv(file_path)
                
                features = df[feature_list].values  # (N, F)
                classes = df[classes_list].values   # (N, C) 또는 (N,) 형태
                
                N = len(df)
                for i in range(N - time_steps):
                    X_seq = features[i:i+time_steps]     # (time_steps, F)
                    y_seq = classes[i+time_steps]      # (time_steps, C) 또는 (time_steps,)

                    trans_X_seq = scaler.transform(X_seq)
                    
                    X_list.append(trans_X_seq)
                    y_list.append(y_seq)
        
        X = np.array(X_list)   # (전체시퀀스수, time_steps, F)
        y = np.array(y_list)   # (전체시퀀스수, time_steps, C) 또는 (전체시퀀스수, time_steps)

    return X, y

def evaluate_prediction(y_true, y_pred, epsilon=1e-8):
    e_list = []
    absolute_error_list = []
    for i in range(y_pred.shape[1]):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        mae = mean_absolute_error(y_t, y_p)
        relative_error = np.abs((y_p - y_t) / (y_t + epsilon))
        mre = np.mean(relative_error) * 100
        r2 = r2_score(y_t, y_p)

        print(f"\n>> [Output {i}]")
        print(f"   MAE: {mae:.4f}")
        print(f"   Mean Relative Error: {mre:.2f}%")
        print(f"   R² Score: {r2:.4f}")
        e_list.append(mre)
        absolute_error_list.append(mae)
    # print(f"   MRE: {sum(e_list)/len(e_list):.2f}%")
    # print(f"   MRE: {sum(absolute_error_list)/len(absolute_error_list):.4f}")
    summary = summarize_metrics(y_true, y_pred)
    for k, v in summary.items():
        print(k, ":", v)