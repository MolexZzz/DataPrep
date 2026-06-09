import numpy as np
import pandas as pd
import sys
sys.path.append("../Datasets")
from dataPreprocess import load_adult_data, load_compas_data, load_hsls_data,load_acs_data, load_synthetic_data
from missing_mechanism import missing_mcar, missing_sim_mar_probability, missing_sim_mnar_probability
from utils import Handling_Discrete_Variables
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from torch.utils.data import Dataset
from torch import nn
import torch
from collections import Counter
from torch.utils.data import DataLoader
from utils import PandasDataSet, InfiniteDataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json

from imputation.MIWAE import MIWAE
from imputation.notmiwae import notMIWAE
from imputation.GAIN import gain_main
from sklearn.impute import IterativeImputer

def get_mask(x, s):
    mask_matrix_x = x.notnull().astype(int)
    mask_matrix_s = s.notnull().astype(int)
    return mask_matrix_x, mask_matrix_s


def read_the_data(opt):
    if opt.balance:
        data_path = f"../Datasets/{opt.dataset}/{opt.dataset}_{opt.missing_rate}_{opt.missing_mechanism}_{opt.seed}_balance/"
    else:
        data_path = f"../Datasets/{opt.dataset}/{opt.dataset}_{opt.missing_rate}_{opt.missing_mechanism}_{opt.seed}/"

    # Read the dataframe
    train_or_test ='train'
    df_train = pd.read_csv(
        data_path + train_or_test + ".csv",
    )

    train_or_test ='test'
    df_test = pd.read_csv(
        data_path + train_or_test + ".csv",
    )

    train_or_test ='valid'
    df_valid = pd.read_csv(
        data_path + train_or_test + ".csv",
    )

    with open(data_path + "dataset_stats.json") as f:
        dataset_stats = json.load(f)

    # Get the column names
    target_column_name = dataset_stats["target_column_name"][0]
    sensitive_column_names = dataset_stats["sensitive_column_names"][0]

    X_train = df_train.drop(columns=[target_column_name,sensitive_column_names])
    X_test = df_test.drop(columns=[target_column_name,sensitive_column_names])
    X_valid = df_valid.drop(columns=[target_column_name,sensitive_column_names])


    cat_dims = dataset_stats['cat_dims']
    continuous_columns = dataset_stats['continuous_columns']
    categorical_columns = dataset_stats['categorical_columns']

    categorical_indicator = list(np.zeros(X_train.shape[1]).astype(bool))

    # continuous columns
    for col in X_train.columns:
        if col in categorical_columns:
            categorical_indicator[X_train.columns.get_loc(col)] = True
        elif col == opt.target:
            pass
        elif col == opt.sensitive:
            pass

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X_train.columns))) - set(cat_idxs))

    y_train = df_train[target_column_name]
    y_test = df_test[target_column_name]
    y_valid = df_valid[target_column_name]

    s_train = df_train[sensitive_column_names]
    s_test = df_test[sensitive_column_names]
    s_valid = df_valid[sensitive_column_names]

    nan_mask_x_train, nan_mask_s_train = get_mask(X_train, s_train)
    nan_mask_x_valid, nan_mask_s_valid = get_mask(X_valid, s_valid)
    nan_mask_x_test, nan_mask_s_test = get_mask(X_test, s_test)


    return X_train, y_train, nan_mask_x_train, s_train, nan_mask_s_train, X_valid, y_valid, nan_mask_x_valid, s_valid, nan_mask_s_valid, X_test, y_test, nan_mask_x_test, s_test, nan_mask_s_test, cat_idxs, con_idxs, cat_dims, categorical_columns, continuous_columns 



def data_split(X, s, y, nan_mask_x, nan_mask_s):
    x_d = {
        'data': X,
        'mask': nan_mask_x.values,
    }

    s_d = {
        'data': s.reshape(-1, 1),
        'mask': nan_mask_s.values.reshape(-1, 1),
    }

    if x_d['data'].shape != x_d['mask'].shape:
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y.reshape(-1, 1)
    } 
    return x_d, s_d, y_d

def load_data(dataset, missing_rate, missing_mechanism, seed, balance):
    if balance:
        data_path = f"../Datasets/{dataset}/{dataset}_{missing_rate}_{missing_mechanism}_{seed}_balance/"
    else:
        data_path = f"../Datasets/{dataset}/{dataset}_{missing_rate}_{missing_mechanism}_{seed}/"

    # read the dataframe
    train_or_test = "train"
    df_train = pd.read_csv(
        data_path + train_or_test + '.csv'
    )

    train_or_test = "test"
    df_test = pd.read_csv(
        data_path + train_or_test + '.csv'
    )

    train_or_test = "valid"
    df_valid = pd.read_csv(
        data_path + train_or_test + '.csv'
    )

    with open(data_path + "dataset_stats.json") as f:
        dataset_stats = json.load(f)

    target_columns_name = dataset_stats["target_column_name"][0]
    sensitive_column_name = dataset_stats["sensitive_column_names"][0]
    categorical_column_name = dataset_stats['categorical_columns']



    df_train_x = df_train.drop(columns=[target_columns_name, sensitive_column_name])
    df_train_y = df_train[target_columns_name]
    df_train_s = df_train[sensitive_column_name]

    df_test_x = df_test.drop(columns=[target_columns_name, sensitive_column_name])
    df_test_y = df_test[target_columns_name]
    df_test_s = df_test[sensitive_column_name]

    df_valid_x = df_valid.drop(columns=[target_columns_name, sensitive_column_name])
    df_valid_y = df_valid[target_columns_name]
    df_valid_s = df_valid[sensitive_column_name]

    return df_train_x, df_train_s, df_train_y, df_test_x, df_test_s, df_test_y, df_valid_x, df_valid_s, df_valid_y, categorical_column_name


def data_prep_concise(opt):
    np.random.seed(opt.seed)
    X_train, y_train, nan_mask_x_train, s_train, nan_mask_s_train, X_valid, y_valid, nan_mask_x_valid, s_valid, nan_mask_s_valid, X_test, y_test, nan_mask_x_test, s_test, nan_mask_s_test, cat_idxs, con_idxs, cat_dims, categorical_columns, continuous_columns = read_the_data(opt)

    full_data_train,_, _, full_data_test, _, _, full_data_val, _, _, _ = load_data(opt.dataset, 0.0, "MCAR", opt.seed, opt.balance)

    missing_columns = X_train.columns[X_train.isnull().any()].tolist()

    # imputation_method
    if opt.imputation_method == "mean":
        X_train = X_train.fillna(X_train.mean())
        X_valid = X_valid.fillna(X_valid.mean())
        X_test = X_test.fillna(X_test.mean())

    elif opt.imputation_method == "zero":
        X_train = X_train.fillna(0)
        X_valid = X_valid.fillna(0)
        X_test = X_test.fillna(0)

    elif opt.imputation_method == "miwae":
        X_train = MIWAE(X_train.values, full_data_train.values)
        X_valid = MIWAE(X_valid.values, full_data_val.values)
        X_test = MIWAE(X_test.values, full_data_test.values)

        if type(X_train) == np.ndarray:
            X_train = pd.DataFrame(X_train, columns=full_data_train.columns)
            X_valid = pd.DataFrame(X_valid, columns=full_data_val.columns)
            X_test = pd.DataFrame(X_test,columns=full_data_test.columns)

    elif opt.imputation_method == "notmiwae":
        missing_mask_train = np.array(~np.isnan(X_train), dtype=np.float32)
        X_train = notMIWAE(np.array(X_train), missing_mask_train, full_data_train)
        missing_mask_val = np.array(~np.isnan(X_valid), dtype=np.float32)
        X_valid = notMIWAE(np.array(X_valid), missing_mask_val, full_data_val)
        missing_mask_test = np.array(~np.isnan(X_test), dtype=np.float32)
        X_test = notMIWAE(np.array(X_test), missing_mask_test, full_data_test)

        if type(X_train) == np.ndarray:
            X_train = pd.DataFrame(X_train, columns=full_data_train.columns)
            X_valid = pd.DataFrame(X_valid, columns=full_data_val.columns)
            X_test = pd.DataFrame(X_test,columns=full_data_test.columns)

    elif opt.imputation_method == "gain":
        X_train = gain_main(X_train)
        X_valid = gain_main(X_valid)
        X_test = gain_main(X_test)
        if type(X_train) == np.ndarray:
            X_train = pd.DataFrame(X_train, columns=full_data_train.columns)
            X_valid = pd.DataFrame(X_valid, columns=full_data_val.columns)
            X_test = pd.DataFrame(X_test,columns=full_data_test.columns)

    elif opt.imputation_method == "mice":
        mice_imputer = IterativeImputer(max_iter=10, random_state=opt.seed)
        X_train = mice_imputer.fit_transform(X_train)
        X_valid = mice_imputer.transform(X_valid)
        X_test = mice_imputer.transform(X_test)

        if type(X_train) == np.ndarray:
            X_train = pd.DataFrame(X_train, columns=full_data_train.columns)
            X_valid = pd.DataFrame(X_valid, columns=full_data_val.columns)
            X_test = pd.DataFrame(X_test,columns=full_data_test.columns)

    
    X_train = X_train.values
    X_valid = X_valid.values
    X_test = X_test.values
    y_train = y_train.values
    y_valid = y_valid.values
    y_test = y_test.values

    s_train = s_train.values
    s_valid = s_valid.values
    s_test = s_test.values
    # constract the data
    X_train, s_train, y_train = data_split(X_train, s_train, y_train, nan_mask_x_train, nan_mask_s_train)

    X_valid, s_valid, y_valid = data_split(X_valid, s_valid,y_valid, nan_mask_x_valid, nan_mask_s_valid)

    X_test, s_test, y_test = data_split(X_test, s_test, y_test, nan_mask_x_test, nan_mask_s_test)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)

    return cat_dims, cat_idxs, con_idxs, X_train, s_train, y_train, X_valid, s_valid, y_valid, X_test, s_test, y_test, train_mean, train_std


class DataSetCatCon(Dataset):
    def __init__(self, X, s, Y, cat_cols, con_cols, task='clf',continuous_mean_std=None):
        cat_cols = list(cat_cols)
        con_cols = list(con_cols)
        
        X_mask = X['mask'].copy()
        X = X['data'].copy()

        self.X1 = X[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2 = X[:, con_cols].copy().astype(np.float32)  # numerical columns
        self.X1_mask = X_mask[:, cat_cols].copy().astype(np.int64)  # categorical columns
        self.X2_mask = X_mask[:, con_cols].copy().astype(np.int64)  # numerical columns

        self.s = s["data"].copy().astype(np.int64)
        self.s_mask = s["mask"].copy().astype(np.int64)
        if task == 'clf':
            self.y = Y['data']#.astype(np.float32)
        else:
            self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous 
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx], self.s[idx], self.y[idx], np.concatenate((self.cls_mask[idx], self.X1_mask[idx])), self.X2_mask[idx], np.array(self.s_mask[idx])


