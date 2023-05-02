import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def min_max_scale_columns(df, cols):
    min_max_scaler = preprocessing.MinMaxScaler()
    df[cols] = min_max_scaler.fit_transform(df[cols])

def standardize_columns(df, cols):
    scaler = preprocessing.StandardScaler()
    scaler.fit(df[cols])
    df[cols] = scaler.transform(df[cols])

def preprocess(path, dataset_name):
    if dataset_name == 'car_evaluation':
        df = pd.read_csv(f"{path}/data/{dataset_name}.csv",
                         names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
    else:
        df = pd.read_csv(f"{path}/data/{dataset_name}.csv")
    if 'Class Name' in df.columns:
        df.rename({'Class Name': 'class'}, axis=1, inplace=True)
    if 'Class_number_of_rings' in df.columns:
        df.rename({'Class_number_of_rings': 'class'}, axis=1, inplace=True)
    if 'bruises%3F' in df.columns:
        df.rename({'bruises%3F': 'bruises'}, axis=1, inplace=True)
    if dataset_name=='flag':
        df.rename({'name': 'class'}, axis=1, inplace=True)
    if 'V10' in df.columns:
        df.rename({'V10': 'class'}, axis=1, inplace=True)
    if 'variety' in df.columns:
        df.rename({'variety': 'class'}, axis=1, inplace=True)
    if 'B' in df.columns:
        df.rename({'B': 'class'}, axis=1, inplace=True)

    df.columns = df.columns.str.replace(' ', '')
    df = df.replace('?', np.nan)
    df = df.dropna().reset_index(drop=True)

    numeric_columns = df._get_numeric_data().columns.values.tolist()
    if len(numeric_columns) > 1:
        min_max_scale_columns(df, numeric_columns)

    train, test = train_test_split(df, test_size=0.2)
    return train, test
