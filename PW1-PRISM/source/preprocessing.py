import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess(path, dataset_name):

    if dataset_name is 'car_evaluation':
        df = pd.read_csv(f"{path}/data/{dataset_name}.csv", names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
    else:
        df = pd.read_csv(f"{path}/data/{dataset_name}.csv")

    if 'Class Name' in df.columns:
        df.rename({'Class Name': 'class'}, axis=1, inplace=True)
    if 'bruises%3F' in df.columns:
        df.rename({'bruises%3F': 'bruises'}, axis=1, inplace=True)
    if 'V10' in df.columns:
        df.rename({'V10': 'class'}, axis=1, inplace=True)
    if 'B' in df.columns:
        df.rename({'B': 'class'}, axis=1, inplace=True)


    df.columns = df.columns.str.replace(' ', '')
    df = df.replace('?', np.nan)
    df = df.dropna().reset_index()
    train, test = train_test_split(df, test_size = 0.2)

    y_test = test['class']
    x_test = test.drop(['class'], axis=1)

    return train, x_test, y_test
