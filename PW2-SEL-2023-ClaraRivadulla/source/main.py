import sys
import time

import numpy as np
from preprocessing import preprocess
from decision_forest import DecisionTree, DecisionForest, feature_importances
from random_forest import RandomForest
import string
import math
import pandas as pd

np.random.seed(0)

path = "/Users/clararivadulla/Repositories/MAI-SEL-2022-23/PW2-SEL-2023-ClaraRivadulla"


def accuracy(predicted, y):
    """
    Calculates the accuracy of the model (#instances correctly classified / #total of instances)
    """
    y = list(y)
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == y[i]:
            correct += 1
    return correct / len(predicted)


def dict_to_pandas(data):
    df_list = []
    for method, nt_dict in data.items():
        for nt, f_dict in nt_dict.items():
            for f, tuple in f_dict.items():
                df_list.append((method, nt, f, tuple[0], tuple[1]))

    df = pd.DataFrame(df_list, columns=['method', 'NT', 'F', 'accuracy', 'time'])
    return df

if __name__ == '__main__':

    datasets = ['iris', 'tic_tac_toe_endgame', 'abalone', 'mushroom']

    for dataset in datasets:

        title = dataset.replace('_', ' ')
        title = string.capwords(title, sep=None)
        print(
            '*************************************\n' + title + ' Data Set' + '\n*************************************')

        train, test = preprocess(path, dataset)
        y_train = train['class'].to_numpy()
        X_train = train.drop(['class'], axis=1)
        feature_names = X_train.columns.to_numpy()
        X_train = X_train.to_numpy()
        y_test = test['class'].to_numpy()
        X_test = test.drop(['class'], axis=1).to_numpy()
        m = X_train.shape[1]  # Total number of features

        dt = DecisionTree(max_depth=1000)
        start = time.time()
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        end = time.time()
        time_elapsed = round(end - start, 2)
        acc_dt = accuracy(y_pred, y_test) * 100
        print(f'Accuracy DT: ' + str(round(acc_dt, 2)) + '%' + ' Time elapsed: ' + str(time_elapsed) + 's')
        print('-------------------------------------------------------')
        
        NT_list = [1, 10, 25, 50, 75, 100]
        F_RF = [1, 2, int(math.log(m + 1, 2)), int(math.sqrt(m))]
        F_DF = [int(m / 4), int(m / 2), int(3 * (m / 4)), ]

        accuracies = {'DF': {}, 'RF': {}}

        for NT in NT_list:
            accuracies['DF'][NT] = {}
            accuracies['RF'][NT] = {}
            # Decision Forest
            for F in F_DF:
                accuracies['DF'][NT][F] = {}
                df = DecisionForest(max_depth=100, F=F, NT=NT, feature_names=feature_names)
                start = time.time()
                df.fit(X_train, y_train)
                y_pred = df.predict(X_test)
                end = time.time()
                time_elapsed = round(end - start, 2)
                acc_df = accuracy(y_pred, y_test) * 100
                acc_df = round(acc_df, 2)
                accuracies['DF'][NT][F] = (acc_df, time_elapsed)
                print(f'Accuracy DF | NT={NT} | F={F}: ' + str(acc_df) + '%' + ' Time elapsed: ' + str(time_elapsed) + 's')
                df.print_most_important_features()
                print('-------------------------------------------------------')

            # Random Forest
            for F in F_RF:
                accuracies['RF'][NT][F] = {}
                rf = RandomForest(max_depth=100, F=F, NT=NT, feature_names=feature_names)
                start = time.time()
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                end = time.time()
                time_elapsed = round(end - start, 2)
                acc_rf = accuracy(y_pred, y_test) * 100
                acc_rf = round(acc_rf, 2)
                accuracies['RF'][NT][F] = (acc_rf, time_elapsed)
                print(f'Accuracy RF | NT={NT} | F={F}: ' + str(acc_rf) + '%' + ' Time elapsed: ' + str(time_elapsed) + 's')

        df = dict_to_pandas(accuracies)
        print(df)
        df.to_csv(f'results/{dataset}_results.csv', index=False)
        print('')
