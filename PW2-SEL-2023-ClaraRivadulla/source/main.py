import numpy as np
from preprocessing import preprocess
from decision_forest import DecisionTree, DecisionForest
from random_forest import RandomForest
import string
import math

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


if __name__ == '__main__':

    datasets = ['iris', 'congressional_voting_records', 'tic_tac_toe_endgame', 'car_evaluation', 'mushroom']

    for dataset in datasets:
        title = dataset.replace('_', ' ')
        title = string.capwords(title, sep=None)
        print(
            '*************************************\n' + title + ' Data Set' + '\n*************************************')

        train, test = preprocess(path, dataset)
        y_train = train['class'].to_numpy()
        X_train = train.drop(['class'], axis=1).to_numpy()
        y_test = test['class'].to_numpy()
        X_test = test.drop(['class'], axis=1).to_numpy()
        m = X_train.shape[1] # Total number of features


        NT_list = [1, 10, 25, 50, 75, 100]
        F_RF = [1, 2, int(math.log(m + 1, 2)), int(math.sqrt(m))]
        F_DF = [int(m/4), int(m/2), int(3 * (m/4)), ]

            # Decision Forest
        for NT in NT_list:
            for F in F_DF:
                df = DecisionForest(max_depth=100, F=F, NT=NT)
                df.fit(X_train, y_train)
                y_pred = df.predict(X_test)
                acc_df = accuracy(y_pred, y_test) * 100
                print(f'Accuracy DF | F={F} | NT={NT}: ' + str(round(acc_df, 2)) + '%')



            for F in F_RF:
                rf = RandomForest(max_depth=100, F=F, NT=NT)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                acc_rf = accuracy(y_pred, y_test) * 100
                print(f'Accuracy RF | F={F} | NT={NT}: ' + str(round(acc_rf, 2)) + '%')

        print('')

