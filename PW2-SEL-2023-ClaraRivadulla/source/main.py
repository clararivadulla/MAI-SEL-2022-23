import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocessing import preprocess
from decision_forest import DecisionTree, DecisionForest
import string
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
        train, test = preprocess(path, dataset)
        y_train = train['class']
        X_train = train.drop(['class'], axis=1)
        y_test = test['class']
        X_test = test.drop(['class'], axis=1)

        dt = DecisionTree(max_depth=100)
        dt.fit(X_train.to_numpy(), y_train.to_numpy())
        y_pred = dt.predict(X_test.to_numpy())
        acc = accuracy(y_pred, y_test.to_numpy()) * 100
        title = dataset.replace('_', ' ')
        title = string.capwords(title, sep=None)
        """
        df = DecisionForest(max_depth=100)
        df.fit(X_train.to_numpy(), y_train.to_numpy())
        y_pred = df.predict(X_test.to_numpy())
        acc_df = accuracy(y_pred, y_test.to_numpy()) * 100"""

        print(
            '*************************************\n' + title + ' Data Set' + '\n*************************************')
        print('Accuracy DT: ' + str(round(acc, 2)) + '%')
        #print('Accuracy DF: ' + str(round(acc_df, 2)) + '%')
        print('Train size: ' + str(len(train)))
        print('Test size: ' + str(len(test)))
        print('')

