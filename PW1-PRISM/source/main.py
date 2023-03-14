import pandas as pd
from prism import fit, classify, accuracy
from preprocessing import preprocess
import string

path = "/Users/clararivadulla/Repositories/MAI-SEL-2022-23/PW1-PRISM/"

if __name__ == '__main__':

    datasets = ['congressional_voting_records', 'tic_tac_toe_endgame', 'mushroom']
    for dataset in datasets:
        train, x_test, y_test = preprocess(path, dataset)
        prism = fit(train)
        predictions = classify(prism, x_test)
        acc = accuracy(predictions, y_test) * 100
        title = dataset.replace('_', ' ')
        title = string.capwords(title, sep = None)
        print('*************************************\n' + title + ' Data Set' + '\n*************************************')
        print('Accuracy: ' + str(round(acc, 2)) + '%')
        print('Rules -------------------------------')
        print('')
