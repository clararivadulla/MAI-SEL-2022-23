from prism import fit, classify, accuracy
from preprocessing import preprocess
import string

path = "/Users/clararivadulla/Repositories/MAI-SEL-2022-23/PW1-PRISM"


def print_rules(prism):
    num = 1
    for C_i in prism:
        for rule_num in prism[C_i]:
            str = f'{num}: if '
            rule = prism[C_i][rule_num]
            for i in range(len(rule)):
                if i == (len(rule) - 1):
                    str += f'{rule[i][0]} == {rule[i][1]} '
                else:
                    str += f'{rule[i][0]} == {rule[i][1]} and '
            str += f'-> class = {C_i}'
            print(str)
            num += 1


if __name__ == '__main__':

    datasets = ['car_evaluation', 'congressional_voting_records', 'tic_tac_toe_endgame', 'mushroom']

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
        print_rules(prism)
        print('')
