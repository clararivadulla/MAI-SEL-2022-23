from prism import prism, classify
from preprocessing import preprocess
import string
import pandas as pd
import numpy as np

np.random.seed(2012)
path = "/Users/clararivadulla/Repositories/MAI-SEL-2022-23/PW1-PRISM"


def print_rules(prism, train, test, dataset):
    df = pd.DataFrame(columns=['Rule', 'Class', 'Precision', 'Recall', 'Coverage'])
    num = 1
    for C_i in prism:
        for rule_num in prism[C_i]:
            str = f'{num}: if '
            rule_str = 'if '
            rule = prism[C_i][rule_num]
            for i in range(len(rule)):
                if i == (len(rule) - 1):
                    rule_str += f'{rule[i][0]} == {rule[i][1]} '
                    str += f'{rule[i][0]} == {rule[i][1]} '
                else:
                    rule_str += f'{rule[i][0]} == {rule[i][1]} and '
                    str += f'{rule[i][0]} == {rule[i][1]} and '
            train_precision = rule_precision(C_i, rule, train)
            test_precision = rule_precision(C_i, rule, test)
            recall = rule_recall(C_i, rule, train)
            coverage = rule_coverage(C_i, rule, train)
            df = df.append({'Rule': f'{rule_str}then', 'Class': C_i, 'Precision': round(test_precision * 100, 2),
                            'Recall': round(recall * 100, 2), 'Coverage': round(coverage * 100, 2)}, ignore_index=True)
            str += f'-> class = {C_i} | Train Precision: {round(train_precision * 100, 2)}% Test Precision: {round(test_precision * 100, 2)}% Coverage: {round(coverage * 100, 2)}% Recall: {round(recall * 100, 2)}%'
            print(str)
            num += 1
    df.sort_values(by=['Precision', 'Recall'], inplace=True, ascending=[False, False])
    df.reset_index(inplace=True, drop=True)
    df.to_csv(f'results/{dataset}.csv')
    df[:3].to_csv(f'results/best-3-rules-{dataset}.csv')


def accuracy(predicted, y):
    y = list(y)
    correct = 0
    for i in range(len(predicted)):
        if predicted[i] == y[i]:
            correct += 1
    return correct / len(predicted)


def rule_coverage(C_i, rule, x):
    """
    Calculates the ratio between the #instances satisfying the antecedent of R, and the #Total instances in the training dataset.
    :param C_i:
    :param rule:
    :param x:
    :return:
    """
    satisfies_antecedent = x.copy()
    for attr, val in rule:
        satisfies_antecedent = satisfies_antecedent.loc[satisfies_antecedent[attr] == val]
    return len(satisfies_antecedent) / len(x)


def rule_recall(C_i, rule, x):
    """
    Calculates the ratio between the #instances satisfying the antecedent of R and the consequent of R, and the #Total instances in the training dataset belonging to the same class label than R is classifying.
    :param C_i:
    :param rule:
    :param x:
    :return:
    """
    satisfies_all = x.copy()
    for attr, val in rule:
        satisfies_all = satisfies_all.loc[satisfies_all[attr] == val]
    satisfies_all = satisfies_all.loc[
        satisfies_all['class'] == C_i]  # instances satisfying the antecedent of R and the consequent of R
    same_class = x.loc[x['class'] == C_i]
    return len(satisfies_all) / len(same_class)


def rule_precision(C_i, rule, x):
    """
    Calculates the ratio between the #instances satisfying the antecedent of R and the consequent of R, and the #instances satisfying the antecedent of R.
    :param C_i:
    :param rule:
    :param x:
    :return:
    """
    pred = x.copy()
    for attr, val in rule:
        pred = pred.loc[pred[attr] == val]  # instances satisfying the antecedent of R
    correct = pred.copy()
    correct = correct.loc[correct['class'] == C_i]  # instances satisfying the antecedent of R and the consequent of R
    if len(pred) == 0:
        return 0
    return len(correct) / len(pred)


if __name__ == '__main__':

    datasets = ['congressional_voting_records', 'tic_tac_toe_endgame', 'car_evaluation', 'mushroom']

    for dataset in datasets:
        train, test = preprocess(path, dataset)
        y_test = test['class']
        x_test = test.drop(['class'], axis=1)

        P = prism(train)
        predictions = classify(P, x_test)
        acc = accuracy(predictions, y_test) * 100
        title = dataset.replace('_', ' ')
        title = string.capwords(title, sep=None)

        print(
            '*************************************\n' + title + ' Data Set' + '\n*************************************')
        print('Accuracy: ' + str(round(acc, 2)) + '%')
        print('Train size: ' + str(len(train)))
        print('Test size: ' + str(len(test)))
        print('Rules -------------------------------')
        print_rules(P, train, test, dataset)
        print('')
