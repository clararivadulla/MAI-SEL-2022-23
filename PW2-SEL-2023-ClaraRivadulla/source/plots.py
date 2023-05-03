import string

import pandas as pd
from matplotlib import pyplot as plt

datasets = ['iris', 'tic_tac_toe_endgame', 'abalone', 'mushroom']


def plot_results(df, dataset_name, method='DF'):
    title = dataset_name.replace('_', ' ')
    title = string.capwords(title, sep=None)
    NTs = df.NT.unique()
    colors = ['r', 'b', 'g', 'y', 'o', 'c']
    if method=='DF':
        df = df[df['method'] == 'DF']
        Fs = df.F.unique()
        method_title = 'Decision Forest'

    elif method=='RF':
        df = df[df['method'] == 'RF']
        Fs = df.F.unique()
        method_title = 'Random Forest'
    i = 0
    for F in Fs:
        df_F = df[df['F'] == F]
        y = df_F['accuracy'].to_numpy()
        plt.plot(NTs, y, color=colors[i], label=f'F={F}')
        i += 1

    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.title(f"{title} Data Set - {method_title}")
    plt.legend()
    plt.savefig(f'figures/{dataset_name}_{method}.png')
    plt.clf()

for dataset in datasets:
    results = pd.read_csv(f'results/{dataset}_results.csv')
    plot_results(results, dataset, method='DF')
    plot_results(results, dataset, method='RF')

