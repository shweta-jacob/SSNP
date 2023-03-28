import csv

import matplotlib.pyplot as plt
import numpy as np


class HyperTunerResults:
    M = [1, 2]
    m = [1, 2, 3, 4, 5]
    views = [1, 2, 3, 4, 5]


if __name__ == '__main__':
    # multi-line graph plots for hyperparameter tuning results
    slice_length = len(HyperTunerResults.m)
    # colors = ['b', 'g', 'r', 'm', 'c', 'y', 'k', 'w']
    cmap = [plt.cm.get_cmap("Reds"), plt.cm.get_cmap("Greens")]
    slicedCM = [cmap[0](np.linspace(0.4, 0.75, 10)), cmap[1](np.linspace(0.5, 0.8, 10))]

    line_style = [
        ('loosely dotted', (0, (1, 10))),
        ('dotted', (0, (1, 1))),
        ('densely dotted', (0, (1, 1))),
        ('long dash with offset', (5, (10, 3))),
        ('loosely dashed', (0, (5, 10))),
        ('dashed', (0, (5, 5))),
        ('densely dashed', (0, (5, 1))),

        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),

        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

    marker_style = ['D', 's', 'o', '^', 'H', '*', 'd', 'p', 'h', 'X']

    dataset_acc_indices = [3, 4, 5, 6]
    datasets = ["ppi-bp", "hpo-metab", "hpo-neuro", "em-user"]

    dataset_vals = {"ppi-bp": {}, "hpo-metab": {}, "hpo-neuro": {}, "em-user": {}}

    with open('data.csv', newline='') as csvfile:
        auc_data = list(csv.reader(csvfile))

        header = auc_data[0]
        for row in auc_data[1:]:
            for dataset_id, dataset_name in zip(dataset_acc_indices, dataset_vals):
                try:
                    dataset_vals[dataset_name][(int(row[0]), int(row[1]))].append(
                        float(row[dataset_id].split("±")[0].strip()))
                except KeyError:
                    dataset_vals[dataset_name][(int(row[0]), int(row[1]))] = [
                        float(row[dataset_id].split("±")[0].strip())]

    for dataset, dataset_values_dict in dataset_vals.items():
        f = plt.figure()
        x = HyperTunerResults.m
        default_x_ticks = range(len(x))
        plt.rcParams.update({'font.size': 16.5})
        plt.xticks(default_x_ticks, x)
        # plt.yticks(np.arange(0, 70, skip_axis[index]))
        # plt.ylim(ylimit[index])

        for index, (view_M, auc_list) in enumerate(dataset_values_dict.items()):
            if view_M[-1] == 2:
                continue
            plt.plot(default_x_ticks, auc_list, label=f"V={view_M[0]}, K={view_M[-1]}",
                     color=slicedCM[0][index],
                     linestyle=line_style[index][1], marker=marker_style[index], linewidth=2, markersize=10)

        plt.ylabel('AUC Scores')
        plt.xlabel('k: Length of Walks')
        plt.legend(loc="lower right", ncol=2, borderpad=0.2, labelspacing=0.25, borderaxespad=0.25)
        plt.tight_layout()
        plt.show()
        f.savefig(f"{dataset}_hypertuner_auc.pdf", bbox_inches='tight')
