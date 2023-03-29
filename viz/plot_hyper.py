import csv

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


class HyperTunerResults:
    M = [1, 2]
    m = [1, 2, 3, 4, 5]
    views = [1, 2, 3, 4, 5]


if __name__ == '__main__':
    # multi-line graph plots for hyperparameter tuning results
    slice_length = len(HyperTunerResults.m)
    colors = [mcolors.CSS4_COLORS['coral'], mcolors.CSS4_COLORS['cadetblue'], 'r', mcolors.CSS4_COLORS['steelblue'],
              'y', 'k', 'w']
    # cmap = [plt.cm.get_cmap("Reds"), plt.cm.get_cmap("Greens")]
    # slicedCM = [cmap[0](np.linspace(0.4, 0.75, 10)), cmap[1](np.linspace(0.5, 0.8, 10))]
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

    dataset_acc_indices = [3, 5, 7, 9]
    timing_indices = [4, 6, 8, 10]
    datasets = ["ppi-bp", "hpo-metab", "hpo-neuro", "em-user"]

    dataset_vals = {"ppi-bp": [[], []], "hpo-metab": [[], []], "hpo-neuro": [[], []], "em-user": [[], []]}

    with open('data.csv', newline='') as csvfile:
        auc_data = list(csv.reader(csvfile))

        header = auc_data[0]
        for row in auc_data[1:]:
            for dataset_id, timing_id, dataset_name in zip(dataset_acc_indices, timing_indices, dataset_vals):
                if dataset_id == 5 or dataset_id == 9:
                    if row[1] == '1' and row[2] == '5':
                        dataset_vals[dataset_name][0].append(
                            float(row[dataset_id].split("±")[0].strip()))
                        dataset_vals[dataset_name][1].append(
                            float(row[timing_id].split("±")[0].strip()))

                else:
                    if row[1] == '1' and row[2] == '1':
                        dataset_vals[dataset_name][0].append(
                            float(row[dataset_id].split("±")[0].strip()))
                        dataset_vals[dataset_name][1].append(
                            float(row[timing_id].split("±")[0].strip()))

    f = plt.figure()
    x = HyperTunerResults.m
    default_x_ticks = range(len(x))
    plt.rcParams.update({'font.size': 16.5})
    plt.xticks(default_x_ticks, x)
    plt.yticks(np.arange(0.35, 1.0, 0.05))
    plt.ylim(bottom=0.45)

    for index, (dataset, dataset_values) in enumerate(dataset_vals.items()):
        plt.plot(default_x_ticks, dataset_values[0], label=f"{dataset}",
                 color=colors[index],
                 linestyle=line_style[1][1], marker=marker_style[index], linewidth=2, markersize=10)

    plt.ylabel('Micro-F1 Score')
    plt.xlabel('Number of Views')
    plt.legend(loc="lower right", ncol=2, borderpad=0.1, labelspacing=0.1, borderaxespad=0.25, columnspacing=0.2)
    plt.tight_layout()
    plt.show()
    f.savefig(f"hypertuner_f1.pdf", bbox_inches='tight')

    f = plt.figure()
    x = HyperTunerResults.m
    default_x_ticks = range(len(x))
    plt.rcParams.update({'font.size': 16.5})
    plt.xticks(default_x_ticks, x)
    # plt.yticks(np.arange(0.35, 1.0, 0.05))
    # plt.ylim(bottom=0.45)

    for index, (dataset, dataset_values) in enumerate(dataset_vals.items()):
        plt.plot(default_x_ticks, dataset_values[1], label=f"{dataset}",
                 color=colors[index],
                 linestyle=line_style[1][1], marker=marker_style[index], linewidth=2, markersize=10)

    plt.ylabel('Avg. time per epoch (sec.)')
    plt.xlabel('Number of Views')
    plt.legend(loc="upper left", ncol=2, borderpad=0.1, labelspacing=0.1, borderaxespad=0.25, columnspacing=0.2)
    plt.tight_layout()
    plt.show()
    f.savefig(f"hypertuner_time.pdf", bbox_inches='tight')
