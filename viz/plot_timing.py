import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

species = ("ppi-bp", "hpo-metab", "hpo-neuro", "em-user")
pre_proc = {
    'OV': (4.489, 17.634, 17.703, 26.930),
    'PV 5 views': (5.733, 20.191, 20.735, 27.304),
    'PV 20 views': (9.346, 26.322, 30.297, 28.403),
    'POV': (8.943, 25.425, 29.159, 28.158),
}

training = {
    'OV': (0.528, 0.810, 1.342, 0.841),
    'PV 5 views': (0.376, 0.740, 1.256, 3.005),
    'PV 20 views': (1.389, 2.861, 4.779, 11.874),
    'POV': (0.370, 0.737, 1.247, 3.006),
}

x = np.arange(len(species))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

plt.rcParams.update({'font.size': 16.5})
fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in training.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, align='center')
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Training Time (s)')
# ax.set_title('Pre-processing Time')
ax.set_xticks(x + 0.3, species)
# plt.xticks(rotation=45)
plt.legend(loc='upper left', ncol=2)
ax.set_ylim(0, 14)

plt.show()
fig.savefig(f"train_time.pdf", bbox_inches='tight')

plt.rcParams.update({'font.size': 16.5})
fig, ax = plt.subplots(layout='constrained')
for attribute, measurement in pre_proc.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute, align='center')
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Preprocessing Time (s)')
# ax.set_title('Pre-processing Time')
ax.set_xticks(x + 1.1, species)
# plt.xticks(rotation=45)
plt.legend(loc='upper left', ncol=2)
ax.set_ylim(0, 40)

plt.show()
fig.savefig(f"prep_time.pdf", bbox_inches='tight')