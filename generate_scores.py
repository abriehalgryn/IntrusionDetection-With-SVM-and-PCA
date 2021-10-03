import matplotlib.pyplot as plt
import numpy as np


EXACT_ATTACK_NAME = 1
CATEGORY_NAME = 2
ATTACK_YES_NO = 3
TRAINING_TIME = 4

PERCENTAGE = 0.15

# connects line between average points
PLOT_LINE = True

TESTING_TARGET = EXACT_ATTACK_NAME

TEST_PARAMETER = "n_eigenvectors"

COLORS = [
    "red", "green", "blue", "orange",
    "purple", "yellow", "cyan"
]

import utils

TITLES = {
    "n_eigenvectors": "Number of Eigenvectors",
    "normalize_method": "Normalization method",
    "kernel": "Kernel",
    "category_classifications": "Category classifications",
    "whiten_eigenvectors": "Whitened Eigenvectors",
    "gamma": "Gamma",
    "C": "C"
}

structure = [
    "n_eigenvectors",
    "normalize_method",
    "kernel",
    "category_classifications",
    "whiten_eigenvectors",
    "gamma",
    "C"
]

with open("scores.txt", "r") as f:
    contents = f.read()

records = []
for line in contents.split("\n"):
    if line == "":
        continue
    words = line.split(" ")
    record = [
        words[0],
        float(words[1]),
        float(words[2]),
        float(words[3]),
        float(words[4])
    ]
    records.append(record)

for record in records:
    description = record[0]

    settings = {}
    for i, setting in enumerate(description.split("__")):
        setting = setting.replace("_", ".")

        if utils.isnumeric(setting):
            setting = float(setting)

        settings[structure[i]] = setting
        record[0] = settings



plot_scores = {}
plot_times = {}

for record in records:
    value = record[0][TEST_PARAMETER]
    if value not in plot_scores:
        plot_scores[value] = []
        plot_times[value] = []
    plot_scores[value].append(record[TESTING_TARGET] * 100)
    plot_times[value].append(record[-1])


for i, value in enumerate(plot_scores):
    if PERCENTAGE < 1.0:
        n = len(plot_scores[value])
        indexes = list(range(n))
        indexes.sort(key=lambda x: plot_scores[value][x], reverse=True)
        n = int(PERCENTAGE * n)
        indexes = indexes[:n]
        plot_scores[value] = [
            plot_scores[value][i] for i in indexes
        ]
        plot_times[value] = [
            plot_times[value][i] for i in indexes
        ]

    plt.scatter(plot_scores[value], plot_times[value], color=COLORS[i], s=15, alpha=0.9, label=str(value))

records.sort(key=lambda r: r[TESTING_TARGET], reverse=True)
n_records = int(len(records) * PERCENTAGE)
records = records[:n_records]


avg_scores = []
avg_times = []
colors = []
for i, value in enumerate(plot_scores):
    avg_scores.append(
        sum(plot_scores[value]) / len(plot_scores[value])
    )
    avg_times.append(
        sum(plot_times[value]) / len(plot_times[value])
    )
    colors.append(COLORS[i])

for i, (score, time) in enumerate(zip(avg_scores, avg_times)):
    plt.scatter(score, time, color=COLORS[i])

if PLOT_LINE:
    plt.plot(avg_scores, avg_times, color="black")


#plt.scatter(x, y)
#plt.plot(x_avg, y_avg, color="orange")
title = f"{TITLES[TEST_PARAMETER]}: training time (s) vs. accuracy (%)"
plt.title(title, loc="center")
plt.legend(loc="upper left")

plt.show()
