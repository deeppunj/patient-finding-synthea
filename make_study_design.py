# draws reports/study_design.png (the picture of the snapshot set-up used in the readme)
# only needs to be run again if the years or windows change

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import settings

# colours: light = look-back window, dark = outcome window, blue = train, orange = test
TRAIN_LOOKBACK, TRAIN_OUTCOME = "#9ecae1", "#3182bd"
TEST_LOOKBACK, TEST_OUTCOME = "#fdd0a2", "#e6550d"

years = list(range(2019, 2026))

fig, ax = plt.subplots(figsize=(9, 5))

for row, year in enumerate(years):
    test = year >= 2024
    # 2 years before the index date
    ax.barh(row, 2, left=year - 2, color=TEST_LOOKBACK if test else TRAIN_LOOKBACK, edgecolor="white")
    # 12 months after
    ax.barh(row, 1, left=year, color=TEST_OUTCOME if test else TRAIN_OUTCOME, edgecolor="white")
    # index date
    ax.plot([year, year], [row - 0.4, row + 0.4], color="black", linewidth=2)

ax.set_yticks(range(len(years)))
ax.set_yticklabels(["Snapshot 1 Jan %d" % y for y in years])
ax.invert_yaxis()   # 2019 on top
ax.set_xlim(2016.8, 2026.2)
ax.set_xticks(range(2017, 2027))

# line between the train snapshots (2019-2023) and the test snapshots (2024-2025)
ax.axhline(4.5, color="grey", linestyle="--", linewidth=1)
ax.text(2016.9, 2.0, "TRAIN", rotation=90, va="center", fontsize=10, color=TRAIN_OUTCOME, fontweight="bold")
ax.text(2016.9, 5.5, "TEST", rotation=90, va="center", fontsize=10, color=TEST_OUTCOME, fontweight="bold")

ax.set_title("Study design: yearly snapshots, features from the past only,\n"
             "train on 2019-2023 and test on 2024-2025", fontsize=11)

# legend: 3 columns -> (look-back train, look-back test), (outcome train, outcome test), (index date, split)
legend_items = [
    Patch(color=TRAIN_LOOKBACK, label="Look-back, 2 years (train): features"),
    Patch(color=TEST_LOOKBACK, label="Look-back, 2 years (test): features"),
    Patch(color=TRAIN_OUTCOME, label="Outcome window, 12 months (train): label"),
    Patch(color=TEST_OUTCOME, label="Outcome window, 12 months (test): label"),
    Line2D([0], [0], color="black", linewidth=2, label="Index date (snapshot)"),
    Line2D([0], [0], color="grey", linewidth=1, linestyle="--", label="Train / test split"),
]
ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.1),
          ncol=3, fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(settings.REPORTS_DIR, "study_design.png"), dpi=120)
print("saved reports/study_design.png")
