# draws reports/survival_design.png (the picture of how patients are followed in step 5, used in the readme)
# the 6 patients are made up, only to show the rules. only needs to be run again if the rules change

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import settings

# same colours as the study design picture: light = look-back for the features, dark = follow-up
TRAIN_LOOKBACK, TRAIN_FOLLOW = "#9ecae1", "#3182bd"
TEST_LOOKBACK, TEST_FOLLOW = "#fdd0a2", "#e6550d"

TRAIN_CLOSES = 2024.0     # 1 Jan 2024: training rows are not followed past this date
DATA_ENDS = 2026.75       # 1 Oct 2026: end of the Synthea data

# (label, snapshot year, end of follow-up, what ended it, is it a test snapshot)
# what ended it: "diagnosis" = event, anything else = censored
patients = [
    ("Snapshot 1 Jan 2020, patient A\ndiagnosed", 2020, 2021.6, "diagnosis", False),
    ("Snapshot 1 Jan 2020, patient B\ndies, no diagnosis before", 2020, 2022.1, "censored", False),
    ("Snapshot 1 Jan 2020, patient C\nno diagnosis in the 3-year limit", 2020, 2023.0, "censored", False),
    ("Snapshot 1 Jan 2023, patient D\nno diagnosis when training closes", 2023, 2024.0, "censored", False),
    ("Snapshot 1 Jan 2024, patient E\ndiagnosed", 2024, 2025.3, "diagnosis", True),
    ("Snapshot 1 Jan 2024, patient F\nno diagnosis at the end of the data", 2024, DATA_ENDS, "censored", True),
]

fig, ax = plt.subplots(figsize=(10, 5.5))

for row, (label, year, end, how, test) in enumerate(patients):
    # 2 years before the snapshot: the features
    ax.barh(row, 2, left=year - 2, color=TEST_LOOKBACK if test else TRAIN_LOOKBACK, edgecolor="white")
    # from the snapshot until the end of the follow-up
    ax.barh(row, end - year, left=year, color=TEST_FOLLOW if test else TRAIN_FOLLOW, edgecolor="white")
    # snapshot (index date)
    ax.plot([year, year], [row - 0.4, row + 0.4], color="black", linewidth=2)
    # end of the follow-up: X = event, open circle = censored
    if how == "diagnosis":
        ax.plot(end, row, marker="X", color="black", markersize=10)
    else:
        ax.plot(end, row, marker="o", markerfacecolor="white", markeredgecolor="black", markersize=9)

ax.set_yticks(range(len(patients)))
ax.set_yticklabels([p[0] for p in patients], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(2017.8, 2027.2)
ax.set_ylim(len(patients) - 0.4, -1.1)
ax.set_xticks(range(2018, 2028))

# the 2 dates that close the follow-up
for x, text in [(TRAIN_CLOSES, "training data\ncloses 1 Jan 2024"), (DATA_ENDS, "end of data\n1 Oct 2026")]:
    ax.axvline(x, color="grey", linestyle="--", linewidth=1)
    ax.text(x, -0.55, text, ha="center", va="center", fontsize=8, color="dimgrey")

ax.set_title("How patients are followed in the survival analysis (made-up example patients)\n"
             "follow-up stops at the diagnosis, or is censored: death, close of the data, 3-year limit", fontsize=10)

legend_items = [
    Patch(color=TRAIN_LOOKBACK, label="Look-back, 2 years (train): features"),
    Patch(color=TEST_LOOKBACK, label="Look-back, 2 years (test): features"),
    Patch(color=TRAIN_FOLLOW, label="Follow-up (train)"),
    Patch(color=TEST_FOLLOW, label="Follow-up (test)"),
    Line2D([0], [0], marker="X", color="black", linestyle="", markersize=9, label="Diagnosis = event"),
    Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="black", linestyle="",
           markersize=8, label="Censored: follow-up stops, no diagnosis seen"),
]
ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=3, fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(settings.REPORTS_DIR, "survival_design.png"), dpi=120)
print("saved reports/survival_design.png")
