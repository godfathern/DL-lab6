"""Build the report's figures from the training history and sweep results."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "figures")
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})

hist = json.load(open(os.path.join(ROOT, "results/outputs/history.json")))
final = json.load(open(os.path.join(ROOT, "outputs/final/results.json")))
tstart = [json.loads(l) for l in open(os.path.join(ROOT, "outputs/tstart_sweep.jsonl"))]


def two_series(ax, x, a, b):
    ax.plot(x, a, color=BLUE, marker="o", markersize=4, label="test.json")
    ax.plot(x, b, color=ORANGE, marker="s", markersize=4, label="new_test.json")


# Figure 1: training loss.
fig, ax = plt.subplots(figsize=(3.3, 2.3))
ax.plot([r["epoch"] + 1 for r in hist], [r["loss"] for r in hist], color=BLUE)
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("noise-prediction MSE")
fig.savefig(os.path.join(OUT, "loss.pdf"))

# Figure 2: accuracy measured during training (50 DDIM steps, CFG 2, start at t=999).
ev = [r for r in hist if "test" in r]
fig, ax = plt.subplots(figsize=(3.3, 2.3))
two_series(ax, [r["epoch"] + 1 for r in ev], [r["test"] for r in ev], [r["new_test"] for r in ev])
ax.set_xlabel("epoch")
ax.set_ylabel("evaluator accuracy")
ax.set_ylim(0.3, 0.75)
ax.legend(loc="lower right")
fig.savefig(os.path.join(OUT, "train_acc.pdf"))

# Figure 3: sampling sweeps, sharing one accuracy axis.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
two_series(a1, [r["t_start"] for r in tstart], [r["test"] for r in tstart], [r["new_test"] for r in tstart])
a1.set_xticks([r["t_start"] for r in tstart])
a1.invert_xaxis()
a1.set_xlabel("first sampling timestep (CFG 3)")
a1.set_ylabel("evaluator accuracy")
cfg = [r for r in final["sweeps"] if r["cg"] == 0]
two_series(a2, [r["cfg"] for r in cfg], [r["test"] for r in cfg], [r["new_test"] for r in cfg])
a2.set_xlabel("CFG scale w (start t=960)")
for ax in (a1, a2):
    ax.axhline(0.8, color=MUTED, linewidth=0.8, linestyle="--")
a1.text(975, 0.805, "0.8 grading band", color=MUTED, fontsize=7, va="bottom")
a1.set_ylim(0.6, 0.95)
a2.legend(loc="lower right")
fig.savefig(os.path.join(OUT, "sweeps.pdf"))
print("figures written to", OUT)
