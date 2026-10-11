"""Figures for the report, read from results/*.csv and saved to figures/ as PDF and PNG.

Usage: python plots.py [e1] [e2] [e5] [e6] [e8] [q]   (default: all figures whose CSVs exist)
Colours: a colour-blind-checked categorical palette; every series also has its own marker shape.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
DATASETS = ["Cora", "Citeseer", "Pubmed", "Texas", "Chameleon", "Photo", "Computers", "LastFMAsia"]
# blue, orange, aqua, yellow, magenta, green, violet, red (validated categorical order)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
DATASET_STYLE = {d: dict(color=c, marker=m) for d, c, m in zip(DATASETS, PALETTE, MARKERS)}
METHODS = ["MLP + C&S", "GCN", "LP", "MLP", "MLP + S"]
METHOD_STYLE = {m: dict(color=c, marker=k) for m, c, k in zip(METHODS, PALETTE, MARKERS)}
METHODS_E1 = METHODS[:4]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.labelcolor": "#0b0b0b", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5, "lines.markersize": 4,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
})
SIZE = (3.4, 2.4)                                               # single-column width in inches


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"saved figures/{name}.pdf/.png")


def gain_vs_homophily(all_datasets=False):
    """C&S gain over the plain MLP (per-split difference) against edge homophily, one point per dataset.

    all_datasets=False: the five E2 datasets (10 splits each). True: also the E5 datasets (3 splits),
    with each point labelled directly, because eight colours cannot all be told apart in a scatter.
    """
    files = ["reproduction_runs.csv", "e2_runs.csv"] + (["e5_runs.csv"] if all_datasets else [])
    runs = pd.concat([pd.read_csv(RESULTS / f) for f in files])
    acc = runs[runs.method.isin(["MLP", "MLP + C&S"])].pivot_table(
        index=["dataset", "seed"], columns="method", values="test_acc")
    gain = (acc["MLP + C&S"] - acc["MLP"]).groupby("dataset").agg(["mean", "std"])
    h = pd.read_csv(RESULTS / "homophily.csv", index_col="dataset").edge_homophily
    datasets = [d for d in DATASETS if d in gain.index]

    fig, ax = plt.subplots(figsize=SIZE)
    ax.axhline(0, color="#52514e", linewidth=0.8, linestyle="--")
    for d in datasets:
        ax.errorbar(h[d], gain.loc[d, "mean"], yerr=gain.loc[d, "std"], linestyle="none", capsize=2,
                    markersize=6, label=d, **DATASET_STYLE[d])
        if all_datasets:
            dx, dy, ha = LABEL_OFFSET.get(d, (6, 0, "left"))
            ax.annotate(d, (h[d], gain.loc[d, "mean"]), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=6.5, color="#0b0b0b")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Edge homophily (share of same-label edges)")
    ax.set_ylabel("MLP + C&S minus MLP (points)")
    ax.legend(loc="lower right" if all_datasets else "upper left", ncol=2 if all_datasets else 1)
    save(fig, "e5_gain_vs_homophily_all" if all_datasets else "e2_gain_vs_homophily")


# Direct-label offsets (points) for the all-dataset gain figure; chosen so labels do not overlap.
LABEL_OFFSET = {"Citeseer": (-7, 0, "right"), "Computers": (7, 3, "left"), "Photo": (7, 2, "left"),
                "Pubmed": (-7, -2, "right")}


def e2_accuracy_vs_alpha():
    """MLP + S test accuracy at fixed alpha_smooth, one line per dataset (alpha 0 = plain MLP)."""
    sweep = pd.read_csv(RESULTS / "e2_smooth_sweep.csv")
    fig, ax = plt.subplots(figsize=SIZE)
    for d in [d for d in DATASETS if d in set(sweep.dataset)]:
        g = sweep[sweep.dataset == d].groupby("alpha_s").test_acc
        mean, std = g.mean(), g.std()
        ax.plot(mean.index, mean.values, label=d, **DATASET_STYLE[d])
        ax.fill_between(mean.index, mean - std, mean + std, color=DATASET_STYLE[d]["color"], alpha=0.15, linewidth=0)
    ax.set_xlabel(r"Smoothing strength $\alpha_{\mathrm{smooth}}$ (0 = plain MLP)")
    ax.set_ylabel("MLP + S test accuracy (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)   # below the axes: keeps column width
    save(fig, "e2_accuracy_vs_alpha")


def e1_label_scarcity(dataset):
    """Test accuracy against training-label share for the four E1 methods, shaded +-1 std over splits."""
    runs = pd.read_csv(RESULTS / "e1_runs.csv")
    runs = runs[runs.dataset == dataset]
    fig, ax = plt.subplots(figsize=SIZE)
    for m in METHODS_E1:
        g = runs[runs.method == m].groupby("train_frac").test_acc
        share, mean, std = 100 * g.mean().index, g.mean().values, g.std().values
        ax.plot(share, mean, label=m, **METHOD_STYLE[m])
        ax.fill_between(share, mean - std, mean + std, color=METHOD_STYLE[m]["color"], alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ticks = sorted(100 * runs.train_frac.unique())
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Training labels (% of nodes, log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(loc="lower right")
    save(fig, f"e1_label_scarcity_{dataset.lower()}")


def e6_gain_by_group(grouping, xlabel, groups):
    """Accuracy change of MLP + C&S over MLP per node group, one bar per dataset.

    Faded bars: groups with fewer than 20 test nodes pooled over the 5 splits (too few to trust).
    """
    summary = pd.read_csv(RESULTS / "e6_summary.csv")
    summary = summary[summary.grouping == grouping]
    datasets = [d for d in DATASETS if d in set(summary.dataset)]
    width = 0.8 / len(datasets)

    fig, ax = plt.subplots(figsize=SIZE)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    for j, d in enumerate(datasets):
        g = summary[summary.dataset == d].set_index("group")
        for i, grp in enumerate(groups):
            if grp not in g.index:
                continue                                        # no test nodes in this group
            r = g.loc[grp]
            x, gain, alpha = i + (j - (len(datasets) - 1) / 2) * width, r.cs_acc - r.mlp_acc, 1.0 if r.n_nodes >= 20 else 0.3
            ax.bar(x, gain, width * 0.9, color=DATASET_STYLE[d]["color"], alpha=alpha)
            if gain == 0:                                       # exactly zero: show a tick so it differs from "no nodes"
                ax.plot(x, 0, marker="_", markersize=7, markeredgewidth=2.5, color=DATASET_STYLE[d]["color"], alpha=alpha)
    ax.set_xticks(range(len(groups)), groups)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)                                      # grid lines behind the bars
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MLP + C&S minus MLP (points)")
    handles = [Patch(color=DATASET_STYLE[d]["color"], label=d) for d in datasets]   # full-opacity swatches
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    save(fig, f"e6_gain_by_{grouping}")


def e8_label_noise(dataset):
    """Clean-test accuracy against the share of flipped training labels, shaded +-1 std over splits."""
    runs = pd.read_csv(RESULTS / "e8_runs.csv")
    runs = runs[runs.dataset == dataset]
    fig, ax = plt.subplots(figsize=SIZE)
    for m in METHODS:
        g = runs[runs.method == m].groupby("label_noise").test_acc
        p, mean, std = 100 * g.mean().index, g.mean().values, g.std().values
        ax.plot(p, mean, label=m, **METHOD_STYLE[m])
        ax.fill_between(p, mean - std, mean + std, color=METHOD_STYLE[m]["color"], alpha=0.15, linewidth=0)
    ax.set_xticks(sorted(100 * runs.label_noise.unique()))
    ax.set_xlabel("Flipped training labels (%)")
    ax.set_ylabel("Clean test accuracy (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    save(fig, f"e8_label_noise_{dataset.lower()}")


CLASS_MARKERS = MARKERS[:7]
OUTCOME_STYLE = {   # recessive greys for unchanged nodes; colour + shape for the nodes C&S changed
    "stayed right": dict(color="#c3c2b7", marker="o", s=6, zorder=1),
    "stayed wrong": dict(color="#52514e", marker="x", s=10, zorder=2, linewidths=0.8),
    "fixed": dict(color="#2a78d6", marker="^", s=18, zorder=3),
    "broken": dict(color="#eb6834", marker="v", s=18, zorder=3),
}


def tsne_axes(ax):
    """t-SNE coordinates have no units: keep axis names, drop the tick values."""
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel("t-SNE 1"), ax.set_ylabel("t-SNE 2")
    ax.grid(False)


def q_tsne_stages(dataset):
    """Three panels: test nodes embedded from Z, after Correct, after Smooth; coloured by true label."""
    pts = pd.read_csv(RESULTS / "q_points.csv").query("dataset == @dataset")
    stages = list(dict.fromkeys(pts.stage))
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5))         # full text width: three panels side by side
    for ax, letter, stage in zip(axes, "abc", stages):
        g = pts[pts.stage == stage]
        for lab in sorted(g.label.unique()):
            h = g[g.label == lab]
            ax.scatter(h.x, h.y, s=8, color=PALETTE[lab], marker=CLASS_MARKERS[lab], linewidths=0, label=f"class {lab}")
        acc = 100 * (g.pred == g.label).mean()
        ax.text(0.0, 1.02, f"({letter}) {stage}, test acc {acc:.1f}%", transform=ax.transAxes,
                va="bottom", fontsize=7, color="#0b0b0b")             # panel label above the axes, off the data
        tsne_axes(ax)
        if letter != "a":
            ax.set_ylabel("")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.65, -0.12), ncol=7, markerscale=1.5,
                   handletextpad=0.1, columnspacing=0.8)
    save(fig, f"q_tsne_stages_{dataset.lower()}")


def q_outcomes(dataset):
    """Test nodes at their t-SNE position from the MLP output Z, coloured by what C&S did to them."""
    pts = pd.read_csv(RESULTS / "q_points.csv").query("dataset == @dataset and stage == 'Z'")
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for name, style in OUTCOME_STYLE.items():
        g = pts[pts.outcome == name]
        ax.scatter(g.x, g.y, label=f"{name} ({len(g)})", **style)
    tsne_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, markerscale=1.3)
    save(fig, f"q_outcomes_{dataset.lower()}")


def main():
    which = sys.argv[1:] or ["e1", "e2", "e5", "e6", "e8", "q"]
    if "e6" in which and (RESULTS / "e6_summary.csv").exists():
        e6_gain_by_group("degree", "Node degree", ["0", "1", "2-3", "4-7", "8+"])
        e6_gain_by_group("local_homophily", "Local homophily (share of same-label neighbours)",
                         ["isolated", "0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"])
    if "e2" in which and (RESULTS / "e2_smooth_sweep.csv").exists():
        gain_vs_homophily()
        e2_accuracy_vs_alpha()
    if "e5" in which and (RESULTS / "e5_runs.csv").exists():
        gain_vs_homophily(all_datasets=True)
    if "e8" in which and (RESULTS / "e8_runs.csv").exists():
        for d in ["Cora", "Citeseer"]:
            e8_label_noise(d)
    if "q" in which and (RESULTS / "q_points.csv").exists():
        for d in ["Cora", "Texas"]:
            q_tsne_stages(d)
            q_outcomes(d)
    if "e1" in which and (RESULTS / "e1_runs.csv").exists():
        for d in ["Cora", "Citeseer"]:
            e1_label_scarcity(d)


if __name__ == "__main__":
    main()
