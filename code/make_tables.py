"""Generate every LaTeX table and every number quoted in the report from results/*.csv.

Writes report/tables/*.tex (booktabs tables) and report/tables/numbers.tex (one LaTeX macro per number
used in the text, e.g. \\nRepLinCSCora). No experiment is run here. Besides reading CSVs, the script only
counts dataset statistics once (cached in results/datasets.csv) and records the software/hardware
environment (cached in results/environment.csv). Hyperparameters are read from the code itself.
Usage: python make_tables.py
"""
import inspect
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

import data
import models
import propagate
import run_e1
import run_e2
import run_e7
import run_e8
import run_main
import train

ROOT = Path(__file__).resolve().parent.parent
RESULTS, TABLES = ROOT / "results", ROOT / "report" / "tables"
DATASETS = ["Cora", "Citeseer", "Pubmed", "Texas", "Chameleon", "Photo", "Computers"]
PAPER_STATS = {"Cora": (2708, 5429), "Citeseer": (3327, 4732), "Pubmed": (19717, 44338)}  # paper Table 1
GCN_EPOCHS_LONG = 1000      # value passed as --gcn-epochs for the fair-GCN re-run (results/run_e5_gcn1000_log.txt)
MACROS = {}

# ---------------------------------------------------------------- helpers

def key(text):
    """Turn a method/dataset/setting name into letters only, usable inside a LaTeX macro name."""
    for old, new in [("Linear", "Lin"), (" + C&S", "CS"), (" + C", "C"), (" + S", "S"),
                     ("GCN (300 epochs)", "GCNshort"), (",", ""), (" ", ""), ("-", "")]:
        text = text.replace(old, new)
    assert text.isalpha(), text
    return text


def macro(name, value):
    """Remember a number for numbers.tex and return it, so tables and text use the same string."""
    assert name.isalpha() and name not in MACROS, name
    MACROS[name] = value
    return value


def tex(text):
    """Escape the ampersand in method names such as 'MLP + C&S' for use inside a LaTeX table cell."""
    return text.replace("&", "\&")


def f2(x):
    return f"{x:.2f}"


def thousands(x):
    return f"{int(x):,}".replace(",", "{,}")


def pm(series, name=None, bold=False):
    """'mean $\\pm$ std' of a series of per-split values; optionally store mean and std as macros."""
    mean, std = series.mean(), series.std()
    m, s = f2(mean), f2(std)
    if name:
        macro(name, m)
        macro(name + "Sd", s)
    cell = f"{m} $\\pm$ {s}"
    return f"\\textbf{{{m}}} $\\pm$ {s}" if bold else cell


def table(name, caption, label, colspec, head, body, size="\\small"):
    """Write a booktabs table, shrunk to the text width only if it would not fit."""
    lines = ["\\begin{table}[htbp]", "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\begin{adjustbox}{max width=\\textwidth}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             *head, "\\midrule", *body, "\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report/tables/{name}.tex")


def best_mean(df, methods, col="method"):
    """Method with the highest mean test accuracy in df (used to bold the best entry)."""
    means = {m: df[df[col] == m].test_acc.mean() for m in methods}
    return max(means, key=means.get)


def splits(df, name):
    return macro(f"Splits{name}", str(df.seed.nunique()))


# ---------------------------------------------------------------- cached facts (not experiments)

def dataset_stats():
    path = RESULTS / "datasets.csv"
    if not path.exists():
        rows = []
        for d in DATASETS:
            X, y, edge_index = data.load_dataset(d, normalise=False)
            n = X.size(0)
            degree = edge_index[0].bincount(minlength=n)
            rows.append(dict(dataset=d, nodes=n, edges=edge_index.size(1) // 2, features=X.size(1),
                             classes=int(y.max()) + 1, edge_homophily=data.edge_homophily(edge_index, y),
                             isolated=int((degree == 0).sum()), zero_feature_rows=int((X.sum(1) == 0).sum()),
                             n_train=int(0.6 * n), n_val=int(0.2 * n), n_test=n - int(0.6 * n) - int(0.2 * n)))
        pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path).set_index("dataset")


def environment():
    path = RESULTS / "environment.csv"
    if not path.exists():
        import matplotlib, numpy, psutil, scipy, sklearn, torch, torch_geometric
        try:
            cpu = subprocess.run(["powershell", "-NoProfile", "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=30).stdout.strip()
        except OSError:
            cpu = platform.processor()
        info = {"CPU": cpu, "Physical cores / threads": f"{psutil.cpu_count(logical=False)} / {psutil.cpu_count()}",
                "RAM": f"{psutil.virtual_memory().total / 2**30:.0f} GiB", "GPU": "none (CPU only)",
                "Operating system": f"{platform.system()} {platform.release()}",
                "Python": sys.version.split()[0], "PyTorch": torch.__version__,
                "PyTorch Geometric": torch_geometric.__version__, "NumPy": numpy.__version__,
                "SciPy": scipy.__version__, "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
                "matplotlib": matplotlib.__version__}
        pd.Series(info, name="value").rename_axis("item").to_csv(path)
    return pd.read_csv(path, index_col="item").value


# ---------------------------------------------------------------- tables

def tab_datasets():
    st = dataset_stats()
    body = []
    for d in DATASETS:
        r = st.loc[d]
        paper = (f"{thousands(PAPER_STATS[d][0])} / {thousands(PAPER_STATS[d][1])}" if d in PAPER_STATS else "--")
        h = macro(f"Hom{d}", f"{r.edge_homophily:.2f}")
        for col, mname in [("nodes", "Nodes"), ("edges", "Edges"), ("features", "Feats"), ("classes", "Classes"),
                           ("isolated", "Iso"), ("n_test", "Test")]:
            macro(f"{mname}{d}", thousands(r[col]))
        body.append(f"{d} & {thousands(r.nodes)} & {thousands(r.edges)} & {thousands(r.features)} & {int(r.classes)} & "
                    f"{h} & {thousands(r.isolated)} & {thousands(r.n_train)} / {thousands(r.n_val)} / "
                    f"{thousands(r.n_test)} & {paper} \\\\")
    head = ["Dataset & Nodes & Edges & Features & Classes & $h$ & Isolated & Train / Val / Test & "
            "Paper: nodes / edges \\\\"]
    table("datasets", "Datasets used in this report. Edges are undirected, each counted once, after removing "
          "self-loops and duplicates; $h$ is the edge homophily (share of edges joining two nodes of the same "
          "class); Train / Val / Test are the node counts of one " + MACROS["Split"] + " split. The last column repeats the "
          "counts reported by \\citet{huang2021cs} (their Table~1) for comparison only. LastFM Asia was planned "
          "but its download source no longer exists.", "tab:datasets", "lrrrrrrcc", head, body,
          size="\\footnotesize")


def tab_hparams():
    tm = inspect.signature(train.train_model).parameters
    mlp = inspect.signature(models.MLP.__init__).parameters
    gcn = inspect.signature(models.GCN.__init__).parameters
    iters = inspect.signature(propagate.propagate).parameters["iters"].default
    sp = inspect.signature(data.random_split).parameters
    tr, va = sp["train_frac"].default, sp["val_frac"].default
    macro("TrainPct", f"{100 * tr:g}"), macro("ValPct", f"{100 * va:g}"), macro("TestPct", f"{100 * (1 - tr - va):g}")
    macro("Split", f"{100 * tr:g}/{100 * va:g}/{100 * (1 - tr - va):g}")
    lin, pub = run_main.DEFAULT_LINEAR, run_main.LINEAR_CFG["Pubmed"]
    alphas = ", ".join(f"{a:g}" for a in run_main.ALPHAS)
    macro("LR", f"{tm['lr'].default:g}"), macro("Epochs", str(tm["epochs"].default))
    macro("Hidden", str(mlp["hidden"].default)), macro("Layers", str(mlp["n_layers"].default))
    macro("Dropout", f"{mlp['dropout'].default:g}"), macro("Iters", str(iters))
    macro("AlphaGrid", alphas), macro("EpochsLong", str(GCN_EPOCHS_LONG))
    macro("LinEpochs", str(lin["epochs"])), macro("LinLR", f"{lin['lr']:g}"), macro("LinWD", f"{lin['weight_decay']:g}")
    macro("LinPubLR", f"{pub['lr']:g}"), macro("LinPubWD", f"{pub['weight_decay']:g}")
    feats = lambda cfg: "row-normalised" if cfg["normalise"] else "raw (unnormalised)"
    rows = [
        ("Optimiser", "Adam, full batch", "all trained models"),
        ("Learning rate", f"{tm['lr'].default:g}", "MLP, GCN"),
        ("Weight decay", f"{tm['weight_decay'].default:g}", "MLP, GCN"),
        ("Training epochs", f"{tm['epochs'].default}; best validation epoch kept", "MLP, GCN (E4 re-run: "
         f"{GCN_EPOCHS_LONG})"),
        ("MLP", f"{mlp['n_layers'].default} layers, {mlp['hidden'].default} hidden units, batch norm, ReLU, "
                f"dropout {mlp['dropout'].default:g}", "base predictor"),
        ("GCN", f"2 layers, {gcn['hidden'].default} hidden units, dropout {gcn['dropout'].default:g}", "baseline"),
        ("Linear (all except Pubmed)", f"{feats(lin)} features, lr {lin['lr']:g}, weight decay {lin['weight_decay']:g}, "
                                        f"{lin['epochs']} epochs", "base predictor"),
        ("Linear (Pubmed)", f"{feats(pub)} features, lr {pub['lr']:g}, weight decay {pub['weight_decay']:g}, "
                            f"{pub['epochs']} epochs", "base predictor"),
        ("Propagation iterations", str(iters), "LP, Correct, Smooth"),
        ("$\\alpha$ grid (chosen on validation)", alphas, "LP, $\\alpha_c$, $\\alpha_s$"),
        ("Training share", ", ".join(f"{100 * s:g}\\%" for s in run_e1.SHARES), "E2"),
        ("Fixed $\\alpha_s$ sweep", ", ".join(f"{a:g}" for a in run_e2.SWEEP), "E3"),
        ("$\\alpha_s$ grid with ``no smoothing''", ", ".join(f"{a:g}" for a in run_e7.EXTENDED_ALPHAS_S), "E6"),
        ("Autoscale cap \\texttt{max\\_scale}", ", ".join("none" if m is None else str(m) for m in run_e7.MAX_SCALES), "E6"),
        ("Flipped training labels $p$", ", ".join(f"{p:g}" for p in run_e8.NOISE), "E7"),
    ]
    body = [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    table("hparams", "Hyperparameters, read directly from the code. The Linear settings were chosen on validation "
          "accuracy only (Section~\\ref{sec:setup}); nothing else was tuned.", "tab:hparams",
          "l>{\\raggedright\\arraybackslash}p{6.6cm}l",
          ["Setting & Value & Used by \\\\"], body)


def tab_environment():
    env = environment()
    for k, name in [("CPU", "CPU"), ("RAM", "RAM"), ("PyTorch", "TorchVersion"), ("PyTorch Geometric", "PygVersion"),
                    ("Python", "PythonVersion")]:
        macro(name, env[k].replace("(R)", "").replace("(TM)", ""))
    body = [f"{k} & {v.replace('(R)', '').replace('(TM)', '')} \\\\" for k, v in env.items()]
    table("environment", "Hardware and software used for every experiment.", "tab:environment", "ll",
          ["Item & Value \\\\"], body)


def tab_reproduction():
    runs = pd.read_csv(RESULTS / "reproduction_runs.csv")
    splits(runs, "Rep")
    methods = ["Linear", "MLP", "LP", "GCN", "Linear + C", "Linear + C&S", "MLP + C&S"]
    best = {d: best_mean(runs[runs.dataset == d], methods) for d in DATASETS[:3]}
    body = []
    for m in methods:
        cells = []
        for i, d in enumerate(DATASETS[:3]):
            g = runs[(runs.dataset == d) & (runs.method == m)].test_acc
            paper = run_main.PAPER[m][i]
            cells += [pm(g, f"Rep{key(m)}{d}", bold=best[d] == m), macro(f"Pap{key(m)}{d}", f2(paper))]
            macro(f"Gap{key(m)}{d}", f"{g.mean() - paper:+.2f}")
        body.append(f"{tex(m)} & " + " & ".join(cells) + " \\\\")
    head = ["& \\multicolumn{2}{c}{Cora} & \\multicolumn{2}{c}{Citeseer} & \\multicolumn{2}{c}{Pubmed} \\\\",
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
            "Method & Ours & Paper & Ours & Paper & Ours & Paper \\\\"]
    table("reproduction", f"Reproduction: test accuracy (\\%), mean $\\pm$ std over {MACROS['SplitsRep']} random "
          f"{MACROS['Split']} splits; best of ours in bold. The \\emph{{Paper}} columns are the numbers reported by "
          "\\citet{huang2021cs} (Tables~2, 5 and~6), copied for comparison; they were not produced by us. Our GCN "
          "is a plain two-layer GCN (Section~\\ref{sec:method}), whereas the paper's GCN adds residual connections.",
          "tab:reproduction", "lcccccc", head, body)


def tab_ablation():
    runs = pd.read_csv(RESULTS / "reproduction_runs.csv")
    body = []
    for base in ["Linear", "MLP"]:
        for m in [base, f"{base} + C", f"{base} + S", f"{base} + C&S"]:
            cells = [pm(runs[(runs.dataset == d) & (runs.method == m)].test_acc, f"Abl{key(m)}{d}")
                     for d in DATASETS[:3]]
            body.append(f"{tex(m)} & " + " & ".join(cells) + " \\\\")
        if base == "Linear":
            body.append("\\addlinespace")
    table("ablation", f"Step ablation (E1): test accuracy (\\%), mean $\\pm$ std over {MACROS['SplitsRep']} splits. "
          "+\\,C: Correct only; +\\,S: Smooth applied directly to the base prediction; +\\,C\\&S: both.",
          "tab:ablation", "lccc", ["Method & Cora & Citeseer & Pubmed \\\\"], body)


def tab_cost():
    """Timing and parameter macros (the cost table itself is no longer part of the report)."""
    runs = pd.read_csv(RESULTS / "reproduction_runs.csv")
    methods = ["Linear", "MLP", "GCN", "LP", "Linear + C&S", "MLP + C&S"]
    body = []
    for d in DATASETS[:3]:                                      # long format: one block of rows per dataset
        for i, m in enumerate(methods):
            g = runs[(runs.dataset == d) & (runs.method == m)]
            params = thousands(g.params.iloc[0])
            train_s = "--" if m == "LP" else f"{g.train_time_s.mean():.1f}"
            prop = f"{1000 * g.prop_time_s.mean():.1f}" if g.prop_time_s.notna().any() else "--"
            macro(f"Params{key(m)}{d}", params)
            if train_s != "--":
                macro(f"Train{key(m)}{d}", train_s)
            if prop != "--":
                macro(f"Prop{key(m)}{d}", prop)
            body.append(f"{d if i == 0 else ''} & {tex(m)} & {params} & {train_s} & {prop} \\\\")
        if d != DATASETS[2]:
            body.append("\\addlinespace")


SHARE_KEY = {0.01: "One", 0.02: "Two", 0.05: "Five", 0.10: "Ten", 0.30: "Thirty", 0.60: "Sixty"}


def tab_e1():
    runs = pd.read_csv(RESULTS / "e1_runs.csv")
    splits(runs, "Lab")
    shares = sorted(runs.train_frac.unique())
    body = []
    for d in ["Cora", "Citeseer"]:                              # rows: dataset x share; columns: methods
        for i, s in enumerate(shares):
            cells = [pm(runs[(runs.dataset == d) & (runs.method == m) & (runs.train_frac == s)].test_acc,
                        f"Lab{key(m)}{d}{SHARE_KEY[round(s, 2)]}") for m in run_e1.REPORT]
            body.append(f"{d if i == 0 else ''} & {100 * s:g}\\% & " + " & ".join(cells) + " \\\\")
        if d == "Cora":
            body.append("\\addlinespace")
    head = ["Dataset & Training labels & " + " & ".join(tex(m) for m in run_e1.REPORT) + " \\\\"]
    table("e1", f"E2, label scarcity: test accuracy (\\%), mean $\\pm$ std over {MACROS['SplitsLab']} splits. "
          "Training labels: share of all nodes used for training. "
          f"Validation is always {MACROS['ValPct']}\\% of the nodes; the remaining nodes are the test set.", "tab:e1",
          "lrcccc", head, body)


def tab_e2():
    runs = pd.read_csv(RESULTS / "e2_runs.csv")
    splits(runs, "Het")
    st = dataset_stats()
    lows = run_e2.LOW_HOMOPHILY
    best = {d: best_mean(runs[runs.dataset == d], run_e2.REPORT) for d in lows}
    body = [f"{tex(m)} & " + " & ".join(pm(runs[(runs.dataset == d) & (runs.method == m)].test_acc,
                                       f"Het{key(m)}{d}", bold=best[d] == m) for d in lows) + " \\\\"
            for m in run_e2.REPORT]
    head = ["Method & " + " & ".join(f"{d} ($h = {st.loc[d].edge_homophily:.2f}$)" for d in lows) + " \\\\"]
    table("e2", f"E3, low-homophily graphs: test accuracy (\\%), mean $\\pm$ std over {MACROS['SplitsHet']} splits, "
          "same settings and $\\alpha$ grid as the reproduction; best in bold.", "tab:e2", "lcc", head, body)

    sweep = pd.read_csv(RESULTS / "e2_smooth_sweep.csv")
    alphas = sorted(sweep.alpha_s.unique())
    akey = {0.0: "Zero", 0.1: "One", 0.3: "Three", 0.5: "Five", 0.7: "Seven", 0.8: "Eight", 0.9: "Nine", 0.99: "NineNine"}
    body = []
    for d in run_e2.ALL:
        means = sweep[sweep.dataset == d].groupby("alpha_s").test_acc.mean()
        top = means.idxmax()
        cells = []
        for a in alphas:
            v = macro(f"Sweep{d}A{akey[a]}", f2(means[a]))
            cells.append(f"\\textbf{{{v}}}" if a == top else v)
        body.append(f"{d} & " + " & ".join(cells) + " \\\\")
    head = ["& \\multicolumn{%d}{c}{$\\alpha_s$ (fixed, not chosen on validation)} \\\\" % len(alphas),
            f"\\cmidrule(lr){{2-{len(alphas) + 1}}}",
            "Dataset & " + " & ".join(f"{a:g}" for a in alphas) + " \\\\"]
    table("e2_sweep", f"E3, smoothing strength: MLP + S test accuracy (\\%) at a fixed $\\alpha_s$, mean over "
          f"{sweep.seed.nunique()} splits ($\\alpha_s = 0$ is the plain MLP); best per dataset in bold. Standard "
          "deviations are in \\texttt{results/e2\\_smooth\\_sweep.csv} and Figure~\\ref{fig:e2alpha}.",
          "tab:e2sweep", "l" + "c" * len(alphas), head, body)


def tab_e5():
    runs = pd.read_csv(RESULTS / "e5_runs.csv")
    splits(runs, "Amz")
    st = dataset_stats()
    names = {"GCN (300 epochs)": f"GCN ({MACROS['Epochs']} epochs)", "GCN": f"GCN ({GCN_EPOCHS_LONG} epochs)"}
    order = ["MLP", "LP", "GCN (300 epochs)", "GCN", "MLP + S", "MLP + C&S"]
    best = {d: best_mean(runs[runs.dataset == d], order) for d in ["Photo", "Computers"]}
    body = []
    for m in order:
        cells = []
        for d in ["Photo", "Computers"]:
            g = runs[(runs.dataset == d) & (runs.method == m)]
            mk = "GCNlong" if m == "GCN" else key(m)
            cells.append(pm(g.test_acc, f"Amz{mk}{d}", bold=best[d] == m))
            if m in ("MLP", "GCN", "GCN (300 epochs)"):
                cells.append(macro(f"AmzTrain{mk}{d}", f2(g.base_train_acc.mean())))
            else:
                cells.append("--")
        body.append(f"{tex(names.get(m, m))} & " + " & ".join(cells) + " \\\\")
    for d in ["Photo", "Computers"]:                            # gain of C&S over its base model, in points
        g = runs[runs.dataset == d]
        macro(f"AmzGain{d}", f2(g[g.method == "MLP + C&S"].test_acc.mean() - g[g.method == "MLP"].test_acc.mean()))
    head = ["& \\multicolumn{2}{c}{Photo ($h = %.2f$)} & \\multicolumn{2}{c}{Computers ($h = %.2f$)} \\\\"
            % (st.loc["Photo"].edge_homophily, st.loc["Computers"].edge_homophily),
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
            "Method & Test acc. & Train acc. & Test acc. & Train acc. \\\\"]
    table("e5", f"E4, Amazon co-purchase graphs: accuracy (\\%), mean $\\pm$ std over {MACROS['SplitsAmz']} splits. "
          f"Both GCN rows are reported: the {MACROS['Epochs']}-epoch GCN (our default) under-fitted Computers, so it "
          f"was re-trained for up to {GCN_EPOCHS_LONG} epochs with nothing else changed. Train acc.: accuracy on the "
          "training nodes at the selected epoch.", "tab:e5", "lcccc", head, body)


GROUP_KEY = {"isolated": "Iso", "0-0.25": "Low", "0.25-0.5": "MidLow", "0.5-0.75": "MidHigh", "0.75-1": "High",
             "0": "DZero", "1": "DOne", "2-3": "DTwo", "4-7": "DFour", "8+": "DEight"}


def tab_e6():
    summ = pd.read_csv(RESULTS / "e6_summary.csv", dtype={"group": str})
    nodes = pd.read_csv(RESULTS / "e6_nodes.csv")
    macro("SplitsNode", str(nodes.seed.nunique()))
    body = []                                                   # one table, two panels stacked vertically
    for grouping, panel, order in [
            ("local_homophily", "(a) Grouped by local homophily (share of same-label neighbours)",
             ["isolated", "0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]),
            ("degree", "(b) Grouped by degree (number of neighbours)", ["0", "1", "2-3", "4-7", "8+"])]:
        if grouping == "degree":
            body.append("\\midrule")
        body.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{panel}}}}} \\\\")
        for d in ["Cora", "Citeseer", "Texas", "Chameleon"]:
            g = summ[(summ.grouping == grouping) & (summ.dataset == d)].set_index("group")
            first = True
            for grp in [x for x in order if x in g.index]:
                r = g.loc[grp]
                k = f"Node{d}{GROUP_KEY[grp]}"
                delta = macro(k + "Delta", f"{r.cs_acc - r.mlp_acc:+.1f}")
                macro(k + "N", str(int(r.n_nodes))), macro(k + "Fixed", str(int(r.fixed)))
                macro(k + "Broken", str(int(r.broken)))
                body.append(f"{d if first else ''} & {grp} & {int(r.n_nodes)} & {r.mlp_acc:.1f} & {r.cs_acc:.1f} & "
                            f"{delta} & {int(r.fixed)} & {int(r.broken)} \\\\")
                first = False
            if d != "Chameleon":
                body.append("\\addlinespace")
    head = ["Dataset & Group & Nodes & MLP & MLP + C\\&S & $\\Delta$ & Fixed & Broken \\\\"]
    table("e6", f"E5, test nodes pooled over {MACROS['SplitsNode']} splits and grouped (a) by local homophily and "
          "(b) by degree: accuracy (\\%) of the MLP and of MLP + C\\&S, their difference $\\Delta$ (points), and "
          "the number of nodes C\\&S fixed (wrong $\\to$ right) and broke (right $\\to$ wrong). Groups with few "
          "nodes are noisy.", "tab:e6", "llrrrrrr", head, body, size="\\footnotesize")


def tab_e7():
    runs = pd.read_csv(RESULTS / "e7_runs.csv")
    cap = runs[runs.part == "max_scale"]
    macro("SplitsCap", str(cap.seed.nunique()))
    settings = list(run_e7.SETTINGS)
    skey = {settings[0]: "Under", settings[1]: "Tuned", settings[2]: "Cora"}
    mkey = {"none": "None", "1": "One", "2": "Two", "5": "Five", "10": "Ten"}
    labels = {settings[0]: "Pubmed, under-fitted Linear", settings[1]: "Pubmed, re-tuned Linear",
              settings[2]: "Cora, standard Linear"}
    body = []
    for s in settings:                                          # rows: setting x cap; columns: + C, + C&S
        for i, ms in enumerate(["none", "1", "2", "5", "10"]):
            cells = [pm(cap[(cap.setting == s) & (cap.max_scale.astype(str) == ms) & (cap.method == m)].test_acc,
                        f"Cap{skey[s]}{key(m)}{mkey[ms]}") for m in ["Linear + C", "Linear + C&S"]]
            body.append(f"{labels[s] if i == 0 else ''} & {ms} & " + " & ".join(cells) + " \\\\")
        if s != settings[-1]:
            body.append("\\addlinespace")
    table("e7_cap", f"E6, the cap on the Autoscale factor (our change): Linear + C and Linear + C\\&S test accuracy (\\%), mean $\\pm$ "
          f"std over {MACROS['SplitsCap']} splits. ``none'' is the paper's Autoscale; a number is the largest factor "
          "allowed (\\texttt{max\\_scale}).", "tab:e7cap", "llcc",
          ["Setting & \\texttt{max\\_scale} & Linear + C & Linear + C\\&S \\\\"], body)

    body = []
    for s in settings:
        g = cap[(cap.setting == s) & (cap.method == "Linear") & (cap.max_scale.astype(str) == "none")]
        body.append(f"{labels[s]} & {pm(g.test_acc, f'Cap{skey[s]}Base')} & "
                    f"{macro(f'Cap{skey[s]}Sigma', f'{g.sigma.mean():.3f}')} & "
                    f"{macro(f'Cap{skey[s]}Median', f'{g.median_factor.mean():.1f}')} & "
                    f"{macro(f'Cap{skey[s]}Share', f'{g.share_factor_gt_2.mean():.0f}')} \\\\")
    table("e7_factor", f"E6, the base models behind Table~\\ref{{tab:e7cap}}, mean over {MACROS['SplitsCap']} splits: "
          "test accuracy of Linear alone, the mean training residual $\\sigma$, and the uncapped Autoscale factor "
          "$\\sigma/\\lVert\\hat e_i\\rVert_1$ on test nodes at $\\alpha_c = 0.9$ (median, and share of test nodes "
          "where it is above 2).", "tab:e7factor", "lcccc",
          ["Setting & Linear alone & $\\sigma$ & Median factor & Factor $> 2$ (\\%) \\\\"], body)

    zero = runs[runs.part == "no_smoothing"]
    macro("SplitsZero", str(zero.seed.nunique()))
    body = []
    for d in ["Texas", "Chameleon"]:
        for grid, gkey, label in [("paper grid", "Paper", "paper grid ($\\alpha_s \\geq 0.5$)"),
                                  ("with alpha 0", "Zero", "with $\\alpha_s = 0$")]:
            cells = []
            for m in ["MLP", "MLP + S", "MLP + C&S"]:
                g = zero[(zero.setting == d) & (zero.grid == grid) & (zero.method == m)]
                cells.append(pm(g.test_acc, f"Zero{d}{gkey}{key(m)}"))
            for m in ["MLP + S", "MLP + C&S"]:
                g = zero[(zero.setting == d) & (zero.grid == grid) & (zero.method == m)]
                cells.append(macro(f"Zero{d}{gkey}{key(m)}Off", f"{int((g.alpha_s == 0).sum())}/{len(g)}"))
            body.append(f"{d if gkey == 'Paper' else ''} & {label} & " + " & ".join(cells) + " \\\\")
        if d == "Texas":
            body.append("\\addlinespace")
    head = ["& & & & & \\multicolumn{2}{c}{Splits with $\\alpha_s = 0$} \\\\", "\\cmidrule(lr){6-7}",
            "Dataset & $\\alpha_s$ grid & MLP & MLP + S & MLP + C\\&S & + S & + C\\&S \\\\"]
    table("e7_zero", f"E6, the no-smoothing option: $\\alpha_s = 0$ added to the validation grid. Test accuracy "
          f"(\\%), mean $\\pm$ std over {MACROS['SplitsZero']} splits (the same splits as E3); the last two columns "
          "count the splits on which validation chose $\\alpha_s = 0$.", "tab:e7zero", "llccccc", head, body,
          size="\\footnotesize")


NOISE_KEY = {0.0: "Zero", 0.1: "Ten", 0.2: "Twenty", 0.3: "Thirty", 0.4: "Forty"}


def tab_e8():
    runs = pd.read_csv(RESULTS / "e8_runs.csv")
    splits(runs, "Noise")
    ps = sorted(runs.label_noise.unique())
    body = []
    for d in run_e8.DATASETS:                                   # rows: dataset x p (+ drop); columns: methods
        sel = lambda m, p: runs[(runs.dataset == d) & (runs.method == m) & (runs.label_noise == p)].test_acc
        for i, p in enumerate(ps):
            cells = [pm(sel(m, p), f"Noise{key(m)}{d}{NOISE_KEY[p]}") for m in run_e8.REPORT]
            body.append(f"{d if i == 0 else ''} & {p:g} & " + " & ".join(cells) + " \\\\")
        drops = [macro(f"Noise{key(m)}{d}Drop", f"{sel(m, ps[0]).mean() - sel(m, ps[-1]).mean():.2f}")
                 for m in run_e8.REPORT]
        body.append(f" & Drop & " + " & ".join(drops) + " \\\\")
        if d == run_e8.DATASETS[0]:
            body.append("\\addlinespace")
    head = ["Dataset & $p$ & " + " & ".join(tex(m) for m in run_e8.REPORT) + " \\\\"]
    table("e8", f"E7, wrong training labels: clean-test accuracy (\\%), mean $\\pm$ std over {MACROS['SplitsNoise']} "
          "splits; $p$ is the share of flipped training labels. Validation and test labels are never flipped. "
          f"Drop: accuracy at $p = 0$ minus accuracy at $p = {ps[-1]:g}$ (points).", "tab:e8",
          "lr" + "c" * len(run_e8.REPORT), head, body, size="\\footnotesize")


FACTOR_THRESHOLD = 2        # mirrors `factor > 2` in run_e7.part1_max_scale
FADED_BELOW = 20            # mirrors `r.n_nodes >= 20` in plots.e6_gain_by_group (fainter bars below this)


def numbers_settings():
    """Experiment settings quoted in the text, read from the experiment scripts."""
    import run_e6
    import run_q
    for s in run_e1.SHARES:
        macro(f"Share{SHARE_KEY[round(s, 2)]}", f"{100 * s:g}")
    macro("NoiseMin", f"{min(run_e8.NOISE):g}"), macro("NoiseMax", f"{max(run_e8.NOISE):g}")
    macro("CapSmallest", str(min(m for m in run_e7.MAX_SCALES if m is not None)))
    macro("FactorThreshold", str(FACTOR_THRESHOLD)), macro("FadedBelow", str(FADED_BELOW))
    macro("LocalLow", run_e6.HOMOPHILY_GROUPS[1].split("-")[1])
    macro("LocalHigh", run_e6.HOMOPHILY_GROUPS[-1].split("-")[0])
    akey = {0.0: "Zero", 0.1: "One", 0.3: "Three", 0.5: "Five", 0.7: "Seven", 0.8: "Eight", 0.9: "Nine", 0.99: "NineNine"}
    for a in run_e2.SWEEP:
        macro(f"SweepAlpha{akey[a]}", f"{a:g}")
    macro("QPerpCora", str(run_q.PERPLEXITY["Cora"])), macro("QPerpTexas", str(run_q.PERPLEXITY["Texas"]))
    macro("QSeed", str(run_q.SEED))


def numbers_misc():
    """Numbers quoted in the text that are not a table cell."""
    under = pd.read_csv(RESULTS / "pubmed_linear_underfit.csv")
    for m in ["Linear", "Linear + C", "Linear + S", "Linear + C&S"]:
        macro(f"Under{key(m)}", f2(under[under.method == m].test_acc.mean()))
    lin = under[under.method == "Linear"]
    macro("UnderTrainAcc", f2(lin.base_train_acc.mean())), macro("UnderSigma", f"{lin.sigma.mean():.3f}")
    rep = pd.read_csv(RESULTS / "reproduction_runs.csv")
    new = rep[(rep.dataset == "Pubmed") & (rep.method == "Linear")]
    macro("TunedTrainAcc", f2(new.base_train_acc.mean())), macro("TunedSigma", f"{new.sigma.mean():.3f}")
    e8 = pd.read_csv(RESULTS / "e8_runs.csv")
    for d in run_e8.DATASETS:
        g = e8[(e8.dataset == d) & (e8.method == "MLP")]
        macro(f"NoiseSigmaMLP{d}", f"{g[g.label_noise == 0].sigma.mean():.3f}")
        macro(f"NoiseTrainMLP{d}Forty", f"{g[g.label_noise == 0.4].base_train_acc.mean():.1f}")
    rep_cs = rep[rep.dataset == "Citeseer"]
    lp = rep_cs[rep_cs.method == "LP"]
    macro("IsoTestCiteseer", f"{lp.n_isolated_test.mean():.1f}")
    macro("LPConnCiteseer", f2(lp.test_acc_connected.mean()))
    q = pd.read_csv(RESULTS / "q_points.csv")
    stage_key = {"Z": "Z", "after Correct": "Correct", "after Smooth": "Smooth"}
    for d in q.dataset.unique():
        for stage, sk in stage_key.items():
            g = q[(q.dataset == d) & (q.stage == stage)]
            macro(f"Q{d}Acc{sk}", f"{100 * (g.pred == g.label).mean():.1f}")
        g = q[(q.dataset == d) & (q.stage == "Z")]
        macro(f"Q{d}N", str(len(g)))
        for outcome, ok in [("fixed", "Fixed"), ("broken", "Broken"), ("stayed right", "Right"),
                            ("stayed wrong", "Wrong")]:
            macro(f"Q{d}{ok}", str(int((g.outcome == outcome).sum())))


def write_numbers():
    lines = ["% Generated by code/make_tables.py from results/*.csv -- do not edit by hand.",
             "% Every number quoted in the text of main.tex is one of these macros (prefix n, e.g. \\nRepLinCSCora)."]
    lines += [f"\\newcommand{{\\n{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]   # prefix n: avoids clashes
    (TABLES / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report/tables/numbers.tex ({len(MACROS)} macros)")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    tab_hparams()
    tab_environment()
    tab_datasets()
    tab_reproduction()
    tab_ablation()
    tab_cost()
    tab_e1()
    tab_e2()
    tab_e5()
    tab_e6()
    tab_e7()
    tab_e8()
    numbers_settings()
    numbers_misc()
    write_numbers()


if __name__ == "__main__":
    main()
