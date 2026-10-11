# Correct and Smooth: reproduction and extension experiments

Course project for EE798R (Pattern Recognition) by Pranaysriharsha.

This repository re-implements **Correct and Smooth (C&S)** from scratch in plain PyTorch and studies when it works.
C&S is the method of Huang, He, Singh, Lim and Benson, *Combining Label Propagation and Simple Models Out-performs
Graph Neural Networks*, ICLR 2021 ([arXiv:2010.13993](https://arxiv.org/abs/2010.13993)). It trains a model that
ignores the graph (a linear model or an MLP) and then uses the graph in two cheap label-propagation steps: *Correct*
spreads the errors made on the training nodes, and *Smooth* spreads the class scores. We reproduce the paper on
Cora, Citeseer and Pubmed and add seven experiments (E1-E7) that are not in the paper. PyTorch Geometric is used
only to download the datasets; the models, the GCN layer and all propagation code are written by hand.

## Repository layout

| Item | Contents |
|---|---|
| `code/` | all Python code: `data.py` (loading, graph matrices, splits), `models.py` (Linear, MLP, GCN), `train.py` (training loop), `propagate.py` (label propagation, Correct, Smooth), one `run_*.py` script per experiment, `plots.py` (figures), `make_tables.py` (LaTeX tables for the report) |
| `results/` | one CSV file per experiment, the console logs (`run_*_log.txt`) and `INDEX.md` (which script made which file, with headline numbers) |
| `figures/` | all figures as PDF and PNG, with suggested captions in `captions.md` |
| `requirements.txt` | pinned package versions |
| `README.md` | this file |

## Setup (CPU only)

No GPU is needed; everything was run on a laptop CPU. The datasets are downloaded automatically into a `data/`
folder next to `code/` the first time a script needs them.

**Windows (PowerShell)**
```
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
cd code
..\.venv\Scripts\python run_step3.py
```

**Linux / macOS**
```
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
cd code
../.venv/bin/python run_step3.py
```

## Quick check

`python run_step3.py` (run from `code/`) trains the base models and runs label propagation and C&S on Cora with 5
random splits. It takes about 1.5 minutes on a laptop CPU and writes `results/step3_cora.csv`. Expected test
accuracies (%, mean ± std over the 5 splits):

| Method | Accuracy |
|---|---|
| Label propagation (LP) | 84.27 ± 1.08 |
| Linear | 75.25 ± 1.84 |
| Linear + Correct | 77.57 ± 1.64 |
| Linear + C&S | 87.81 ± 1.97 |
| MLP | 72.27 ± 2.10 |
| MLP + C&S | 87.00 ± 1.62 |

## Experiments and their files

The report names the experiments R, E1-E7 and Q. The script, CSV and figure files keep the numbering used while the
experiments were run, so use this table to find them.

| Experiment | What it does | Script (`code/`) | Result files (`results/`) | Figures (`figures/`) |
|---|---|---|---|---|
| R | reproduction of the paper on Cora, Citeseer, Pubmed | `run_main.py`, `tune_linear.py` | `reproduction_*.csv`, `linear_grid_*.csv`, `pubmed_linear_underfit.csv` | -- |
| E1 | step ablation: base model, Correct only, Smooth only, both | `run_main.py` | `reproduction_*.csv` | -- |
| E2 | label scarcity (1% to 60% training labels) | `run_e1.py` | `e1_*.csv` | `e1_label_scarcity_*` |
| E3 | low-homophily graphs (Texas, Chameleon) and smoothing-strength sweep | `run_e2.py` | `e2_*.csv`, `homophily.csv` | `e2_gain_vs_homophily`, `e2_accuracy_vs_alpha` |
| E4 | Amazon Photo and Computers | `run_e5.py` | `e5_*.csv` | `e5_gain_vs_homophily_all` |
| E5 | per-node analysis (which nodes C&S fixes or breaks) | `run_e6.py` | `e6_*.csv` | `e6_gain_by_*` |
| E6 | two simple changes aimed at the two failures we found, and whether they help | `run_e7.py` | `e7_*.csv` | -- |
| E7 | wrong training labels | `run_e8.py` | `e8_*.csv` | `e8_label_noise_*` |
| Q | qualitative t-SNE views and fixed/broken maps | `run_q.py` | `q_points.csv` | `q_tsne_stages_*`, `q_outcomes_*` |

## Re-running everything

Run every command from `code/` with the virtual-environment Python. Each script prints a summary and writes its
CSV files to `results/`. Times were measured on a laptop CPU (Intel Core i7-1260P) and are indicative.

| Experiment | Command | Writes | Time |
|---|---|---|---|
| Data check | `python check_data.py` | prints dataset statistics | < 1 min |
| Linear tuning (validation only) | `python tune_linear.py`; `python tune_linear.py Pubmed --reduced`; `python tune_linear.py Pubmed --lrs 0.1 0.5 1.0 --wds 0 5e-5 --seeds 2 --out linear_grid_pubmed_fix.csv` | `linear_grid_*.csv` | ~8 + 3 + 4 min |
| R + E1 | `python run_main.py` | `reproduction_runs.csv`, `reproduction_summary.csv` | ~30 min |
| E2 | `python run_e1.py --seeds 3` | `e1_runs.csv`, `e1_summary.csv` | ~15 min |
| E3 | `python run_e2.py` | `e2_runs.csv`, `e2_summary.csv`, `e2_smooth_sweep.csv`, `homophily.csv` | ~18 min |
| E4 | `python run_e5.py --seeds 3`, then `python run_e5.py --seeds 3 --gcn-epochs 1000` | `e5_runs.csv`, `e5_summary.csv` | ~15 min + ~45 min |
| E5 | `python run_e6.py` | `e6_nodes.csv`, `e6_summary.csv` | ~3 min |
| E6 | `python run_e7.py` | `e7_runs.csv`, `e7_summary.csv` | ~5 min |
| E7 | `python run_e8.py` | `e8_runs.csv`, `e8_summary.csv` | ~13 min |
| Q | `python run_q.py` | `q_points.csv` | ~1 min |
| Figures | `python plots.py` | `figures/*.pdf`, `figures/*.png` | < 1 min |
| Report tables | `python make_tables.py` | `report/tables/*.tex` | < 1 min |

A full re-run of all experiments, including the Linear tuning grids, takes about 2.6 hours; without the tuning
grids about 2.4 hours.

## Notes

- **Repeatability.** All random number generators are seeded, so a re-run on the same machine gives exactly the
  same accuracies. On another machine or library version, expect small numerical differences. Times always vary
  with machine load.
- **LastFM Asia** is listed in `data.py` but can no longer be downloaded (its download host no longer exists);
  `run_e5.py` skips it.
- **`make_tables.py`** writes the LaTeX tables and the number macros used in the report into `report/tables/`.
  The report is not part of this repository; the script creates that folder if it is missing.
- **`run_main.py`** already contains the re-tuned Pubmed Linear setting. `results/pubmed_linear_underfit.csv` holds
  earlier Pubmed Linear runs with the setting chosen on Cora (raw features, learning rate 0.5, weight decay 5e-4);
  it was kept as an analysis result.
- **`run_step2.py` and `run_step3.py`** are the early development checks on Cora (`step2_cora.csv`,
  `step3_cora.csv`).
