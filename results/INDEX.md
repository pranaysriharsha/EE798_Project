# Experiment index

All scripts live in `code/` and are run as `.venv\Scripts\python code\<script>.py` (from the project root,
or from `code/`). All figures are drawn from the CSVs by `code/plots.py` and are saved to `figures/` as
PDF and PNG; suggested captions are in `figures/captions.md`. Splits are random 60/20/20
(train/validation/test) unless stated otherwise. Accuracies are test accuracy in %, mean ± std over splits.

Experiment names follow the report (E1-E7). Script, CSV and figure file names keep the numbering used while the
experiments were run, so they do not always match: for example E2 (label scarcity) is `run_e1.py` with
`e1_*.csv`. The "Script(s)" and "CSV files" columns below give the right file for every experiment.

| Exp. | What | Script(s) | CSV files (`results/`) | Figures (`figures/`) | Splits | Headline numbers |
|---|---|---|---|---|---|---|
| R | Reproduction of the paper on Cora, Citeseer, Pubmed: Linear, MLP, LP, GCN, + C, + C&S | `run_main.py`; `tune_linear.py` (Linear grids, validation only) | `reproduction_runs.csv`, `reproduction_summary.csv`, `linear_grid_cora.csv`, `linear_grid_pubmed.csv`, `linear_grid_pubmed_fix.csv`, `pubmed_linear_underfit.csv` | – | 10 (seeds 0-9) | Linear + C&S 88.14 / 77.39 / 89.67 vs paper 88.62 / 76.31 / 89.99 (Cora / Citeseer / Pubmed); only gap > 3 points: our plain GCN on Citeseer, +3.06 above the paper |
| E1 | Step ablation: base, + C only, + S only, + C&S, for Linear and MLP | `run_main.py` (same runs as R) | `reproduction_runs.csv`, `reproduction_summary.csv` | – | 10 (seeds 0-9) | Smooth gives nearly all the gain (Cora: Linear 75.87 → + S 88.10 → + C&S 88.14); Correct changes nothing for the MLP (σ ≈ 0) |
| E2 | Label scarcity: training share 1-60%, validation 20% | `run_e1.py` | `e1_runs.csv`, `e1_summary.csv` | `e1_label_scarcity_cora`, `e1_label_scarcity_citeseer` | 3 (seeds 0-2; 5 would have exceeded 25 min) | Cora at 1% labels: GCN 58.7 vs MLP + C&S 41.1; C&S only draws level at 60% (87.6 vs 87.4); on Citeseer GCN leads at every share |
| E3 | Failure case: low-homophily graphs (Texas, Chameleon) and smoothing-strength sweep on five graphs | `run_e2.py` | `e2_runs.csv`, `e2_summary.csv`, `e2_smooth_sweep.csv`, `homophily.csv` | `e2_gain_vs_homophily`, `e2_accuracy_vs_alpha` | 10 (seeds 0-9) | Texas (h = 0.06): MLP 67.63 → MLP + S 61.58 (Smooth hurts); Chameleon (h = 0.23): 50.68 → 51.38 (no harm); GCN best on Chameleon at 66.29 |
| E4 | New application datasets: Amazon Photo and Computers (LastFM Asia unavailable: download host no longer exists) | `run_e5.py` (`--gcn-epochs 1000` for the fair-GCN re-run) | `e5_runs.csv`, `e5_summary.csv`, `e5_runs_backup_before_gcn1000.csv`, `homophily.csv` | `e5_gain_vs_homophily_all` | 3 (seeds 0-2; 5 would have exceeded 20 min) | With a fairly trained GCN (1000 epochs) GCN is best on both: Photo GCN 93.49 vs MLP + C&S 92.48; Computers GCN 89.54 vs LP 88.63 vs MLP + C&S 86.43. The 300-epoch GCN (kept as "GCN (300 epochs)") had under-fitted Computers: 78.94 |
| E5 | Per-node analysis by degree and by local homophily (Cora, Citeseer, Texas, Chameleon) | `run_e6.py` | `e6_nodes.csv` (per-node predictions), `e6_summary.csv` | `e6_gain_by_degree`, `e6_gain_by_local_homophily` | 5 (seeds 0-4), test nodes pooled | Nodes with local homophily < 0.25 lose 6-20 points under C&S on all four graphs (Cora: 4 fixed vs 44 broken); nodes with ≥ 0.5 gain 12-20 points |
| E6 | Safeguards: (1) cap on the Autoscale factor, (2) α_smooth = 0 ("no smoothing") added to the validation grid | `run_e7.py` | `e7_runs.csv`, `e7_summary.csv` | – | (1) 5 (seeds 0-4); (2) 10 (seeds 0-9) | (1) Cap of 1 lifts under-fitted Pubmed Linear + C&S from 84.12 to 87.19, costs ≤ 0.2 elsewhere; (2) validation picks "no smoothing" on 10/10 Texas splits: MLP + S 61.58 → 67.63 (= MLP, full loss recovered) |
| E7 | Wrong training labels: share p ∈ {0, 0.1, …, 0.4} of training labels flipped; validation/test clean | `run_e8.py` | `e8_runs.csv`, `e8_summary.csv` | `e8_label_noise_cora`, `e8_label_noise_citeseer` | 3 (seeds 0-2) | Drop from p = 0 to 0.4 (Cora / Citeseer): MLP −29.9 / −23.4 (largest), MLP + C&S −8.5 / −11.3, GCN −2.2 / −3.0 |
| Q | Qualitative: t-SNE of the C&S stages and fixed/broken maps, Cora and Texas | `run_q.py` | `q_points.csv` | `q_tsne_stages_cora`, `q_tsne_stages_texas`, `q_outcomes_cora`, `q_outcomes_texas` | 1 (seed 0) | Cora: Smooth lifts 72.4 → 85.8 (87 test nodes fixed, 14 broken); Texas: 63.2 → 57.9 (1 fixed, 3 broken) |

Other files: `step2_cora.csv`, `step3_cora.csv` (early development checks on Cora, 5 splits; superseded by R);
`run_*_log.txt` (console output of each run).
