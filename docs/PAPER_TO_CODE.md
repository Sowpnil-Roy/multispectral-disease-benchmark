# Where every part of the paper comes from

Stage names refer to `python src/stages.py <stage>`. Every number quoted in
the text is also checked automatically by `src/verify_paper.py`; the full list
with computed values is in [`CLAIMS_REPORT.md`](CLAIMS_REPORT.md).

## Methodology

| Paper | Code |
|---|---|
| Sec. III-A, Table I (archetypes) | `simulate.py`: `sample_healthy_leaf`, `apply_disease` |
| Sec. III-B, Table II (domains) | `simulate.py`: `sample_scene`, `add_sensor_noise`, `build_split` |
| Sec. III-C (SRF, band sets) | `simulate.py`: `gaussian_srf`; `stages.py`: `COMMERCIAL`, `CAND_BANDS`, `EXT_BANDS`, `band_features` |
| Sec. III-D (greedy selection) | `stages.py`: `_greedy`, stages `sel`, `sel2` |
| Sec. III-D (classifiers) | `stages.py`: `make_models`, `_models_seeded` |
| Sec. III-D (bootstrap, replicates) | stages `boot`, `reps` |

## Figures

| Figure | Script | Data |
|---|---|---|
| Fig. 1 framework | `figs.py` | – |
| Fig. 2 class-mean spectra (s ≥ 0.5) | `figs.py` | `data/spectra.npz` (train split) |
| Fig. 3 permutation importance | `figs.py` | `edge.json` (`pi_mean`, `pi_std`), `selection.json` |
| Fig. 4a band count vs. macro-F1 | `figs.py` | `bench_B.json`, `bench_C.json`, `bench_A.json`, `full59.json` (best domain-A model per configuration) |
| Fig. 4b early-stage F1 | `figs.py` | `bench_*.json` (`early_f1_A/B` of the best domain-A model) |
| Fig. 4c recall vs. severity | `figs.py` (refits the two MLPs) | stored in `results.json` → `sev_curve` |
| Fig. 5 confusion matrices | `figs.py` | `bench_C.json` → `conf` (GreedyX-5 + LR); copy in `results.json` |

## Tables

| Table | Script | Data |
|---|---|---|
| Table III | `tables.py` → `results/tables/table3_macro_f1.tex` | `bench_A/B/C.json`, `full59.json` |
| Table IV | `tables.py` → `results/tables/table4_replicates.tex` | `replicates.json` (mean ± sd, ddof = 0) |
| Table V | `tables.py` → `results/tables/table5_edge.tex` | `edge.json` → `edge` |

The generated rows are identical, character for character, to the rows in the
paper's LaTeX source.

## Numbers in the text that are not in a table or figure

| Paper | Value | File → key |
|---|---|---|
| IV-B | LR bootstrap CI [0.20, 0.25] | `bootstrap.json` → `Greedy-5.ci95` |
| IV-B | importance regions hold; greedy picks move 0–20 nm / 10–140 nm; 0.73–0.75 | `selection_stability.json`, `selection_stability_eval.json` |
| Abstract, I, IV-C | first SWIR pick 1450 nm (reported run); 1550 / 2000 nm under other realizations; 0.79–0.82 vs. 0.73–0.75; wilt recall 0.74–0.78 vs. 0.46–0.64 | `selection_ext.json`, `selection_stability.json`, `selection_stability_eval.json` |
| IV-C | LR bootstrap +0.31, CI [0.28, 0.33] | `bootstrap.json` → `GreedyX-5` |
| IV-C | wilt recall 0.62 / 0.76, confusions 0.22/0.24 → 0.13/0.18 | `perclass_logreg.json` → `Greedy-5`, `GreedyX-5` |
| IV-C | 59-band LR wilt recall 0.78 | `perclass_logreg.json` → `Full-59.wilt` (0.775) |
| IV-D | GreedyX-5 HGB 0.359 ± 0.018 on B | `replicates.json` → `GreedyX-5.HistGB.f1_test_B` |
| IV-D | 59-band MLP 0.543, RF 0.369, HGB 0.385, SVM 0.449 on B | `full59.json` |
| IV-E | early-stage F1 range, false-alarm rates | `bench_*.json` (`early_f1_A`), `early_far.json` |
