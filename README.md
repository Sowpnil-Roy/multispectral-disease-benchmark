# Physics-Guided Band Configuration for Multispectral Crop Disease Detection

Code, results and figures for the paper

> **Physics-Guided Band Configuration and Lightweight Machine Learning for
> Multispectral Crop Disease Detection: A Radiative-Transfer Simulation
> Benchmark**
> <AUTHORS>, <CONFERENCE>, 2026.

The benchmark couples the PROSAIL radiative-transfer model (PROSPECT-D +
4SAIL) with three disease archetypes (chlorosis, necrosis, wilt), randomized
canopy and soil confounders, synthetic multispectral sensors, and five
lightweight classifiers. It compares commercial camera band sets with
data-driven band selection, tests robustness under an acquisition-domain
shift, and profiles edge-deployment cost. No field data are used.

Every number in the paper is produced by this repository, and
`src/verify_paper.py` checks all of them automatically
(**108 / 108 pass**, see [`docs/CLAIMS_REPORT.md`](docs/CLAIMS_REPORT.md)).

## Main results

| Finding | Value |
|---|---|
| Task-selected silicon bands vs. commercial sets at equal band count (best model) | +0.16 to +0.24 macro-F1 |
| Five bands with one SWIR band vs. RedEdge-MX-like (best model) | 0.81 vs. 0.51 |
| LR gain, paired bootstrap 95 % CI | +0.31 [0.28, 0.33] |
| First band picked from the SWIR-extended library | 1450 nm (1550 / 2000 nm under other noise realizations) |
| Wilt recall at five bands, silicon → SWIR-extended (LR) | 0.62 → 0.76 |
| Domain shift: LR / MLP vs. RF / HGB (5 and 59 bands) | 0.54–0.60 vs. 0.36–0.38 |
| Edge profile (5 bands): LR / MLP / RF model size | 1.4 kB / 51 kB / 28 MB |

## Quick start

Python 3.12 is recommended (tested on Linux x86-64).

```bash
git clone (https://github.com/Sowpnil-Roy/multispectral-disease-benchmark.git)
cd multispectral-disease-benchmark
pip install -r requirements.txt

# 1. Check every number in the paper against the stored results (seconds)
python src/verify_paper.py

# 2. Unit tests (no simulation needed)
python -m pytest -q

# 3. Full reproduction from scratch (about 12 minutes on one CPU core)
make reproduce
```

Without `make` (e.g. on Windows):

```bash
python src/simulate.py          # data/spectra.npz, ~20 s
python src/stages.py all        # all experiments -> results/*.json, ~10 min
python src/figs.py              # figures/*.pdf and results/results.json
python src/tables.py            # results/tables/*.tex (Tables III-V)
python src/verify_paper.py      # paper-claim check
```

`make reproduce` first moves the stored results to `results_ref/`, reruns
everything, and then runs `src/compare_results.py results_ref results`, which
reports any number that changed (timings are ignored). With the pinned
versions in `requirements.txt` the rerun is identical to the paper run.

## Pipeline

| Stage (`python src/stages.py <stage>`) | Output in `results/` | Used for |
|---|---|---|
| `sel` | `selection.json` | Greedy-k band sets, Fig. 3 |
| `sel2` | `selection_ext.json` | GreedyX-k band sets |
| `benchA` | `bench_A.json` | Table III (commercial sets, vegetation indices) |
| `benchB` | `bench_B.json` | Table III (Greedy-3 to 6) |
| `benchC` | `bench_C.json` | Table III (GreedyX-3 to 6), Fig. 5 |
| `edge` | `edge.json` | Table V, Fig. 3 |
| `full59` | `full59.json` | Table III (59 bands), Sec. IV-D |
| `perclass` | `perclass_logreg.json` | Sec. IV-C (per-class recall) |
| `boot` | `bootstrap.json` | Secs. IV-B and IV-C (paired bootstrap) |
| `early` | `early_far.json` | Sec. IV-E (early stage, false alarms) |
| `reps` | `replicates.json` | Table IV |
| `selstab` | `selection_stability.json` | Secs. IV-B and IV-C (stability) |
| `selstab_eval` | `selection_stability_eval.json` | Sec. IV-C (stability) |

Stages depend on `sel` and `sel2`; `all` runs them in order. `reps` resumes
from an existing `replicates.json`, so delete it to rerun the replicates.

A full mapping from every table, figure and in-text number to its file and
key is in [`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md).

## Repository layout

```
src/
  simulate.py        PROSAIL data generator (archetypes, domains, sensor noise)
  stages.py          band selection, benchmarks, bootstrap, replicates, edge profile
  figs.py            Figures 1-5 and results/results.json
  tables.py          LaTeX rows of Tables III-V
  verify_paper.py    checks every number quoted in the paper
  compare_results.py compares a rerun with the reference results
results/             JSON output of every stage (the paper run)
results/tables/      generated LaTeX table rows
figures/             the five paper figures (PDF)
data/checksums.json  SHA-256 of the simulated arrays (the 80 MB data file is
                     regenerated, not committed)
docs/                method details, paper-to-code map, audit, claims report
tests/               unit tests (run in GitHub Actions on every push)
```

## Documentation

* [`docs/METHOD_DETAILS.md`](docs/METHOD_DETAILS.md): every setting that the
  paper does not have room for (band FWHMs, seeds, hyperparameters, bins).
* [`docs/PAPER_TO_CODE.md`](docs/PAPER_TO_CODE.md): where each figure, table
  and in-text number comes from.
* [`docs/ANALYSIS.md`](docs/ANALYSIS.md): the code–paper audit, including the
  stability checks.
* [`docs/CLAIMS_REPORT.md`](docs/CLAIMS_REPORT.md): all checked claims with
  printed and computed values.

## Reproducibility notes

* All noise and model seeds are fixed. Noise seeds come from
  `stable_seed(tag, split)` (CRC32), so they do not depend on Python's salted
  `hash()`.
* Latencies (Table V) depend on the machine. The Makefile sets one BLAS/OpenMP
  thread to match the paper's single-core profile.
* Other library versions will run, but can change the last digits of some
  results; use `requirements.txt` for an exact match.
* The MLP prints `ConvergenceWarning` for some configurations; this is
  expected with the fixed `max_iter=350`.

## Using the benchmark for a new sensor

Add the band set to `COMMERCIAL` in `src/stages.py` as a list of
`(center_nm, fwhm_nm)` pairs, add its name to a `bench_group([...])` call (for
example in `stage_bench_A`), and rerun that stage. The new configuration is
evaluated with the same five classifiers, both domains and the early-stage
sets.

## Limitations

All results are simulation-derived and inherit PROSAIL's assumptions (a 1-D
turbid-medium canopy, no lesion geometry or mixed infections). They are
intended as testable predictions for field validation, not as field accuracy
estimates.

## Citation

See [`CITATION.cff`](CITATION.cff). Please replace the placeholders
(`<AUTHORS>`, `<CONFERENCE>`, `<LINK>`) after publication.

## License

MIT, see [`LICENSE`](LICENSE).
