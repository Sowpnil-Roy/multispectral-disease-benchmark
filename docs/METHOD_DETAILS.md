# Implementation details

Everything the paper states is implemented as stated. This page lists the
settings that a 6-page paper cannot hold, so that the benchmark can be rebuilt
from this page and `src/` alone.

## 1. Simulation (`src/simulate.py`)

| Item | Setting |
|---|---|
| Radiative-transfer model | `prosail.run_prosail(..., prospect_version="D")` (PROSPECT-D + 4SAIL), 400–2500 nm, 1 nm |
| Leaf angle distribution | ellipsoidal, `typelidf=2`, `lidfa` = mean leaf angle |
| Healthy priors, archetype factors | exactly Table I (`sample_healthy_leaf`, `apply_disease`) |
| Scene confounders | exactly Table II (`sample_scene`) |
| Solar zenith | Normal draw clipped to mean ± 10° (A: 20–40°, B: 38–58°) |
| Parameter guards passed to PROSAIL | Cab ≥ 0.5, Car ≥ 0.5, Cbrown ≤ 1.0, Cw ≥ 1e-4, Cm ≥ 1e-4, Cant ≥ 0, effective LAI ≥ 0.3 |
| Split sizes (per class) | train 1250, test A 400, test B 400, early A 300, early B 300 → 10,600 spectra |
| Severity | diseased: s ~ U(0.15, 1) (train/test), U(0.05, 0.30) (early); healthy: s = 0 |
| Random seeds | `build_split(..., seed=1..5)` for train, test_A, test_B, early_A, early_B |
| Output | `data/spectra.npz` (~80 MB, not committed); SHA-256 of every array in `data/checksums.json` |

## 2. Sensor synthesis (`simulate.py`, `stages.py`)

* Each band is a Gaussian spectral response with the given centre and FWHM
  (`sigma = FWHM / 2.3548`), normalised to unit sum over the 1 nm grid.
* Noise (`add_sensor_noise`): per-sample, per-band multiplicative gain
  N(1, sd), one illumination factor drawn once per call (i.e. once per data
  split), additive N(0, sd), then clipping to [0, 1]. Domain A: gain sd 1 %,
  additive sd 0.003, illumination U(0.98, 1.02). Domain B: 2 %, 0.006,
  U(0.94, 1.06).
* Every band configuration gets its own noise realization, seeded by
  `stable_seed(tag, split)` (CRC32 of the text, so it does not depend on
  Python's salted `hash()`). The tag is the configuration name, so the same
  configuration always sees the same noise in every stage.
* The vegetation-index baseline recomputes the RedEdge-MX bands with its own
  tag (`"vi"`), i.e. a separate noise realization of the same bands.

### Band sets

| Configuration | Centres (nm) | FWHM (nm) |
|---|---|---|
| RGB | 470, 550, 660 | 70, 80, 60 |
| Parrot Sequoia-like | 550, 660, 735, 790 | 40, 40, 10, 40 |
| MicaSense RedEdge-MX-like | 475, 560, 668, 717, 842 | 32, 27, 14, 12, 57 |
| Vegetation indices | NDVI, GNDVI, red-edge NDVI from the RedEdge-MX-like bands | – |
| Silicon library | 410–990, step 10 (59 bands) | 12 |
| SWIR extension | 1050–2350, step 50 (27 bands; 86 in total) | 25 |
| Greedy-k | first k of 590, 760, 980, 450, 850, 990 | 12 |
| GreedyX-k | first k of 1450, 620, 750, 970, 720, 880 | 25 above 1000 nm, else 12 |

## 3. Band selection (`stages.py: _greedy`)

* Features: the full candidate library with the `"cand"` (silicon) or `"ext"`
  (extended) noise realization of the training split.
* Subset: 1,500 training samples drawn with `numpy.random.default_rng(7)`.
* Score: mean 2-fold `cross_val_score` accuracy (scikit-learn default
  stratified, unshuffled folds) of `RandomForestClassifier(n_estimators=40,
  random_state=0)`.
* Forward selection to six bands; `greedy_cv_curve` stores the score after
  each addition.

## 4. Classifiers (`stages.py: make_models`)

| Name in paper | scikit-learn object |
|---|---|
| LR | `StandardScaler` + `LogisticRegression(max_iter=2000)` (lbfgs, multinomial, C = 1) |
| SVM | `StandardScaler` + `SVC(C=10, gamma="scale")` |
| RF | `RandomForestClassifier(n_estimators=150, random_state=0)` |
| HGB | `HistGradientBoostingClassifier(random_state=0)` (defaults) |
| MLP | `StandardScaler` + `MLPClassifier((64, 32), max_iter=350, random_state=0)` |

The MLP stops at 350 iterations and scikit-learn prints a
`ConvergenceWarning` for some configurations. This is expected; the reported
numbers are for this fixed setting.

"Best-A model" (Table III, Fig. 4a/b, Sec. IV-E) is the classifier with the
highest domain-A macro-F1 for that configuration.

## 5. Evaluation

| Quantity | Definition / stage |
|---|---|
| Macro-F1, accuracy | `f1_score(average="macro")`, `accuracy_score` on test A / test B |
| Early-stage F1 | binary F1 of (prediction ≠ healthy) vs. (label ≠ healthy) on the early sets |
| False-alarm rate | fraction of healthy early-stage samples predicted as any disease (`early`) |
| Per-class recall | row-normalized confusion matrix of LR on test A (`perclass`) |
| Paired bootstrap | 2,000 resamples of the test-A indices (`default_rng(0)`), LR predictions of both configurations on the same resample, 2.5/97.5 percentiles (`boot`) |
| Replicates (Table IV) | noise tags `rep{r}-{config}`, model seeds r = 0…4; mean ± **population** sd (`numpy.std`, ddof = 0) (`reps`) |
| Permutation importance | `RandomForestClassifier(120, random_state=0)` on all 59 silicon bands (`"cand"` noise), `permutation_importance(n_repeats=4, random_state=0)` on test A (`edge`) |
| Recall vs. severity (Fig. 4c) | MLP, test A + early A pooled, bins [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0]; bins with ≤ 20 diseased samples are skipped (`figs.py`) |
| Edge profile (Table V) | Greedy-5 with its own noise tag `"edgeprof"`; size = `len(pickle.dumps(model))`; latency = 3 timed passes over test A after a 100-sample warm-up, `time.perf_counter`, one CPU core (`edge`) |

## 6. Stability check (not needed for any table, used for one sentence)

`stages.py selstab` repeats the silicon and SWIR-extended greedy selection and
the silicon permutation importance under two further noise realizations
(`cand-alt1/2`, `ext-alt1/2`). Output: `results/selection_stability.json`.
`stages.py selstab_eval` then fits all five classifiers on the alternative
Greedy-5 / GreedyX-5 sets (their own noise tags) and stores macro-F1 on A and
B plus LR wilt recall in `results/selection_stability_eval.json`.
These feed the stability sentences in Secs. III-D, IV-B and IV-C; see
`docs/ANALYSIS.md`.

## 7. Extra fields that the paper does not use

`edge.json` also stores `full59_f1A` / `full59_f1B` (0.749 / 0.271): the
macro-F1 of the 120-tree RF that is fitted only to compute permutation
importance. The 59-band results in the paper come from `full59.json`
(150-tree RF and the other four classifiers).
