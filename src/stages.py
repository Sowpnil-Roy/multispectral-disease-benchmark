"""Experiment stages for the PROSAIL multispectral disease benchmark.

Usage:  python stages.py <stage> [<stage> ...]
        python stages.py all          # every stage in dependency order

Stages (output file -> paper element):
  sel       greedy selection, silicon library   -> results/selection.json      (Greedy-k, Fig. 3)
  sel2      greedy selection, SWIR-extended     -> results/selection_ext.json  (GreedyX-k)
  benchA    commercial sets + VI baseline       -> results/bench_A.json        (Table III)
  benchB    Greedy-3..6                         -> results/bench_B.json        (Table III)
  benchC    GreedyX-3..6 (+ confusion matrices) -> results/bench_C.json        (Table III, Fig. 5)
  edge      edge profile + permutation import.  -> results/edge.json           (Table V, Fig. 3)
  full59    five classifiers on 59 bands        -> results/full59.json         (Table III, Sec. IV-D)
  perclass  per-class LR recall (5 and 59 bands)-> results/perclass_logreg.json (Sec. IV-C)
  boot      paired bootstrap of LR gains        -> results/bootstrap.json      (Sec. IV-B/C)
  early     early-stage F1 + false-alarm rate   -> results/early_far.json      (Sec. IV-E)
  reps      5 noise/model-seed replicates       -> results/replicates.json     (Table IV)
  selstab   selection / importance stability    -> results/selection_stability.json (Sec. IV-B/C)
  selstab_eval  classifiers on the alternative sets -> results/selection_stability_eval.json

Figures and results/results.json are produced afterwards by figs.py;
LaTeX tables by tables.py; every number quoted in the paper is checked by
verify_paper.py.
"""
import json, os, pickle, sys, time, zlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance

from simulate import make_srf_matrix, add_sensor_noise, CLASSES, WL, ROOT


def stable_seed(*parts):
    """Deterministic seed (Python's hash() is salted per process)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2**31)

DATA_PATH = os.path.join(ROOT, "data", "spectra.npz")
if not os.path.exists(DATA_PATH):
    sys.exit("data/spectra.npz not found. Run `python src/simulate.py` first "
             "(about 20 s on one CPU core).")
DATA = np.load(DATA_PATH)
FIGD = os.path.join(ROOT, "figures"); os.makedirs(FIGD, exist_ok=True)
RESD = os.path.join(ROOT, "results"); os.makedirs(RESD, exist_ok=True)

COMMERCIAL = {
    "RGB (3)":        [(470, 70), (550, 80), (660, 60)],
    "Sequoia (4)":    [(550, 40), (660, 40), (735, 10), (790, 40)],
    "RedEdge-MX (5)": [(475, 32), (560, 27), (668, 14), (717, 12), (842, 57)],
}
CAND_CENTERS = np.arange(410, 991, 10)
CAND_BANDS = [(int(c), 12) for c in CAND_CENTERS]
SPLITS = ["train", "test_A", "test_B", "early_A", "early_B"]
DOMAIN_OF = {"train": "A", "test_A": "A", "test_B": "B",
             "early_A": "A", "early_B": "B"}
HEALTHY = CLASSES.index("healthy")


def band_features(bands, split, seed):
    S = make_srf_matrix(bands)
    R = DATA[f"X_{split}"] @ S.T
    rng = np.random.default_rng(seed)
    return add_sensor_noise(R, DOMAIN_OF[split], rng).astype(np.float32)


def featurize_all(bands, tag):
    return {sp: band_features(bands, sp, seed=stable_seed(tag, sp))
            for sp in SPLITS}


def make_models():
    return {
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000)),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
        "RF": RandomForestClassifier(n_estimators=150, random_state=0),
        "HistGB": HistGradientBoostingClassifier(random_state=0),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier((64, 32), max_iter=350,
                                           random_state=0)),
    }


def early_f1(model, X, y):
    yb = (y != HEALTHY).astype(int)
    pb = (model.predict(X) != HEALTHY).astype(int)
    return f1_score(yb, pb)


def get_sensors():
    sensors = dict(COMMERCIAL)
    sel = json.load(open(f"{RESD}/selection.json"))["selected_centers"]
    for k in (3, 4, 5, 6):
        sensors[f"Greedy-{k}"] = [(c, 12) for c in sel[:k]]
    return sensors


def _greedy(F_tr, centers, n_select=6, tag=""):
    """Greedy forward selection: at each step add the band that maximizes the
    2-fold CV accuracy of a 40-tree RF on a fixed 1,500-sample subset."""
    t0 = time.time()
    ytr = DATA["y_train"]
    rng = np.random.default_rng(7)
    idx = rng.choice(len(ytr), 1500, replace=False)
    Xs, ys = F_tr[idx], ytr[idx]
    chosen, curve = [], []
    remaining = list(range(F_tr.shape[1]))
    for k in range(n_select):
        best_j, best_sc = None, -1
        for j in remaining:
            cols = chosen + [j]
            clf = RandomForestClassifier(n_estimators=40, random_state=0)
            sc = cross_val_score(clf, Xs[:, cols], ys, cv=2,
                                 scoring="accuracy").mean()
            if sc > best_sc:
                best_sc, best_j = sc, j
        chosen.append(best_j); remaining.remove(best_j)
        curve.append(round(float(best_sc), 4))
        print(f"{tag} k={k+1}: +{centers[best_j]} nm cv={best_sc:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return [int(centers[j]) for j in chosen], curve


def stage_sel():
    F_tr = band_features(CAND_BANDS, "train", seed=stable_seed("cand", "train"))
    sel, curve = _greedy(F_tr, CAND_CENTERS, tag="silicon")
    json.dump({"selected_centers": sel, "greedy_cv_curve": curve},
              open(f"{RESD}/selection.json", "w"))
    print("selection saved")


def bench_group(names):
    sensors = get_sensors()
    results, conf = {}, {}
    for sname in names:
        t0 = time.time()
        F = featurize_all(sensors[sname], sname)
        results[sname] = {}
        for mname, model in make_models().items():
            model.fit(F["train"], DATA["y_train"])
            m = {}
            for sp in ("test_A", "test_B"):
                yp = model.predict(F[sp])
                m[f"acc_{sp}"] = float(accuracy_score(DATA[f"y_{sp}"], yp))
                m[f"f1_{sp}"] = float(f1_score(DATA[f"y_{sp}"], yp,
                                               average="macro"))
            m["early_f1_A"] = float(early_f1(model, F["early_A"],
                                             DATA["y_early_A"]))
            m["early_f1_B"] = float(early_f1(model, F["early_B"],
                                             DATA["y_early_B"]))
            results[sname][mname] = m
            if sname == "Greedy-5" and mname == "LogReg":
                conf["A"] = confusion_matrix(
                    DATA["y_test_A"], model.predict(F["test_A"])).tolist()
                conf["B"] = confusion_matrix(
                    DATA["y_test_B"], model.predict(F["test_B"])).tolist()
        print(f"{sname}: done in {time.time()-t0:.0f}s", flush=True)
    return results, conf


def stage_bench_A():
    results, _ = bench_group(["RGB (3)", "Sequoia (4)", "RedEdge-MX (5)"])
    # VI baseline
    def vi(split):
        R = band_features(COMMERCIAL["RedEdge-MX (5)"], split,
                          seed=stable_seed("vi", split))
        B, G, Rd, RE, NIR = [R[:, i] for i in range(5)]
        e = 1e-6
        return np.stack([(NIR-Rd)/(NIR+Rd+e), (NIR-G)/(NIR+G+e),
                         (NIR-RE)/(NIR+RE+e)], axis=1)
    V = {sp: vi(sp) for sp in SPLITS}
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(V["train"], DATA["y_train"])
    results["VI baseline"] = {"LogReg-VI": {
        "acc_test_A": float(accuracy_score(DATA["y_test_A"], m.predict(V["test_A"]))),
        "f1_test_A": float(f1_score(DATA["y_test_A"], m.predict(V["test_A"]), average="macro")),
        "acc_test_B": float(accuracy_score(DATA["y_test_B"], m.predict(V["test_B"]))),
        "f1_test_B": float(f1_score(DATA["y_test_B"], m.predict(V["test_B"]), average="macro")),
        "early_f1_A": float(early_f1(m, V["early_A"], DATA["y_early_A"])),
        "early_f1_B": float(early_f1(m, V["early_B"], DATA["y_early_B"])),
    }}
    json.dump(results, open(f"{RESD}/bench_A.json", "w"))
    print("bench_A saved")


def stage_bench_B():
    results, conf = bench_group(["Greedy-3", "Greedy-4", "Greedy-5",
                                 "Greedy-6"])
    json.dump({"results": results, "conf": conf},
              open(f"{RESD}/bench_B.json", "w"))
    print("bench_B saved")


def stage_edge():
    t0 = time.time()
    sensors = get_sensors()
    F5 = featurize_all(sensors["Greedy-5"], "edgeprof")
    edge = {}
    for mname, model in make_models().items():
        model.fit(F5["train"], DATA["y_train"])
        size_kb = len(pickle.dumps(model)) / 1024
        X = F5["test_A"]; model.predict(X[:100])
        t = time.perf_counter()
        for _ in range(3):
            model.predict(X)
        us = (time.perf_counter() - t) / (3 * len(X)) * 1e6
        acc = float(accuracy_score(DATA["y_test_A"], model.predict(X)))
        edge[mname] = {"size_kb": round(size_kb, 1),
                       "latency_us": round(us, 2), "acc": round(acc, 4)}
        print(mname, edge[mname], flush=True)
    # permutation importance over 59-band library
    F_tr = band_features(CAND_BANDS, "train", seed=stable_seed("cand", "train"))
    F_teA = band_features(CAND_BANDS, "test_A", seed=stable_seed("cand", "test_A"))
    F_teB = band_features(CAND_BANDS, "test_B", seed=stable_seed("cand", "test_B"))
    rf = RandomForestClassifier(n_estimators=120, random_state=0)
    rf.fit(F_tr, DATA["y_train"])
    pi = permutation_importance(rf, F_teA, DATA["y_test_A"], n_repeats=4,
                                random_state=0)
    out = {"edge": edge,
           "pi_mean": pi.importances_mean.tolist(),
           "pi_std": pi.importances_std.tolist(),
           "full59_f1A": round(float(f1_score(DATA["y_test_A"], rf.predict(F_teA), average="macro")), 4),
           "full59_f1B": round(float(f1_score(DATA["y_test_B"], rf.predict(F_teB), average="macro")), 4)}
    json.dump(out, open(f"{RESD}/edge.json", "w"))
    print("edge saved", round(time.time()-t0), "s")





# ---------------------------------------------------------- SWIR extension
SWIR_CENTERS = list(range(1050, 2351, 50))
EXT_BANDS = CAND_BANDS + [(c, 25) for c in SWIR_CENTERS]
EXT_CENTERS = np.array([c for c, _ in EXT_BANDS])


def stage_sel2():
    F_tr = band_features(EXT_BANDS, "train", seed=stable_seed("ext", "train"))
    sel, curve = _greedy(F_tr, EXT_CENTERS, tag="extended")
    json.dump({"selected_centers": sel, "greedy_cv_curve": curve},
              open(f"{RESD}/selection_ext.json", "w"))
    print("selection_ext saved")


def ext_sensors():
    sel = json.load(open(f"{RESD}/selection_ext.json"))["selected_centers"]
    out = {}
    for k in (3, 4, 5, 6):
        out[f"GreedyX-{k}"] = [(c, 25 if c > 1000 else 12)
                               for c in sel[:k]]
    return out


def stage_bench_C():
    sensors = ext_sensors()
    results, conf = {}, {}
    for sname, bands in sensors.items():
        t0 = time.time()
        F = featurize_all(bands, sname)
        results[sname] = {}
        for mname, model in make_models().items():
            model.fit(F["train"], DATA["y_train"])
            m = {}
            for sp in ("test_A", "test_B"):
                yp = model.predict(F[sp])
                m[f"acc_{sp}"] = float(accuracy_score(DATA[f"y_{sp}"], yp))
                m[f"f1_{sp}"] = float(f1_score(DATA[f"y_{sp}"], yp,
                                               average="macro"))
            m["early_f1_A"] = float(early_f1(model, F["early_A"],
                                             DATA["y_early_A"]))
            m["early_f1_B"] = float(early_f1(model, F["early_B"],
                                             DATA["y_early_B"]))
            results[sname][mname] = m
            if sname == "GreedyX-5" and mname == "LogReg":
                conf["A"] = confusion_matrix(
                    DATA["y_test_A"], model.predict(F["test_A"])).tolist()
                conf["B"] = confusion_matrix(
                    DATA["y_test_B"], model.predict(F["test_B"])).tolist()
        best = max(results[sname], key=lambda k: results[sname][k]["f1_test_A"])
        print(f"{sname}: best={best} "
              f"f1A={results[sname][best]['f1_test_A']:.3f} "
              f"f1B={results[sname][best]['f1_test_B']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    json.dump({"results": results, "conf": conf},
              open(f"{RESD}/bench_C.json", "w"))
    print("bench_C saved")


# ------------------------------------------------ added in revision (v2)
def _fit_eval(model, F):
    model.fit(F["train"], DATA["y_train"])
    out = {}
    for sp in ("test_A", "test_B"):
        yp = model.predict(F[sp])
        out[f"f1_{sp}"] = float(f1_score(DATA[f"y_{sp}"], yp, average="macro"))
        out[f"acc_{sp}"] = float(accuracy_score(DATA[f"y_{sp}"], yp))
    return out, model


def stage_full59():
    """All five classifiers on the full 59-band silicon library."""
    F = featurize_all(CAND_BANDS, "full59")
    res = {}
    for mname, model in make_models().items():
        t0 = time.time()
        res[mname], _ = _fit_eval(model, F)
        print(mname, {k: round(v, 3) for k, v in res[mname].items()},
              f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(f"{RESD}/full59.json", "w"), indent=1)


def _models_seeded(seed):
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
        "RF": RandomForestClassifier(n_estimators=150, random_state=seed),
        "HistGB": HistGradientBoostingClassifier(random_state=seed),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(
            (64, 32), max_iter=350, random_state=seed)),
    }


def stage_reps():
    """Replicate stability: 5 independent sensor-noise realizations and
    model seeds per configuration -> mean and sd of macro-F1."""
    sensors = {**get_sensors(), **ext_sensors()}
    names = ["RGB (3)", "Sequoia (4)", "RedEdge-MX (5)", "Greedy-4",
             "Greedy-5", "GreedyX-5"]
    path = f"{RESD}/replicates.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for sname in names:
        if sname in res:
            continue
        t0 = time.time(); res[sname] = {}
        for r in range(5):
            F = featurize_all(sensors[sname], f"rep{r}-{sname}")
            for mname, model in _models_seeded(r).items():
                m, _ = _fit_eval(model, F)
                d = res[sname].setdefault(mname, {"f1_test_A": [], "f1_test_B": []})
                d["f1_test_A"].append(m["f1_test_A"]); d["f1_test_B"].append(m["f1_test_B"])
        json.dump(res, open(path, "w"), indent=1)
        print(sname, f"{time.time()-t0:.0f}s", flush=True)


def stage_boot():
    """Paired bootstrap (2000 resamples of the domain-A test set) of the
    macro-F1 difference, LR on Greedy-5 / GreedyX-5 vs. RedEdge-MX."""
    sensors = {**get_sensors(), **ext_sensors()}
    preds = {}
    for sname in ["RedEdge-MX (5)", "Greedy-5", "GreedyX-5"]:
        F = featurize_all(sensors[sname], sname)
        m = make_models()["LogReg"].fit(F["train"], DATA["y_train"])
        preds[sname] = m.predict(F["test_A"])
    y = DATA["y_test_A"]; rng = np.random.default_rng(0); n = len(y)
    out = {}
    for a in ["Greedy-5", "GreedyX-5"]:
        diffs = []
        for _ in range(2000):
            i = rng.integers(0, n, n)
            diffs.append(f1_score(y[i], preds[a][i], average="macro")
                         - f1_score(y[i], preds["RedEdge-MX (5)"][i], average="macro"))
        diffs = np.array(diffs)
        out[a] = {"mean": float(diffs.mean()),
                  "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    json.dump(out, open(f"{RESD}/bootstrap.json", "w"), indent=1)
    print(out)


def stage_early():
    """Early-stage F1, recall and healthy false-alarm rate (best in-domain
    model per configuration) for the configurations shown in Fig. 4b."""
    sensors = {**COMMERCIAL, **get_sensors(), **ext_sensors()}
    allres = {**json.load(open(f"{RESD}/bench_A.json")),
              **json.load(open(f"{RESD}/bench_B.json"))["results"],
              **json.load(open(f"{RESD}/bench_C.json"))["results"]}
    out = {}
    for sname in ["RGB (3)", "Sequoia (4)", "RedEdge-MX (5)", "GreedyX-5"]:
        best = max(allres[sname], key=lambda k: allres[sname][k]["f1_test_A"])
        F = featurize_all(sensors[sname], sname)
        model = make_models()[best].fit(F["train"], DATA["y_train"])
        d = {"model": best}
        for dom in ("A", "B"):
            y = DATA[f"y_early_{dom}"]; p = model.predict(F[f"early_{dom}"])
            yb, pb = y != HEALTHY, p != HEALTHY
            d[f"f1_{dom}"] = float(f1_score(yb, pb))
            d[f"recall_{dom}"] = float(pb[yb].mean())
            d[f"far_{dom}"] = float(pb[~yb].mean())
        out[sname] = d; print(sname, d, flush=True)
    json.dump(out, open(f"{RESD}/early_far.json", "w"), indent=1)


def _perclass(model, F, split):
    cm = confusion_matrix(DATA[f"y_{split}"], model.predict(F[split])).astype(float)
    cm /= cm.sum(1, keepdims=True)
    d = {c: round(float(cm[i, i]), 3) for i, c in enumerate(CLASSES)}
    d["healthy_as_wilt"] = round(float(cm[0, 3]), 3)
    d["wilt_as_healthy"] = round(float(cm[3, 0]), 3)
    return d


def stage_perclass():
    """Row-normalized per-class LR recall on the domain-A test set for the
    two five-band sets (Sec. IV-C) and for the dense 59-band library (the
    0.78 wilt recall quoted in Sec. IV-C). Feature tags are identical to the
    benchmark stages, so these are the same fitted models as in Table III."""
    sensors = {**get_sensors(), **ext_sensors()}
    out = {}
    for sname in ["Greedy-5", "GreedyX-5"]:
        F = featurize_all(sensors[sname], sname)
        m = make_models()["LogReg"].fit(F["train"], DATA["y_train"])
        out[sname] = _perclass(m, F, "test_A")
    F = featurize_all(CAND_BANDS, "full59")
    m = make_models()["LogReg"].fit(F["train"], DATA["y_train"])
    out["Full-59"] = _perclass(m, F, "test_A")
    json.dump(out, open(f"{RESD}/perclass_logreg.json", "w"), indent=1)
    print(out)


def stage_selstab():
    """Stability check (Sec. IV-B/IV-C): greedy selection on the silicon and
    SWIR-extended libraries, and RF permutation importance on the silicon
    library, repeated under two alternative sensor-noise realizations of the
    same simulated spectra. The paper's band sets use the baseline realization
    (results/selection*.json); this stage shows how much they move."""
    base = json.load(open(f"{RESD}/selection.json"))["selected_centers"]
    base_x = json.load(open(f"{RESD}/selection_ext.json"))["selected_centers"]
    base_pi = np.array(json.load(open(f"{RESD}/edge.json"))["pi_mean"])
    out = {"baseline": {"selected_centers": base, "selected_centers_ext": base_x,
                        "pi_top5": [int(CAND_CENTERS[i]) for i in np.argsort(-base_pi)[:5]]},
           "alternatives": {}}
    for r in (1, 2):
        tag = f"cand-alt{r}"
        F_tr = band_features(CAND_BANDS, "train", seed=stable_seed(tag, "train"))
        F_teA = band_features(CAND_BANDS, "test_A", seed=stable_seed(tag, "test_A"))
        sel, curve = _greedy(F_tr, CAND_CENTERS, tag=tag)
        rf = RandomForestClassifier(n_estimators=120, random_state=0)
        rf.fit(F_tr, DATA["y_train"])
        pi = permutation_importance(rf, F_teA, DATA["y_test_A"], n_repeats=4,
                                    random_state=0)
        Fx = band_features(EXT_BANDS, "train", seed=stable_seed(f"ext-alt{r}", "train"))
        selx, curvex = _greedy(Fx, EXT_CENTERS, tag=f"ext-alt{r}")
        shift = [int(min(abs(c - a) for a in sel[:5])) for c in base[:5]]
        out["alternatives"][tag] = {
            "selected_centers": sel, "greedy_cv_curve": curve,
            "abs_shift_vs_baseline_first5_nm": shift,
            "pi_top5": [int(CAND_CENTERS[i]) for i in np.argsort(-pi.importances_mean)[:5]],
            "pi_mean": pi.importances_mean.tolist(),
            "selected_centers_ext": selx, "greedy_cv_curve_ext": curvex}
        print(tag, sel, "shift", shift, "ext", selx, flush=True)
    json.dump(out, open(f"{RESD}/selection_stability.json", "w"), indent=1)


def stage_selstab_eval():
    """Do the conclusions hold for the alternative band sets of `selstab`?
    Fits LR (and all five classifiers) on each alternative Greedy-5 /
    GreedyX-5 set and reports macro-F1 on A and B and LR wilt recall."""
    stab = json.load(open(f"{RESD}/selection_stability.json"))
    out = {}
    for tag, alt in stab["alternatives"].items():
        sets = {"Greedy-5": [(c, 12) for c in alt["selected_centers"][:5]],
                "GreedyX-5": [(c, 25 if c > 1000 else 12)
                              for c in alt["selected_centers_ext"][:5]]}
        out[tag] = {}
        for sname, bands in sets.items():
            F = featurize_all(bands, f"{tag}-{sname}")
            res = {}
            for mname, model in make_models().items():
                res[mname], fitted = _fit_eval(model, F)
                if mname == "LogReg":
                    res[mname]["wilt_recall_A"] = _perclass(fitted, F, "test_A")["wilt"]
            out[tag][sname] = {"bands": [c for c, _ in bands], "results": res}
            print(tag, sname, [c for c, _ in bands],
                  {m: round(v["f1_test_A"], 3) for m, v in res.items()}, flush=True)
    json.dump(out, open(f"{RESD}/selection_stability_eval.json", "w"), indent=1)


STAGES = {"sel": stage_sel, "sel2": stage_sel2, "benchA": stage_bench_A,
          "benchB": stage_bench_B, "benchC": stage_bench_C, "edge": stage_edge,
          "full59": stage_full59, "perclass": stage_perclass,
          "boot": stage_boot, "early": stage_early, "reps": stage_reps,
          "selstab": stage_selstab, "selstab_eval": stage_selstab_eval}


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    order = list(STAGES) if args == ["all"] else args
    for st in order:
        if st not in STAGES:
            sys.exit(f"unknown stage '{st}'. choose from: all, {', '.join(STAGES)}")
        print(f"==== stage {st}", flush=True)
        STAGES[st]()
