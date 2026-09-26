"""Check every number quoted in the paper against results/*.json.

Usage:  python verify_paper.py            # prints a report, writes docs/CLAIMS_REPORT.md
Exit code is 1 if any claim fails.

Rounding rule: a value printed with d decimals passes if it equals the
computed value rounded to d decimals (half-up), e.g. "0.39" for 0.385.
Latencies are machine-dependent; they are checked against the stored
results/edge.json of the paper run, not against a fresh timing.
"""
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESD = os.path.join(ROOT, "results")


def load(name):
    return json.load(open(os.path.join(RESD, name)))


A, B, C = load("bench_A.json"), load("bench_B.json"), load("bench_C.json")
R = {**A, **B["results"], **C["results"]}
FULL = load("full59.json")
REP = load("replicates.json")
BOOT = load("bootstrap.json")
EARLY = load("early_far.json")
EDGE = load("edge.json")
PERC = load("perclass_logreg.json")
SEL = load("selection.json")
SELX = load("selection_ext.json")
FINAL = load("results.json")
STAB = load("selection_stability.json")
MODELS = ["LogReg", "SVM-RBF", "RF", "HistGB", "MLP"]

ROWS = []


def rnd(x, d):
    q = Decimal(1).scaleb(-d)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def check(where, text, claimed, value, d=2, hard=True):
    """claimed: number as printed; value: computed; d: printed decimals.
    hard=False marks machine-dependent claims (latency): reported, never fatal."""
    ok = abs(rnd(value, d) - claimed) < 1e-9
    ROWS.append((where, text, claimed, value, ok if hard else (ok or None)))


def check_true(where, text, cond, detail):
    ROWS.append((where, text, "", detail, bool(cond)))


def best(cfg, metric="f1_test_A"):
    ms = {m: v for m, v in R[cfg].items()}
    m = max(ms, key=lambda k: ms[k][metric])
    return m, ms[m]


def rep_mean(cfg, model, split):
    return float(np.mean(REP[cfg][model][f"f1_test_{split}"]))


# ------------------------------------------------------------ data set size
n = FINAL["n"]
check_true("III-B", "10,600 observations (5000+1600+1600+1200+1200)",
           sum(n.values()) == 10600 and n["train"] == 5000 and n["early_A"] == 1200,
           str(n))

# ------------------------------------------------------------ band selection
check_true("Tab. III", "Greedy-k = first k of (590, 760, 980, 450, 850, 990)",
           SEL["selected_centers"] == [590, 760, 980, 450, 850, 990],
           str(SEL["selected_centers"]))
check_true("Tab. III", "GreedyX-k = first k of (1450, 620, 750, 970, 720, 880)",
           SELX["selected_centers"] == [1450, 620, 750, 970, 720, 880],
           str(SELX["selected_centers"]))
check_true("Abstract, IV-C", "1450 nm is picked first", SELX["selected_centers"][0] == 1450,
           str(SELX["selected_centers"][0]))

# ------------------------------------------------------------ IV-B gains (best model)
g3 = best("Greedy-3")[1]["f1_test_A"] - best("RGB (3)")[1]["f1_test_A"]
g4 = best("Greedy-4")[1]["f1_test_A"] - best("Sequoia (4)")[1]["f1_test_A"]
g5 = best("Greedy-5")[1]["f1_test_A"] - best("RedEdge-MX (5)")[1]["f1_test_A"]
check("Abstract/I/IV-B/VI", "+0.24 at three bands (best model)", 0.24, g3)
check("Abstract/I/IV-B/VI", "+0.16 at four bands (best model)", 0.16, g4)
check("Abstract/I/IV-B/VI", "+0.22 at five bands (best model)", 0.22, g5)
check("IV-B", "Greedy-3 0.68 (best model)", 0.68, best("Greedy-3")[1]["f1_test_A"])
check("IV-B", "RGB 0.44 (best model)", 0.44, best("RGB (3)")[1]["f1_test_A"])
check("IV-B", "Greedy-4 0.71 (best model)", 0.71, best("Greedy-4")[1]["f1_test_A"])
check("IV-B", "Sequoia-like 0.54 (best model)", 0.54, best("Sequoia (4)")[1]["f1_test_A"])
check("IV-B", "Greedy-5 0.73 (best model)", 0.73, best("Greedy-5")[1]["f1_test_A"])
check("IV-B", "RedEdge-MX-like 0.51 (best model)", 0.51, best("RedEdge-MX (5)")[1]["f1_test_A"])
check("IV-B", "LR alone: Greedy-5 0.732", 0.732, R["Greedy-5"]["LogReg"]["f1_test_A"], 3)
check("IV-B", "LR alone: RedEdge-MX-like 0.504", 0.504, R["RedEdge-MX (5)"]["LogReg"]["f1_test_A"], 3)
check("IV-B", "LR bootstrap CI lower 0.20", 0.20, BOOT["Greedy-5"]["ci95"][0])
check("IV-B", "LR bootstrap CI upper 0.25", 0.25, BOOT["Greedy-5"]["ci95"][1])
check("IV-B", "VI baseline (LR) 0.46", 0.46, R["VI baseline"]["LogReg-VI"]["f1_test_A"])
check("IV-B", "raw RedEdge bands (LR) 0.50", 0.50, R["RedEdge-MX (5)"]["LogReg"]["f1_test_A"])
sdA = max(float(np.std(REP[c][m]["f1_test_A"])) for c in REP for m in ("LogReg", "MLP", "RF"))
check_true("IV-B", "in-domain replicate sd <= 0.012 (Table IV)", rnd(sdA, 3) <= 0.012, f"{sdA:.4f}")
check("IV-B", "Sequoia vs RedEdge, LR 0.525", 0.525, R["Sequoia (4)"]["LogReg"]["f1_test_A"], 3)
seq_gt = all(np.mean(REP["Sequoia (4)"][m]["f1_test_A"]) > np.mean(REP["RedEdge-MX (5)"][m]["f1_test_A"])
             for m in ("LogReg", "MLP", "RF"))
check_true("IV-B", "Sequoia > RedEdge in domain A, consistent across replicates", seq_gt, "LR/MLP/RF replicate means")

pim = np.array(EDGE["pi_mean"]); cen = np.arange(410, 991, 10)
top = [int(cen[i]) for i in np.argsort(-pim)[:6]]
check_true("IV-B, Fig. 3", "importance concentrates on 740-770 nm and 960-990 nm",
           all(740 <= c <= 770 or 960 <= c <= 990 for c in top), f"top-6 centers {top}")
# selection stability (Sec. III-D, IV-B, IV-C): results/selection_stability*.json
STABE = load("selection_stability_eval.json")
alts = list(STAB["alternatives"].values())
base5 = SEL["selected_centers"][:5]           # 590, 760, 980, 450, 850
def _shift(c, alt):
    return min(abs(c - a) for a in alt["selected_centers"][:5])
re_w = [_shift(c, a) for a in alts for c in (760, 980)]
vis_nir = [_shift(c, a) for a in alts for c in (590, 450, 850)]
check_true("IV-B", "importance regions held under two other noise realizations",
           all(all(740 <= c <= 770 or 960 <= c <= 990 for c in a["pi_top5"]) for a in alts),
           str([a["pi_top5"] for a in alts]))
check("IV-B", "red-edge / water-shoulder picks moved 0 nm (min)", 0, min(re_w), 0)
check("IV-B", "red-edge / water-shoulder picks moved 20 nm (max)", 20, max(re_w), 0)
check("IV-B", "visible / NIR picks moved 10 nm (min)", 10, min(vis_nir), 0)
check("IV-B", "visible / NIR picks moved 140 nm (max)", 140, max(vis_nir), 0)


def _best_alt(tag, sname):
    r = STABE[tag][sname]["results"]
    return max(r[m]["f1_test_A"] for m in r)


si = [best("Greedy-5")[1]["f1_test_A"]] + [_best_alt(t, "Greedy-5") for t in STABE]
check("IV-B", "five selected silicon bands 0.73 (min over realizations)", 0.73, min(si))
check("IV-B", "five selected silicon bands 0.75 (max over realizations)", 0.75, max(si))
firsts = [SELX["selected_centers"][0]] + [a["selected_centers_ext"][0] for a in alts]
check_true("IV-C", "first extended pick under other realizations: 1550 or 2000 nm (SWIR > 1400 nm)",
           sorted(firsts[1:]) == [1550, 2000] and all(c > 1400 for c in firsts), str(firsts))
swx = [_best_alt(t, "GreedyX-5") for t in STABE]
sis = [_best_alt(t, "Greedy-5") for t in STABE]
check("IV-C", "alternative SWIR five-band sets 0.79 (min, best model)", 0.79, min(swx))
check("IV-C", "alternative SWIR five-band sets 0.82 (max, best model)", 0.82, max(swx))
check("IV-C", "alternative silicon five-band sets 0.73 (min, best model)", 0.73, min(sis))
check("IV-C", "alternative silicon five-band sets 0.75 (max, best model)", 0.75, max(sis))
wx = [STABE[t]["GreedyX-5"]["results"]["LogReg"]["wilt_recall_A"] for t in STABE]
ws = [STABE[t]["Greedy-5"]["results"]["LogReg"]["wilt_recall_A"] for t in STABE]
check("IV-C", "alternative SWIR sets wilt recall 0.74 (min, LR)", 0.74, min(wx))
check("IV-C", "alternative SWIR sets wilt recall 0.78 (max, LR)", 0.78, max(wx))
check("IV-C", "alternative silicon sets wilt recall 0.46 (min, LR)", 0.46, min(ws))
check("IV-C", "alternative silicon sets wilt recall 0.64 (max, LR)", 0.64, max(ws))
check("IV-B", "saturates near k=5 (best model ~0.73)", 0.73, best("Greedy-5")[1]["f1_test_A"])
check("IV-B", "four SWIR-extended bands 0.79", 0.79, best("GreedyX-4")[1]["f1_test_A"])
check("IV-B", "59-band LR 0.822", 0.822, FULL["LogReg"]["f1_test_A"], 3)
check("IV-B", "59-band RF 0.733", 0.733, FULL["RF"]["f1_test_A"], 3)

# ------------------------------------------------------------ IV-C SWIR
check("Abstract/IV-C", "GreedyX-5 0.81 (best model)", 0.81, best("GreedyX-5")[1]["f1_test_A"])
check("I/VI", "SWIR gain 0.30 (best model)", 0.30,
      best("GreedyX-5")[1]["f1_test_A"] - best("RedEdge-MX (5)")[1]["f1_test_A"])
check("IV-C", "LR bootstrap gain +0.31", 0.31, BOOT["GreedyX-5"]["mean"])
check("Abstract/IV-C", "LR bootstrap CI lower 0.28", 0.28, BOOT["GreedyX-5"]["ci95"][0])
check("Abstract/IV-C", "LR bootstrap CI upper 0.33", 0.33, BOOT["GreedyX-5"]["ci95"][1])
check("Abstract/IV-C/VI", "wilt recall Greedy-5 0.62", 0.62, PERC["Greedy-5"]["wilt"])
check("Abstract/IV-C/VI", "wilt recall GreedyX-5 0.76", 0.76, PERC["GreedyX-5"]["wilt"])
check("IV-C", "healthy->wilt 0.22 (Greedy-5)", 0.22, PERC["Greedy-5"]["healthy_as_wilt"])
check("IV-C", "wilt->healthy 0.24 (Greedy-5)", 0.24, PERC["Greedy-5"]["wilt_as_healthy"])
check("IV-C", "healthy->wilt 0.13 (GreedyX-5)", 0.13, PERC["GreedyX-5"]["healthy_as_wilt"])
check("IV-C", "wilt->healthy 0.18 (GreedyX-5)", 0.18, PERC["GreedyX-5"]["wilt_as_healthy"])
check("IV-C", "59-band LR wilt recall 0.78", 0.78, PERC["Full-59"]["wilt"])

# ------------------------------------------------------------ IV-D shift
check("IV-D", "GreedyX-5 LR 0.578 (replicate mean)", 0.578, rep_mean("GreedyX-5", "LogReg", "B"), 3)
check("IV-D", "GreedyX-5 LR sd 0.010", 0.010, np.std(REP["GreedyX-5"]["LogReg"]["f1_test_B"]), 3)
check("IV-D", "GreedyX-5 MLP 0.552", 0.552, rep_mean("GreedyX-5", "MLP", "B"), 3)
check("IV-D", "GreedyX-5 MLP sd 0.023", 0.023, np.std(REP["GreedyX-5"]["MLP"]["f1_test_B"]), 3)
check("IV-D", "GreedyX-5 RF 0.383", 0.383, rep_mean("GreedyX-5", "RF", "B"), 3)
check("IV-D", "GreedyX-5 RF sd 0.026", 0.026, np.std(REP["GreedyX-5"]["RF"]["f1_test_B"]), 3)
check("IV-D", "GreedyX-5 HGB 0.359 (not tabulated)", 0.359, rep_mean("GreedyX-5", "HistGB", "B"), 3)
check("IV-D", "GreedyX-5 HGB sd 0.018", 0.018, np.std(REP["GreedyX-5"]["HistGB"]["f1_test_B"]), 3)
check("IV-D", "59-band LR 0.599", 0.599, FULL["LogReg"]["f1_test_B"], 3)
check("IV-D", "59-band MLP 0.543", 0.543, FULL["MLP"]["f1_test_B"], 3)
check("IV-D", "59-band RF 0.369", 0.369, FULL["RF"]["f1_test_B"], 3)
check("IV-D", "59-band HGB 0.385", 0.385, FULL["HistGB"]["f1_test_B"], 3)
check("IV-D", "59-band SVM 0.449", 0.449, FULL["SVM-RBF"]["f1_test_B"], 3)
lin = [rep_mean("GreedyX-5", "LogReg", "B"), rep_mean("GreedyX-5", "MLP", "B"),
       R["GreedyX-5"]["LogReg"]["f1_test_B"], FULL["LogReg"]["f1_test_B"], FULL["MLP"]["f1_test_B"]]
check("Abstract/VI", "LR/MLP retain 0.54 (lower end)", 0.54, min(lin))
check("Abstract/VI", "LR/MLP retain 0.60 (upper end)", 0.60, max(lin))
trees = [rep_mean("GreedyX-5", "RF", "B"), rep_mean("GreedyX-5", "HistGB", "B"),
         FULL["RF"]["f1_test_B"], FULL["HistGB"]["f1_test_B"]]
check("Abstract/VI", "tree ensembles 0.36 (lower end)", 0.36, min(trees))
check("Abstract/VI", "tree ensembles 0.38 (upper end)", 0.38, max(trees))
for k in (4, 5, 6):
    cfg = f"GreedyX-{k}"
    check_true("IV-D", f"SVM wins domain A for {cfg}", best(cfg)[0] == "SVM-RBF", best(cfg)[0])
gaps = [R[f"GreedyX-{k}"]["LogReg"]["f1_test_B"] - R[f"GreedyX-{k}"]["SVM-RBF"]["f1_test_B"] for k in (4, 5, 6)]
check("IV-D", "SVM trails LR on B by 0.03 (min)", 0.03, min(gaps))
check("IV-D", "SVM trails LR on B by 0.05 (max)", 0.05, max(gaps))
check("IV-D", "GreedyX-5 LR 0.596", 0.596, R["GreedyX-5"]["LogReg"]["f1_test_B"], 3)
check("IV-D", "GreedyX-3 collapses to 0.25 (LR)", 0.25, R["GreedyX-3"]["LogReg"]["f1_test_B"])
comm = [R[c][m]["f1_test_B"] for c in ("RGB (3)", "Sequoia (4)", "RedEdge-MX (5)")
        for m in ["LogReg", best(c)[0]]]
check_true("IV-D", "all commercial sets fall to <= 0.32", rnd(max(comm), 2) <= 0.32, f"max {max(comm):.3f}")
cmB = np.array(FINAL["conf_B"], float); cmB /= cmB.sum(1, keepdims=True)
check("IV-D, Fig. 5b", "89% of true wilt retained", 89, cmB[3, 3] * 100, 0)
check("IV-D, Fig. 5b", "55% of healthy called wilt", 55, cmB[0, 3] * 100, 0)
check("IV-D, Fig. 5b", "necrosis recall 61%", 61, cmB[2, 2] * 100, 0)
cmA = np.array(FINAL["conf_A"], float); cmA /= cmA.sum(1, keepdims=True)
check("IV-C, Fig. 5a", "Fig. 5a wilt recall 76%", 76, cmA[3, 3] * 100, 0)

# ------------------------------------------------------------ IV-E early stage
mids_re, rec_re = FINAL["sev_curve"]["rededge"]
mids_gx, rec_gx = FINAL["sev_curve"]["greedyx5"]
hi = [r for m, r in zip(mids_re + mids_gx, rec_re + rec_gx) if m >= 0.65]
check("IV-E", "s>=0.65 recall lower end 0.92", 0.92, min(hi))
check("IV-E", "s>=0.65 recall upper end 1.00", 1.00, max(hi))
check("IV-E", "lowest bin recall ~0.5 (RedEdge-MX)", 0.5, rec_re[0], 1)
check("IV-E", "lowest bin recall ~0.5 (GreedyX-5)", 0.5, rec_gx[0], 1)
i6 = int(np.argmin(np.abs(np.array(mids_gx) - 0.6)))
check("IV-E", "0.92 (GreedyX-5) at s~0.6", 0.92, rec_gx[i6])
check("IV-E", "0.88 (RedEdge-MX) at s~0.6", 0.88, rec_re[i6])
show = ["RGB (3)", "Sequoia (4)", "RedEdge-MX (5)", "GreedyX-5"]
eA = [best(s)[1]["early_f1_A"] for s in show]
check("IV-E, Fig. 4b", "in-domain early F1 0.59 (min, Fig. 4b)", 0.59, min(eA))
check("IV-E, Fig. 4b", "in-domain early F1 0.62 (max, Fig. 4b)", 0.62, max(eA))
allA = [v["early_f1_A"] for cfg in R.values() for v in cfg.values()]
check("IV-E", "early F1 across all configurations 0.55 (min)", 0.55, min(allA))
check("IV-E", "early F1 across all configurations 0.72 (max)", 0.72, max(allA))
check("IV-E", "Sequoia-like + MLP early F1 on B 0.85", 0.85, EARLY["Sequoia (4)"]["f1_B"])
check("IV-E", "Sequoia-like + MLP early F1 on A 0.62", 0.62, EARLY["Sequoia (4)"]["f1_A"])
check("IV-E", "RedEdge-MX-like + SVM early F1 on B 0.84", 0.84, EARLY["RedEdge-MX (5)"]["f1_B"])
check("IV-E", "RedEdge-MX-like + SVM early F1 on A 0.59", 0.59, EARLY["RedEdge-MX (5)"]["f1_A"])
check("IV-E", "false alarms 100% (Sequoia-like, B)", 100, EARLY["Sequoia (4)"]["far_B"] * 100, 0)
check("IV-E", "false alarms 93% (RedEdge-MX-like, B)", 93, EARLY["RedEdge-MX (5)"]["far_B"] * 100, 0)
check("IV-E", "GreedyX-5 + SVM early F1 0.62 (A)", 0.62, EARLY["GreedyX-5"]["f1_A"])
check("IV-E", "GreedyX-5 + SVM early F1 0.62 (B)", 0.62, EARLY["GreedyX-5"]["f1_B"])
check("IV-E", "GreedyX-5 + SVM false alarms 0.18 (A)", 0.18, EARLY["GreedyX-5"]["far_A"])
check("IV-E", "GreedyX-5 + SVM false alarms 0.42 (B)", 0.42, EARLY["GreedyX-5"]["far_B"])

# ------------------------------------------------------------ IV-F edge (paper run)
e = EDGE["edge"]
check_true("IV-F, Tab. V", "LR and MLP are the two most accurate models",
           sorted(e, key=lambda m: -e[m]["acc"])[:2] == ["LogReg", "MLP"], "accuracy ranking")
check("IV-F", "RF about 50x slower than MLP (latency, machine-dependent)", 50, e["RF"]["latency_us"] / e["MLP"]["latency_us"], -1, hard=False)
check("IV-F", "RF 125x slower than LR (latency, machine-dependent)", 125, e["RF"]["latency_us"] / e["LogReg"]["latency_us"], 0, hard=False)
check("IV-F", "28 MB random forest", 28, e["RF"]["size_kb"] / 1024, 0)


def main():
    fails = [r for r in ROWS if r[4] is False]
    infos = [r for r in ROWS if r[4] is None]
    lines = ["# Paper-claim verification report", "",
             "Generated by `python src/verify_paper.py` from `results/*.json`.", "",
             f"**{len(ROWS) - len(fails) - len(infos)} / {len(ROWS)} claims pass; "
             f"{len(infos)} machine-dependent (info); {len(fails)} fail.**", "",
             "Latency ratios are checked against the stored paper run; a fresh run on",
             "another machine reports them as info rather than failing.", "",
             "| Section | Claim in paper | Printed | Computed | OK |",
             "|---|---|---|---|---|"]
    for where, text, claimed, value, ok in ROWS:
        v = f"{value:.4f}" if isinstance(value, float) else str(value)
        mark = "yes" if ok else ("info (machine-dependent)" if ok is None else "**NO**")
        lines.append(f"| {where} | {text} | {claimed} | {v} | {mark} |")
    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    open(os.path.join(ROOT, "docs", "CLAIMS_REPORT.md"), "w").write(report)
    for where, text, claimed, value, ok in ROWS:
        tag = "PASS " if ok else ("INFO " if ok is None else "FAIL ")
        print(tag + f"[{where}] {text}: printed {claimed}, computed {value}")
    print(f"\n{len(ROWS) - len(fails) - len(infos)}/{len(ROWS)} claims pass, "
          f"{len(infos)} machine-dependent (info), {len(fails)} fail")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
