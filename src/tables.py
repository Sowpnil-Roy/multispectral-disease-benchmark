"""Regenerate the LaTeX bodies of Tables III, IV and V from results/*.json.

Usage:  python tables.py
Writes results/tables/table3_macro_f1.tex, table4_replicates.tex,
table5_edge.tex (tabular rows only, ready to paste into the paper).

Conventions used in the paper:
  * Table III: single run, fixed seeds, macro-F1 to 3 decimals. Bold marks the
    best value per column within each block (commercial, Greedy, GreedyX).
    "Best-A model" is the classifier with the highest domain-A macro-F1 for
    that configuration; its domain-B score is reported.
  * Table IV: mean +- population standard deviation (numpy.std, ddof=0) over
    five noise realizations and model seeds.
  * Table V: model size is the pickled model size; latency is
    machine-dependent (the paper reports one single-core CPU run).
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESD = os.path.join(ROOT, "results")
OUTD = os.path.join(RESD, "tables")

MODELS = ["LogReg", "SVM-RBF", "RF", "HistGB", "MLP"]
SHORT = {"LogReg": "LR", "SVM-RBF": "SVM", "RF": "RF", "HistGB": "HGB",
         "MLP": "MLP", "LogReg-VI": "LR"}


def load(name):
    return json.load(open(os.path.join(RESD, name)))


def all_results():
    return {**load("bench_A.json"), **load("bench_B.json")["results"],
            **load("bench_C.json")["results"]}


def best_model(res_cfg):
    return max(res_cfg, key=lambda m: res_cfg[m]["f1_test_A"])


def table3():
    R = all_results()
    full = load("full59.json")
    blocks = [
        [("RGB", "RGB (3)", 3), ("Sequoia-like", "Sequoia (4)", 4),
         ("RedEdge-MX-like", "RedEdge-MX (5)", 5),
         ("Vegetation indices", "VI baseline", 3)],
        [(f"Greedy-{k}", f"Greedy-{k}", k) for k in (3, 4, 5, 6)],
        [(f"GreedyX-{k}", f"GreedyX-{k}", k) for k in (3, 4, 5, 6)],
    ]
    lines = []
    for block in blocks:
        rows = []
        for label, key, nb in block:
            r = R[key]
            if key == "VI baseline":
                v = r["LogReg-VI"]
                cells = [v["f1_test_A"], None, None, None, None, v["f1_test_B"]]
                bm, bB = "LogReg-VI", v["f1_test_B"]
            else:
                cells = [r[m]["f1_test_A"] for m in MODELS] + [r["LogReg"]["f1_test_B"]]
                bm = best_model(r)
                bB = r[bm]["f1_test_B"]
            rows.append((label, nb, cells, bB, bm))
        # column maxima within the block (for bold)
        col_max = []
        for c in range(6):
            vals = [round(x[2][c], 3) for x in rows if x[2][c] is not None]
            col_max.append(max(vals))
        bestB_max = max(round(x[3], 3) for x in rows)
        for label, nb, cells, bB, bm in rows:
            out = []
            for c, v in enumerate(cells):
                if v is None:
                    out.append("--")
                else:
                    s = f"{v:.3f}"
                    out.append(f"\\textbf{{{s}}}" if round(v, 3) == col_max[c] else s)
            sB = f"{bB:.3f}"
            sB = f"\\textbf{{{sB}}}" if round(bB, 3) == bestB_max else sB
            lines.append(f"{label} & {nb} & " + " & ".join(out) +
                         f" & {sB} ({SHORT[bm]})\\\\")
        lines.append("\\midrule")
    f = [full[m]["f1_test_A"] for m in MODELS] + [full["LogReg"]["f1_test_B"]]
    bm = best_model(full)
    lines.append("Full 59-band silicon library & 59 & " +
                 " & ".join(f"{v:.3f}" for v in f) +
                 f" & {full[bm]['f1_test_B']:.3f} ({SHORT[bm]})\\\\")
    return "\n".join(lines)


def fmt_ms(vals):
    m, s = float(np.mean(vals)), float(np.std(vals))  # ddof=0
    return f"{m:.3f}"[1:] + "$\\pm$" + f"{s:.3f}"[1:]


def table4():
    rep = load("replicates.json")
    names = [("RGB", "RGB (3)"), ("Sequoia", "Sequoia (4)"),
             ("RedEdge", "RedEdge-MX (5)"), ("Greedy-4", "Greedy-4"),
             ("Greedy-5", "Greedy-5"), ("GreedyX-5", "GreedyX-5")]
    lines = []
    for label, key in names:
        cells = []
        for m in ("LogReg", "MLP", "RF"):
            cells += [fmt_ms(rep[key][m]["f1_test_A"]),
                      fmt_ms(rep[key][m]["f1_test_B"])]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    return "\n".join(lines)


def table5():
    e = load("edge.json")["edge"]
    lines = []
    for m, lab in [("LogReg", "LR"), ("MLP", "MLP"), ("SVM-RBF", "SVM"),
                   ("HistGB", "HGB"), ("RF", "RF")]:
        size = f"{e[m]['size_kb']:,.1f}".replace(",", "{,}")
        lines.append(f"{lab:<5} & {size} & {e[m]['latency_us']:.2f} & "
                     f"{e[m]['acc']:.3f}\\\\")
    return "\n".join(lines)


if __name__ == "__main__":
    os.makedirs(OUTD, exist_ok=True)
    for name, fn in [("table3_macro_f1.tex", table3),
                     ("table4_replicates.tex", table4),
                     ("table5_edge.tex", table5)]:
        body = fn()
        open(os.path.join(OUTD, name), "w").write(body + "\n")
        print(f"--- {name}\n{body}\n")
