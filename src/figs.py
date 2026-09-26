"""Generate all paper figures from cached stage results + quick refits."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from stages import (band_features, featurize_all, make_models, CAND_CENTERS,
                    COMMERCIAL, ext_sensors, DATA, CLASSES, WL, RESD, FIGD,
                    HEALTHY)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9,
    "axes.labelsize": 8.5, "legend.fontsize": 7.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 300, "savefig.bbox": "tight",
})
CLS_COLORS = {"healthy": "#2a9d3f", "chlorosis": "#e0a10a",
              "necrosis": "#a1401f", "wilt": "#2a6ea6"}

bA = json.load(open(f"{RESD}/bench_A.json"))
bB = json.load(open(f"{RESD}/bench_B.json"))
bC = json.load(open(f"{RESD}/bench_C.json"))
edge = json.load(open(f"{RESD}/edge.json"))
selV = json.load(open(f"{RESD}/selection.json"))
selX = json.load(open(f"{RESD}/selection_ext.json"))
RES = {**bA, **bB["results"], **bC["results"]}
MODELS = ["LogReg", "SVM-RBF", "RF", "HistGB", "MLP"]


def best(sname, metric="f1_test_A"):
    ms = RES[sname]
    if "LogReg-VI" in ms:
        return "LogReg-VI", ms["LogReg-VI"]
    mn = max(ms, key=lambda m: ms[m][metric])
    return mn, ms[mn]

# ----------------------------------------------- Fig. 1 (framework)
fig, ax = plt.subplots(figsize=(7.16, 1.05)); ax.axis("off")
ax.set_xlim(0, 7.16); ax.set_ylim(0, 1.05)
boxes = [
    ("Disease\narchetypes\n(PROSPECT-D)", "#fdecd4"),
    ("PROSAIL\ncanopy +\nconfounders", "#e2efda"),
    ("Sensor\nsynthesis\n(SRF, noise)", "#dbe7f5"),
    ("Band\nconfiguration\n(fixed / greedy)", "#f0dcec"),
    ("Lightweight ML\n(LR, SVM, RF,\nHGB, MLP)", "#f5e3db"),
    ("Evaluation\n(shift, early,\nedge cost)", "#e8e8e8"),
]
n = len(boxes); gap = 0.26; bw = (7.16 - gap * (n - 1)) / n
for i, (txt, col) in enumerate(boxes):
    x = i * (bw + gap)
    ax.add_patch(plt.Rectangle((x, 0.05), bw, 0.95, facecolor=col,
                 edgecolor="#555555", lw=0.8))
    ax.text(x + bw / 2, 0.525, txt, ha="center", va="center", fontsize=6.4,
            linespacing=1.15)
    if i < n - 1:
        ax.annotate("", xy=(x + bw + gap - 0.02, 0.525),
                    xytext=(x + bw + 0.02, 0.525),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.0))
fig.savefig(f"{FIGD}/fig_framework.pdf"); plt.close(fig)

# ----------------------------------------------- Fig. 2 (spectra)
fig, ax = plt.subplots(1, 2, figsize=(4.6, 1.55),
                       gridspec_kw={"width_ratios": [1.6, 1]})
Xtr, ytr, str_ = DATA["X_train"], DATA["y_train"], DATA["s_train"]
for ci, cls in enumerate(CLASSES):
    mask = ytr == ci
    if cls != "healthy":
        mask &= str_ >= 0.5
    mu, sd = Xtr[mask].mean(0), Xtr[mask].std(0)
    for a, sl in zip(ax, [slice(0, 601), slice(0, 2101)]):
        a.plot(WL[sl], mu[sl], color=CLS_COLORS[cls], lw=1.2,
               label=cls if a is ax[0] else None)
        a.fill_between(WL[sl], mu[sl] - sd[sl], mu[sl] + sd[sl],
                       color=CLS_COLORS[cls], alpha=0.14, lw=0)
for c, f in COMMERCIAL["RedEdge-MX (5)"]:
    ax[0].axvline(c, color="0.45", ls=":", lw=0.7)
ax[1].axvline(1450, color="#c22f2f", ls="--", lw=0.8)
ax[1].text(1500, 0.44, "1450 nm\nwater abs.", fontsize=6, color="#c22f2f")
ax[0].set_xlabel("Wavelength (nm)"); ax[0].set_ylabel("Canopy reflectance")
ax[0].set_title("(a) VNIR; RedEdge-MX centers dotted", fontsize=7.5)
ax[1].set_xlabel("Wavelength (nm)"); ax[1].set_title("(b) Full 400–2500 nm", fontsize=7.5)
ax[0].legend(frameon=False, loc="upper left", fontsize=6)
fig.savefig(f"{FIGD}/fig_spectra.pdf"); plt.close(fig)

# ----------------------------------------------- Fig. 3 (importance)
fig, ax = plt.subplots(figsize=(3.45, 1.75))
pim = np.array(edge["pi_mean"]); pis = np.array(edge["pi_std"])
ax.plot(CAND_CENTERS, pim, color="#333333", lw=1.1)
ax.fill_between(CAND_CENTERS, pim - pis, pim + pis, color="#333333",
                alpha=0.2, lw=0)
for c in selV["selected_centers"][:5]:
    ax.axvline(c, color="#c22f2f", lw=1.0, ls="--", alpha=0.9)
for c, f in COMMERCIAL["RedEdge-MX (5)"]:
    ax.axvline(c, color="#3a6fb0", lw=0.8, ls=":", alpha=0.85)
ax.set_xlabel("Band center (nm)"); ax.set_ylabel("Perm. importance")
ax.set_ylim(top=pim.max() * 1.45)
ax.legend(handles=[
    Line2D([], [], color="#c22f2f", ls="--", lw=1, label="Greedy-5 (silicon)"),
    Line2D([], [], color="#3a6fb0", ls=":", lw=1, label="RedEdge-MX")],
    frameon=False, loc="upper left", ncol=2, fontsize=6.5)
fig.savefig(f"{FIGD}/fig_importance.pdf"); plt.close(fig)

# ----------------------------------------------- Fig. 5 (confusion)
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.85))
short = ["hea.", "chl.", "nec.", "wilt"]
for a, key, ttl in zip(axes, ["A", "B"], ["(a) In-domain", "(b) Shifted"]):
    cm = np.array(bC["conf"][key], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    a.set_xticks(range(4)); a.set_yticks(range(4))
    a.set_xticklabels(short, fontsize=6.5)
    a.set_yticklabels(short if key == "A" else [], fontsize=6.5)
    for i in range(4):
        for j in range(4):
            a.text(j, i, f"{cmn[i, j]*100:.0f}", ha="center", va="center",
                   fontsize=6.5,
                   color="white" if cmn[i, j] > 0.6 else "#1a1a1a")
    a.set_title(ttl, fontsize=7.5); a.set_xlabel("Predicted", fontsize=7)
axes[0].set_ylabel("True", fontsize=7)
fig.tight_layout(w_pad=0.6)
fig.savefig(f"{FIGD}/fig_confusion.pdf"); plt.close(fig)

# ----------------------------------------------- Fig. 4 (band count, early stage, severity)
# quick refit MLPs for severity-detectability curves
def refit_mlp(bands, tag):
    F = featurize_all(bands, tag)
    m = make_pipeline(StandardScaler(),
                      MLPClassifier((64, 32), max_iter=350, random_state=0))
    m.fit(F["train"], DATA["y_train"])
    return m, F

mlp_re, F_re = refit_mlp(COMMERCIAL["RedEdge-MX (5)"], "RedEdge-MX (5)")
mlp_gx, F_gx = refit_mlp(ext_sensors()["GreedyX-5"], "GreedyX-5")

def sev_curve(model, F):
    X = np.vstack([F["test_A"], F["early_A"]])
    y = np.concatenate([DATA["y_test_A"], DATA["y_early_A"]])
    s = np.concatenate([DATA["s_test_A"], DATA["s_early_A"]])
    dis = y != HEALTHY
    pred_dis = model.predict(X) != HEALTHY
    bins = np.array([0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 1.001])
    mids, rec = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = dis & (s >= lo) & (s < hi)
        if m.sum() > 20:
            mids.append((lo + hi) / 2)
            rec.append(pred_dis[m].mean())
    return mids, rec

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2})
fig, axes = plt.subplots(1, 3, figsize=(5.75, 1.78),
                         gridspec_kw={"width_ratios": [1.3, 1, 1]})
show = ["RGB (3)", "Sequoia (4)", "RedEdge-MX (5)", "GreedyX-5"]
lbls = ["RGB\n(3)", "Seq.\n(4)", "RE-MX\n(5)", "GX-5\n(5)"]
xs = np.arange(len(show)); w = 0.38
ks = [3, 4, 5, 6]
ax = axes[0]
gA = [best(f"Greedy-{k}")[1]["f1_test_A"] for k in ks]
xA = [best(f"GreedyX-{k}")[1]["f1_test_A"] for k in ks]
gB = [best(f"Greedy-{k}")[1]["f1_test_B"] for k in ks]
xB = [best(f"GreedyX-{k}")[1]["f1_test_B"] for k in ks]
ax.plot(ks, gA, "o-", color="#c26f2f", lw=1.2, ms=3.5, label="Si (A)")
ax.plot(ks, xA, "o-", color="#7a2fc2", lw=1.2, ms=3.5, label="Si+SWIR (A)")
ax.plot(ks, gB, "o--", color="#c26f2f", lw=0.9, ms=3.5, alpha=0.55, label="Si (B)")
ax.plot(ks, xB, "o--", color="#7a2fc2", lw=0.9, ms=3.5, alpha=0.55, label="Si+SWIR (B)")
mk = {"RGB (3)": ("^", "#777777", "RGB"), "Sequoia (4)": ("D", "#3a8f5f", "Seq."),
      "RedEdge-MX (5)": ("v", "#3a6fb0", "RE-MX")}
for sname, (m_, col, lab) in mk.items():
    nb = int(sname.split("(")[1][0])
    yv = best(sname)[1]["f1_test_A"]
    ax.scatter([nb], [yv], marker=m_, s=20, color=col, zorder=5)
    ax.text(nb + 0.1, yv - 0.012, lab, fontsize=5.5, color=col, va="center")
f59 = json.load(open(f"{RESD}/full59.json"))["LogReg"]
ax.axhline(f59["f1_test_A"], color="0.35", ls=":", lw=0.9)
ax.axhline(f59["f1_test_B"], color="0.35", ls="-.", lw=0.8)
ax.text(3.0, f59["f1_test_A"] + 0.015, "59-band LR (A)", fontsize=5.4, color="0.3", ha="left")
ax.text(3.0, f59["f1_test_B"] + 0.015, "59-band LR (B)", fontsize=5.4, color="0.3", ha="left")
ax.set_xticks(ks); ax.set_xlabel("Number of bands")
ax.set_ylabel("Macro-F1"); ax.set_ylim(0.0, 0.9)
ax.legend(frameon=False, fontsize=5.5, loc="lower right", ncol=2,
          handlelength=1.6, columnspacing=0.8)
ax.set_title("(a) Band count, domain shift", fontsize=8)
eA = [best(s)[1]["early_f1_A"] for s in show]
eB = [best(s)[1]["early_f1_B"] for s in show]
axes[1].bar(xs - w/2, eA, w, color="#3a6fb0", label="Domain A")
axes[1].bar(xs + w/2, eB, w, color="#c9803a", label="Domain B")
axes[1].legend(frameon=False, fontsize=5.5, loc="upper left", ncol=2, handlelength=1.0, columnspacing=0.6)
axes[1].set_ylabel("Early-stage F1"); axes[1].set_ylim(0, 1.2); axes[1].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axes[1].set_title("(b) Early stage ($s \\leq 0.3$)", fontsize=8)
axes[1].set_xticks(xs); axes[1].set_xticklabels(lbls, fontsize=5.4)
m1, r1 = sev_curve(mlp_re, F_re)
m2, r2 = sev_curve(mlp_gx, F_gx)
axes[2].plot(m1, r1, "v-", color="#3a6fb0", ms=3, lw=1.0, label="RedEdge-MX")
axes[2].plot(m2, r2, "o-", color="#7a2fc2", ms=3, lw=1.0, label="GreedyX-5")
axes[2].set_xlabel("Disease severity $s$")
axes[2].set_ylabel("Detection recall")
axes[2].set_title("(c) Recall vs. severity", fontsize=8)
axes[2].legend(frameon=False, fontsize=5.5, loc="lower right")
fig.tight_layout(w_pad=1.0)
fig.savefig(f"{FIGD}/fig_domain.pdf"); plt.close(fig)

# ------------------------------------------------------------- merge
final = {
    "selection_vnir": selV, "selection_ext": selX,
    "edge": edge["edge"], "full59_f1A": edge["full59_f1A"],
    "full59_f1B": edge["full59_f1B"],
    "results": {s: {m: {k: round(float(v), 4) for k, v in mm.items()}
                    for m, mm in ms.items()} for s, ms in RES.items()},
    "conf_A": bC["conf"]["A"], "conf_B": bC["conf"]["B"],
    "sev_curve": {"rededge": [m1, r1], "greedyx5": [m2, r2]},
    "n": {sp: int(len(DATA[f"y_{sp}"]))
          for sp in ["train", "test_A", "test_B", "early_A", "early_B"]},
}
json.dump(final, open(f"{RESD}/results.json", "w"), indent=1)
print("figures + results.json done")
print(sorted(os.listdir(FIGD)))
