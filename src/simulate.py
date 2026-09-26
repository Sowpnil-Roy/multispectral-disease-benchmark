"""
Physics-guided simulation of canopy-level multispectral observations of
healthy and diseased crops using the PROSAIL radiative transfer model.

Disease archetypes are encoded as severity-dependent trajectories of
PROSPECT-D biophysical parameters, grounded in plant-pathology literature:
  - chlorosis-type  : chlorophyll/carotenoid degradation (e.g., viral yellows,
                      early biotrophic infection)
  - necrosis-type   : brown-pigment accumulation + desiccation (e.g.,
                      necrotrophic fungal lesions)
  - wilt-type       : water loss + canopy collapse (e.g., vascular wilts)

Canopy confounders (LAI, leaf angle, soil, sun-sensor geometry) are randomized
to create realistic domain variability; a shifted domain (B) stresses
generalization.
"""
import os
import numpy as np
import prosail

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WL = np.arange(400, 2501)  # PROSAIL output grid, 1 nm

RNG = np.random.default_rng(42)

CLASSES = ["healthy", "chlorosis", "necrosis", "wilt"]


def sample_healthy_leaf(rng):
    return dict(
        n=rng.uniform(1.2, 1.8),
        cab=rng.uniform(35.0, 60.0),
        car=rng.uniform(7.0, 13.0),
        cbrown=rng.uniform(0.0, 0.05),
        cw=rng.uniform(0.010, 0.018),
        cm=rng.uniform(0.006, 0.011),
        ant=rng.uniform(0.0, 1.0),
    )


def apply_disease(leaf, cls, s, rng):
    """Transform healthy leaf parameters according to disease archetype and
    severity s in (0, 1]. Returns (leaf_params, lai_multiplier)."""
    p = dict(leaf)
    lai_mult = 1.0
    if cls == "healthy":
        return p, lai_mult
    if cls == "chlorosis":
        # chlorophyll degrades faster than carotenoids -> yellowing
        p["cab"] = leaf["cab"] * (1.0 - rng.uniform(0.55, 0.75) * s)
        p["car"] = leaf["car"] * (1.0 - rng.uniform(0.25, 0.45) * s)
        p["cbrown"] = leaf["cbrown"] + rng.uniform(0.05, 0.15) * s
        p["ant"] = leaf["ant"] + rng.uniform(0.0, 1.5) * s
    elif cls == "necrosis":
        p["cbrown"] = leaf["cbrown"] + rng.uniform(0.5, 0.9) * s
        p["cab"] = leaf["cab"] * (1.0 - rng.uniform(0.35, 0.60) * s)
        p["car"] = leaf["car"] * (1.0 - rng.uniform(0.30, 0.50) * s)
        p["cw"] = leaf["cw"] * (1.0 - rng.uniform(0.30, 0.55) * s)
        p["n"] = leaf["n"] + rng.uniform(0.1, 0.4) * s  # mesophyll breakdown
    elif cls == "wilt":
        p["cw"] = leaf["cw"] * (1.0 - rng.uniform(0.45, 0.70) * s)
        p["cab"] = leaf["cab"] * (1.0 - rng.uniform(0.10, 0.30) * s)
        p["cm"] = leaf["cm"] * (1.0 + rng.uniform(0.0, 0.25) * s)
        lai_mult = 1.0 - rng.uniform(0.25, 0.50) * s  # canopy collapse
    return p, lai_mult


def sample_scene(domain, rng):
    """Canopy + geometry + soil confounders for domain 'A' or shifted 'B'."""
    if domain == "A":
        return dict(
            lai=rng.uniform(2.0, 5.5),
            ala=rng.uniform(35.0, 65.0),      # mean leaf inclination angle
            hspot=rng.uniform(0.05, 0.2),
            tts=float(np.clip(rng.normal(30.0, 4.0), 20.0, 40.0)),
            tto=rng.uniform(0.0, 5.0),
            psi=rng.uniform(0.0, 180.0),
            psoil=rng.uniform(0.4, 1.0),      # drier / brighter soil
            rsoil=rng.uniform(0.7, 1.1),
        )
    else:  # shifted acquisition domain
        return dict(
            lai=rng.uniform(1.0, 2.5),        # sparse canopy -> soil mixing
            ala=rng.uniform(45.0, 75.0),
            hspot=rng.uniform(0.05, 0.3),
            tts=float(np.clip(rng.normal(48.0, 4.0), 38.0, 58.0)),  # low sun
            tto=rng.uniform(0.0, 10.0),
            psi=rng.uniform(0.0, 180.0),
            psoil=rng.uniform(0.0, 0.5),      # wet / dark soil
            rsoil=rng.uniform(0.4, 0.8),
        )


def simulate_one(cls, s, domain, rng):
    leaf = sample_healthy_leaf(rng)
    leaf, lai_mult = apply_disease(leaf, cls, s, rng)
    sc = sample_scene(domain, rng)
    r = prosail.run_prosail(
        n=leaf["n"], cab=max(leaf["cab"], 0.5), car=max(leaf["car"], 0.5),
        cbrown=min(leaf["cbrown"], 1.0), cw=max(leaf["cw"], 1e-4),
        cm=max(leaf["cm"], 1e-4), ant=max(leaf.get("ant", 0.0), 0.0),
        lai=max(sc["lai"] * lai_mult, 0.3),
        lidfa=sc["ala"], typelidf=2, hspot=sc["hspot"],
        tts=sc["tts"], tto=sc["tto"], psi=sc["psi"],
        psoil=sc["psoil"], rsoil=sc["rsoil"],
        prospect_version="D",
    )
    return np.asarray(r, dtype=np.float32)


def add_sensor_noise(band_refl, domain, rng):
    """Per-band gain error + global illumination scaling + additive noise."""
    if domain == "A":
        gain_sd, add_sd, illum = 0.01, 0.003, (0.98, 1.02)
    else:
        gain_sd, add_sd, illum = 0.02, 0.006, (0.94, 1.06)
    g = rng.normal(1.0, gain_sd, size=band_refl.shape)
    ill = rng.uniform(*illum)
    out = band_refl * g * ill + rng.normal(0.0, add_sd, size=band_refl.shape)
    return np.clip(out, 0.0, 1.0)


def gaussian_srf(center, fwhm):
    sigma = fwhm / 2.3548
    w = np.exp(-0.5 * ((WL - center) / sigma) ** 2)
    return w / w.sum()


def make_srf_matrix(bands):
    """bands: list of (center, fwhm). Returns (n_bands, 2101) matrix."""
    return np.stack([gaussian_srf(c, f) for c, f in bands])


def build_split(n_per_class, domain, severity_range, seed):
    rng = np.random.default_rng(seed)
    X, y, sev = [], [], []
    for ci, cls in enumerate(CLASSES):
        for _ in range(n_per_class):
            s = 0.0 if cls == "healthy" else rng.uniform(*severity_range)
            X.append(simulate_one(cls, s, domain, rng))
            y.append(ci)
            sev.append(s)
    return np.stack(X), np.array(y), np.array(sev)


if __name__ == "__main__":
    import time
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    t0 = time.time()
    specs = {}
    # Train + in-domain test: full severity range
    specs["train"] = build_split(1250, "A", (0.15, 1.0), seed=1)
    specs["test_A"] = build_split(400, "A", (0.15, 1.0), seed=2)
    # Shifted-domain test
    specs["test_B"] = build_split(400, "B", (0.15, 1.0), seed=3)
    # Early-stage evaluation sets (low severity only)
    specs["early_A"] = build_split(300, "A", (0.05, 0.30), seed=4)
    specs["early_B"] = build_split(300, "B", (0.05, 0.30), seed=5)
    out = {}
    for k, (X, y, s) in specs.items():
        out[f"X_{k}"] = X
        out[f"y_{k}"] = y
        out[f"s_{k}"] = s
        print(k, X.shape, "elapsed", round(time.time() - t0, 1), "s")
    np.savez_compressed(os.path.join(ROOT, "data", "spectra.npz"), wl=WL, **out)
    print("saved. total", round(time.time() - t0, 1), "s")
    # compare with the checksums of the paper run (data/checksums.json)
    import hashlib, json
    ref_path = os.path.join(ROOT, "data", "checksums.json")
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        arrays = {"wl": WL, **out}
        bad = [k for k, v in ref.items() if hashlib.sha256(
            np.ascontiguousarray(arrays[k]).tobytes()).hexdigest() != v]
        if bad:
            print("WARNING: arrays differ from the paper run:", bad,
                  "(different library versions or CPU can change float bits;"
                  " run src/compare_results.py after the stages to see if any"
                  " reported number changes)")
        else:
            print("checksums: identical to the paper run")
