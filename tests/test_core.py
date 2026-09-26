"""Fast checks that run without the 80 MB simulated data set (CI-friendly)."""
import json
import os
import subprocess
import sys
import zlib

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import simulate  # noqa: E402


def test_wavelength_grid():
    assert simulate.WL[0] == 400 and simulate.WL[-1] == 2500
    assert len(simulate.WL) == 2101


@pytest.mark.parametrize("center,fwhm", [(450, 12), (760, 12), (1450, 25), (842, 57)])
def test_srf_is_normalized_and_centered(center, fwhm):
    w = simulate.gaussian_srf(center, fwhm)
    assert np.isclose(w.sum(), 1.0)
    assert simulate.WL[np.argmax(w)] == center
    half = w >= w.max() / 2
    width = simulate.WL[half][-1] - simulate.WL[half][0] + 1
    assert abs(width - fwhm) <= 2


def test_archetype_directions():
    rng = np.random.default_rng(0)
    leaf = simulate.sample_healthy_leaf(rng)
    h, m = simulate.apply_disease(leaf, "healthy", 0.0, rng)
    assert h == leaf and m == 1.0
    c, _ = simulate.apply_disease(leaf, "chlorosis", 1.0, rng)
    assert c["cab"] < leaf["cab"] and c["car"] < leaf["car"] and c["cbrown"] > leaf["cbrown"]
    n, _ = simulate.apply_disease(leaf, "necrosis", 1.0, rng)
    assert n["cbrown"] > leaf["cbrown"] + 0.4 and n["cw"] < leaf["cw"] and n["n"] > leaf["n"]
    w, lai = simulate.apply_disease(leaf, "wilt", 1.0, rng)
    assert w["cw"] < leaf["cw"] and w["cm"] >= leaf["cm"] and 0.5 <= lai <= 0.75


def test_domain_ranges():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a, b = simulate.sample_scene("A", rng), simulate.sample_scene("B", rng)
        assert 2.0 <= a["lai"] <= 5.5 and 1.0 <= b["lai"] <= 2.5
        assert 20 <= a["tts"] <= 40 and 38 <= b["tts"] <= 58
        assert a["psoil"] >= 0.4 and b["psoil"] <= 0.5


def test_prosail_forward_is_deterministic_and_bounded():
    x1 = simulate.simulate_one("wilt", 0.8, "A", np.random.default_rng(5))
    x2 = simulate.simulate_one("wilt", 0.8, "A", np.random.default_rng(5))
    assert x1.shape == (2101,)
    assert np.array_equal(x1, x2)
    assert np.all(np.isfinite(x1)) and x1.min() >= 0 and x1.max() < 1


def test_sensor_noise_clipped():
    rng = np.random.default_rng(0)
    out = simulate.add_sensor_noise(np.full((100, 5), 0.999), "B", rng)
    assert out.min() >= 0 and out.max() <= 1


def test_stable_seed_is_process_independent():
    # same formula as stages.stable_seed; must not depend on PYTHONHASHSEED
    seed = lambda *p: zlib.crc32("|".join(map(str, p)).encode()) % (2**31)
    assert seed("GreedyX-5", "train") == seed("GreedyX-5", "train")
    assert seed("GreedyX-5", "train") != seed("GreedyX-5", "test_A")


REQUIRED = ["selection.json", "selection_ext.json", "bench_A.json", "bench_B.json",
            "bench_C.json", "edge.json", "full59.json", "perclass_logreg.json",
            "bootstrap.json", "early_far.json", "replicates.json",
            "selection_stability.json", "results.json"]


@pytest.mark.parametrize("name", REQUIRED)
def test_result_files_present(name):
    path = os.path.join(ROOT, "results", name)
    assert os.path.exists(path)
    json.load(open(path))


def test_tables_match_committed():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "src", "tables.py")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_every_paper_claim_passes():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "src", "verify_paper.py")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout[-3000:]
