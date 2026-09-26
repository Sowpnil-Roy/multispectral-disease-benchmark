"""Compare a fresh reproduction run with the reference results.

Usage:  python compare_results.py <reference_dir> <new_dir> [--tol 1e-9]

Every numeric leaf of every JSON file is compared. Timing fields
(latency_us) are machine-dependent and are reported but never fail.
Exit code 1 on any other difference.
"""
import argparse
import json
import os
import sys

SKIP_KEYS = {"latency_us"}


def walk(a, b, path, out, tol):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            out.append((path, "key mismatch", a if not isinstance(a, dict) else sorted(a),
                        b if not isinstance(b, dict) else sorted(b)))
            return
        for k in a:
            walk(a[k], b[k], f"{path}.{k}", out, tol)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            out.append((path, "length mismatch", len(a), len(b) if isinstance(b, list) else b))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > tol:
            out.append((path, "value", a, b))
    elif a != b:
        out.append((path, "value", a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()
    hard, soft, missing = [], [], []
    for name in sorted(os.listdir(args.ref)):
        if not name.endswith(".json"):
            continue
        pn = os.path.join(args.new, name)
        if not os.path.exists(pn):
            missing.append(name); continue
        diffs = []
        walk(json.load(open(os.path.join(args.ref, name))), json.load(open(pn)),
             name, diffs, args.tol)
        for d in diffs:
            (soft if d[0].split(".")[-1] in SKIP_KEYS else hard).append(d)
    for d in soft:
        print("TIMING (ignored)", *d)
    for d in hard:
        print("DIFF", *d)
    for m in missing:
        print("MISSING", m)
    print(f"\n{len(hard)} differences, {len(missing)} missing files, "
          f"{len(soft)} timing fields ignored")
    sys.exit(1 if hard or missing else 0)


if __name__ == "__main__":
    main()
