# bb-embeddings | tests/golden_test.py | v1.0.0 | 2026-11-17 | BB
# Golden-vector test. Embeds every image in tests/fixtures/ with the service's own
# model.py and compares each vector to tests/golden/vectors.json.
#
# Why: stored gallery vectors (ControlTower bbscan_app.asset_signatures) were made by
# one torch build. A dependency bump must not move the vectors, or old rows stop
# matching. The pass bar is cosine >= 0.9999 per image (match thresholds are 0.75 / 0.55).
#
# Usage (inside the service image, repo root as cwd):
#   python3 tests/golden_test.py              embed, write tests/out/vectors.json, compare to golden
#   python3 tests/golden_test.py --self-test  test the comparison logic only (no torch needed)
#   python3 tests/golden_test.py --compare A.json B.json   compare two vector files (no torch needed)
#
# To (re)capture the golden file: take tests/out/vectors.json from a run on the OLD
# pins (CI uploads it as the "vectors" artifact) and commit it as tests/golden/vectors.json.
# Never regenerate it on the new pins: that compares the new build to itself.

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
GOLDEN = os.path.join(HERE, "golden", "vectors.json")
OUT = os.path.join(HERE, "out", "vectors.json")

DIM = 384
MIN_COSINE = 0.9999
NORM_TOL = 1e-5
IMAGE_EXT = (".png", ".jpg", ".jpeg")


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def cosine(a, b):
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def check_vector(name, v):
    """Shape checks on one vector. Returns a list of problems (empty = fine)."""
    if not isinstance(v, list) or len(v) != DIM:
        return [f"{name}: dim {len(v) if isinstance(v, list) else 'n/a'}, want {DIM}"]
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
        return [f"{name}: non-finite value"]
    n = norm(v)
    if abs(n - 1.0) > NORM_TOL:
        return [f"{name}: norm {n:.8f}, want 1.0 +/- {NORM_TOL}"]
    return []


def compare(golden, got):
    """Compare two {name: vector} maps. Returns (problems, rows); rows = (name, cosine)."""
    problems, rows = [], []
    if not golden:
        problems.append("golden file has no vectors")
    for name in sorted(set(golden) | set(got)):
        if name not in got:
            problems.append(f"{name}: in golden, not produced (fixture removed?)")
            continue
        if name not in golden:
            problems.append(f"{name}: produced, not in golden (new fixture needs a golden vector)")
            continue
        bad = check_vector(name, got[name]) + check_vector(f"{name} (golden)", golden[name])
        if bad:
            problems.extend(bad)
            continue
        c = cosine(golden[name], got[name])
        rows.append((name, c))
        if c < MIN_COSINE:
            problems.append(f"{name}: cosine {c:.8f} below {MIN_COSINE}")
    return problems, rows


def report(problems, rows, label):
    for name, c in rows:
        print(f"  {'ok  ' if c >= MIN_COSINE else 'FAIL'} {name:28s} cosine {c:.8f}")
    if rows:
        print(f"  min cosine {min(c for _, c in rows):.8f} over {len(rows)} images ({label})")
    for p in problems:
        print(f"  PROBLEM {p}")
    print(f"golden test: {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
    return 1 if problems else 0


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def self_test():
    """The comparison logic, on made-up vectors. Each case names what it must do."""
    unit = [1.0 / math.sqrt(DIM)] * DIM
    other = [0.0] * DIM
    other[0] = 1.0
    nudged = list(unit)
    nudged[0] += 0.05                      # cosine ~0.9987 after renorm: below the bar
    n = norm(nudged)
    nudged = [x / n for x in nudged]
    tiny = list(unit)
    tiny[0] += 1e-7                        # float noise: must pass
    cases = [
        ("identical passes", {"a": unit}, {"a": unit}, 0),
        ("float noise passes", {"a": unit}, {"a": tiny}, 0),
        ("a different vector fails", {"a": unit}, {"a": other}, 1),
        ("a small real shift fails", {"a": unit}, {"a": nudged}, 1),
        ("a missing image fails", {"a": unit, "b": unit}, {"a": unit}, 1),
        ("an extra image fails", {"a": unit}, {"a": unit, "b": unit}, 1),
        ("a wrong dim fails", {"a": unit}, {"a": unit[:-1]}, 1),
        ("an un-normalized vector fails", {"a": unit}, {"a": [x * 1.01 for x in unit]}, 1),
        ("a NaN fails", {"a": unit}, {"a": [float("nan")] + unit[1:]}, 1),
        ("an empty golden fails", {}, {}, 1),
    ]
    failed = 0
    for label, golden, got, want in cases:
        problems, _ = compare(golden, got)
        ok = (1 if problems else 0) == want
        failed += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    print(f"self-test: {len(cases) - failed} pass, {failed} fail")
    return 1 if failed else 0


def embed_all():
    import io
    import torch
    import torchvision
    import PIL
    from model import load_model, encode_image

    model = load_model()
    names = sorted(n for n in os.listdir(FIXTURES) if n.lower().endswith(IMAGE_EXT))
    vectors, problems = {}, []
    for name in names:
        with open(os.path.join(FIXTURES, name), "rb") as f:
            raw = f.read()
        v1 = encode_image(model, io.BytesIO(raw))
        v2 = encode_image(model, io.BytesIO(raw))
        if v1 != v2:
            problems.append(f"{name}: not deterministic (same image, two vectors)")
        vectors[name] = v1
    meta = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "python": sys.version.split()[0],
        "model": "dinov2_vits14",
    }
    return meta, vectors, problems


def main(argv):
    if "--self-test" in argv:
        return self_test()
    if "--compare" in argv:
        i = argv.index("--compare")
        a, b = load(argv[i + 1]), load(argv[i + 2])
        problems, rows = compare(a["vectors"], b["vectors"])
        return report(problems, rows, f"{a.get('meta', {}).get('torch')} vs {b.get('meta', {}).get('torch')}")

    sys.path.insert(0, os.path.dirname(HERE))
    meta, vectors, problems = embed_all()
    print(f"embedded {len(vectors)} images on {meta}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "vectors": vectors}, f)
    print(f"wrote {OUT}")

    if not os.path.exists(GOLDEN):
        # a check that cannot run is not a pass
        print(f"golden test: FAIL (no golden file at {GOLDEN}; commit tests/out/vectors.json from the OLD pins)")
        return 1
    golden = load(GOLDEN)
    more, rows = compare(golden["vectors"], vectors)
    return report(problems + more, rows, f"golden torch {golden.get('meta', {}).get('torch')} vs {meta['torch']}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
