"""Paired comparison of complaint-routing predictions on the 81 strict cases."""
from pathlib import Path
import argparse
import json
import math
import random


def load(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return {row["id"]: row for row in rows if row["expected_leaf"] is not None}


def correct(row):
    return row["predicted_leaf"] == row["expected_leaf"]


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(reference, candidate, seed=42, samples=10000):
    ids = sorted(set(reference) & set(candidate))
    pairs = [(correct(reference[k]), correct(candidate[k])) for k in ids]
    b = sum(a and not z for a, z in pairs)
    c = sum(not a and z for a, z in pairs)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(sum(z - a for a, z in draw) / len(draw))
    deltas.sort()
    return {
        "n": len(pairs),
        "reference_correct": sum(a for a, _ in pairs),
        "candidate_correct": sum(z for _, z in pairs),
        "accuracy_delta": (sum(z for _, z in pairs) - sum(a for a, _ in pairs)) / len(pairs),
        "paired_bootstrap_95": [deltas[249], deltas[9749]],
        "discordant": {"reference_only": b, "candidate_only": c},
        "mcnemar_exact_two_sided_p": exact_mcnemar(b, c),
        "bootstrap_samples": samples,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--laya", required=True)
    parser.add_argument("--kobert", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    base, laya, kobert = load(args.base), load(args.laya), load(args.kobert)
    result = {
        "scope": "flat27, 81 single-answer held-out synthetic complaint cases",
        "laya_domain_vs_base": compare(base, laya),
        "kobert_mask_vs_base_laya": compare(base, kobert),
        "note": "Exploratory paired tests; no multiple-comparison correction.",
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
