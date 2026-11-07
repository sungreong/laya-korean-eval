"""Verify Kanana metrics and compare paired predictions on the shared complaint fixture."""
from pathlib import Path
import json
import math
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation/kanana-2-v-decide-flat27-cpu"
OUT = ROOT / "results/kanana-2-v-decide-2026-11-08"


def wilson(correct, n, z=1.959963984540054):
    p = correct / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_delta(a, b, repeats=10000, seed=42):
    rng = random.Random(seed)
    deltas = []
    n = len(a)
    for _ in range(repeats):
        indexes = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(a[i] - b[i] for i in indexes) / n)
    deltas.sort()
    return [deltas[int(0.025 * repeats)], deltas[int(0.975 * repeats)]]


rows = json.loads((EVAL / "predictions.json").read_text("utf-8"))
strict = [row for row in rows if not row["needs_clarification"]]
labels = list(strict[0]["probabilities"])
correct = [int(row["predicted_leaf"] == row["expected_leaf"]) for row in strict]
max_probs = [max(row["probabilities"].values()) for row in strict]
nll = -sum(math.log(max(row["probabilities"][row["expected_leaf"]], 1e-15)) for row in strict) / len(strict)
brier = sum(sum((prob - int(label == row["expected_leaf"])) ** 2 for label, prob in row["probabilities"].items()) for row in strict) / len(strict)
ece = 0.0
for lower in [i / 10 for i in range(10)]:
    ids = [i for i, prob in enumerate(max_probs) if lower <= prob < lower + 0.1 or (lower == 0.9 and prob == 1.0)]
    if ids:
        ece += len(ids) / len(strict) * abs(sum(correct[i] for i in ids) / len(ids) - sum(max_probs[i] for i in ids) / len(ids))

kanana_by_id = {row["id"]: row for row in strict}
baselines = {
    "LAYA base flat27": (ROOT / "results/complaints-2026-10-07/base/flat27_predictions.json", lambda row: row["predicted_leaf"]),
    "LAYA complaint-trained flat27": (ROOT / "results/complaint-training-2026-10-07/laya/test/flat27_predictions.json", lambda row: row["predicted_leaf"]),
    "EmbeddingGemma 2 256d top1": (ROOT / "results/embeddinggemma2-retrieval-2026-10-07/retrieval-main27/predictions.json", lambda row: row["rankings"]["256"]["top"][0]["leaf_id"]),
}
comparisons = {}
for name, (path, predict) in baselines.items():
    source = [row for row in json.loads(path.read_text("utf-8")) if not row["needs_clarification"]]
    source_by_id = {row["id"]: row for row in source}
    base_correct = [int(predict(source_by_id[row["id"]]) == row["expected_leaf"]) for row in strict]
    b = sum(k == 1 and v == 0 for k, v in zip(correct, base_correct))
    c = sum(k == 0 and v == 1 for k, v in zip(correct, base_correct))
    comparisons[name] = {
        "baseline_correct": sum(base_correct),
        "baseline_accuracy": sum(base_correct) / len(base_correct),
        "kanana_correct": sum(correct),
        "kanana_accuracy": sum(correct) / len(correct),
        "difference_percentage_points": 100 * (sum(correct) - sum(base_correct)) / len(correct),
        "paired_bootstrap_95_difference": bootstrap_delta(correct, base_correct),
        "kanana_only_correct": b,
        "baseline_only_correct": c,
        "exact_mcnemar_p": exact_mcnemar(b, c),
    }

errors = []
for row in strict:
    if row["predicted_leaf"] != row["expected_leaf"]:
        ranked = sorted(row["probabilities"].items(), key=lambda item: item[1], reverse=True)[:3]
        errors.append({
            "id": row["id"],
            "slice": row["slice"],
            "summary": row["summary"],
            "expected_leaf": row["expected_leaf"],
            "predicted_leaf": row["predicted_leaf"],
            "top3": ranked,
            "gold_probability": row["probabilities"][row["expected_leaf"]],
        })

summary = {
    "scope": "Synthetic author-defined diagnostic; no independent human review; not a production benchmark.",
    "n": len(strict),
    "correct": sum(correct),
    "accuracy": sum(correct) / len(correct),
    "wilson95_accuracy": wilson(sum(correct), len(correct)),
    "multiclass_brier": brier,
    "nll": nll,
    "ece_10_bin": ece,
    "mean_max_probability": sum(max_probs) / len(max_probs),
    "mean_max_probability_correct": sum(p for p, c in zip(max_probs, correct) if c) / sum(correct),
    "mean_max_probability_wrong": sum(p for p, c in zip(max_probs, correct) if not c) / (len(correct) - sum(correct)),
    "comparisons": comparisons,
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
(OUT / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), "utf-8")
shutil.copy2(EVAL / "summary.json", OUT / "summary.json")
shutil.copy2(EVAL / "predictions.json", OUT / "predictions.json")
shutil.copy2(EVAL / "executed-run_kanana_complaints.py", OUT / "executed-run_kanana_complaints.py")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(errors, ensure_ascii=False, indent=2))
