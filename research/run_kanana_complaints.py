"""CPU zero-shot evaluation of Kanana-2-v-decide on the 27-way complaint fixture."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import os
import platform
import statistics
import time

import psutil
import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, p):
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def flatten_taxonomy(taxonomy):
    leaves = {leaf["id"]: leaf for leaf in taxonomy["leaves"]}
    criteria = {
        leaf_id: " > ".join(leaf["path"]) + ": " + leaf["description"]
        for leaf_id, leaf in leaves.items()
    }
    return leaves, criteria


def macro_f1(y_true, y_pred, labels):
    scores = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(scores) / len(scores)


def summarize(rows, leaf_ids):
    strict = [row for row in rows if not row["needs_clarification"]]
    ambiguous = [row for row in rows if row["needs_clarification"]]

    def score(items):
        y_true = [row["expected_leaf"] for row in items]
        y_pred = [row["predicted_leaf"] for row in items]
        return {
            "n": len(items),
            "correct": sum(a == b for a, b in zip(y_true, y_pred)),
            "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(items),
            "macro_f1": macro_f1(y_true, y_pred, leaf_ids),
        }

    latencies = [row["latency_ms"] for row in rows]
    confidences = [row["confidence"] for row in rows]
    return {
        "strict": score(strict),
        "by_slice": {name: score([row for row in strict if row["slice"] == name]) for name in sorted({row["slice"] for row in strict})},
        "under_specified": {
            "n": len(ambiguous),
            "acceptable_hits": sum(row["predicted_leaf"] in row["acceptable_leaves"] for row in ambiguous),
            "note": "Forced-choice diagnostic; these cases require clarification.",
        },
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "confidence_mean": statistics.mean(confidences),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="evaluation/models/kanana-2-v-decide-3b-2611")
    parser.add_argument("--fixture", default="datasets/complaints")
    parser.add_argument("--out", default="evaluation/kanana-2-v-decide-flat27-cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    model_dir = ROOT / args.model
    fixture = ROOT / args.fixture
    out = ROOT / args.out
    if out.exists():
        raise RuntimeError(f"Choose a fresh output directory: {out}")
    out.mkdir(parents=True)

    taxonomy_path, cases_path = fixture / "taxonomy.json", fixture / "cases.json"
    taxonomy = json.loads(taxonomy_path.read_text("utf-8"))
    cases = json.loads(cases_path.read_text("utf-8"))
    leaves, criteria = flatten_taxonomy(taxonomy)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    started = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    load_seconds = time.perf_counter() - started
    rss_after_load = process.memory_info().rss

    question = {
        "route": {
            "type": "choice",
            "instructions": "민원 요약을 읽고 가장 적합한 하나의 대분류 > 중분류 > 소분류 경로를 선택하세요.",
            "criteria": criteria,
        }
    }

    # Warm-up also fills the model's candidate embedding cache. Excluded from latency.
    warm_state = f"상담 경로: {cases[0]['channel']}\n민원 요약: {cases[0]['summary']}"
    with torch.inference_mode():
        model.system_one(state=warm_state, questions=question)
    rss_after_warmup = process.memory_info().rss

    rows = []
    for index, case in enumerate(cases, 1):
        state = f"상담 경로: {case['channel']}\n민원 요약: {case['summary']}"
        t0 = time.perf_counter()
        with torch.inference_mode():
            result = model.system_one(state=state, questions=question)
        latency_ms = (time.perf_counter() - t0) * 1000
        answer = result["answers"]["route"]
        row = {
            **case,
            "predicted_leaf": answer["choice"],
            "probabilities": answer["probabilities"],
            "confidence": answer["confidence"],
            "latency_ms": latency_ms,
        }
        rows.append(row)
        if index % 5 == 0:
            (out / "predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
            print(f"flat27 {index}/{len(cases)} latency_ms={latency_ms:.1f}", flush=True)

    summary = {
        "status": "complete",
        "model": "kakaocorp/kanana-2-v-decide-3b-2611",
        "model_local_path": args.model,
        "model_revision": "1489062def4145ae3a0a9261fbac4f4d88a8c6d1",
        "model_weight_bytes": 7330945212,
        "model_weight_sha256": "528930fa261b31ed1f47eecbe7d08efd2e226207508124ae186e1a65dbbbda3b",
        "fixture": args.fixture,
        "fixture_sha256": {"taxonomy.json": sha256(taxonomy_path), "cases.json": sha256(cases_path)},
        "script_sha256": sha256(Path(__file__)),
        "scope": "Synthetic author-defined diagnostic; no independent human review; not a production benchmark.",
        "environment": {
            "device": "cpu",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "threads": args.threads,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "container_memory_limit_bytes": Path("/sys/fs/cgroup/memory.max").read_text().strip() if Path("/sys/fs/cgroup/memory.max").exists() else None,
        },
        "counts": {"candidates": len(criteria), "cases": len(cases), "strict": sum(not x["needs_clarification"] for x in cases)},
        "load_seconds": load_seconds,
        "rss_after_load_bytes": rss_after_load,
        "rss_after_warmup_bytes": rss_after_warmup,
        "method": "flat27 zero-shot; candidate path and description supplied at inference; fixed taxonomy order",
        "metrics": summarize(rows, list(leaves)),
        "prediction_distribution": Counter(row["predicted_leaf"] for row in rows),
    }
    (out / "predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    (out / "executed-run_kanana_complaints.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
