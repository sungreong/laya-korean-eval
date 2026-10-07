"""Rerank EmbeddingGemma shortlists with the trained LAYA decision model."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/laya"))
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import laya
from complaint_schema import build_questions, question
from metrics import wilson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--ks", nargs="+", type=int, default=[3, 5, 8, 12])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = ROOT / args.out
    if out.exists():
        raise RuntimeError("Choose a fresh output directory; existing results are preserved")
    out.mkdir(parents=True)
    taxonomy_path = ROOT / args.taxonomy
    retrieval_path = ROOT / args.retrieval
    taxonomy = json.loads(taxonomy_path.read_text("utf-8"))
    retrieval_rows = json.loads(retrieval_path.read_text("utf-8"))
    leaves, _, questions = build_questions(taxonomy)

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    agent = laya.load(str(ROOT / args.model), device="cpu", compile=False)
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}

    def ask(state, candidate_ids):
        criteria = {leaf_id: questions["flat"][leaf_id] for leaf_id in candidate_ids}
        response = agent.predict(state, question(criteria), max_len=2048, head_max_len=1536)
        answer = response["answers"]["route"]
        if len(answer["probabilities"]) != len(criteria):
            raise RuntimeError("Candidate count changed during encoding")
        return answer["choice"], answer, response.get("usage", {})

    results = {}
    summaries = {}
    for k in args.ks:
        first = retrieval_rows[0]
        warm_ids = [item["leaf_id"] for item in first["rankings"][str(args.dimension)]["top"][:k]]
        warm_state = f"상담 경로: {first['channel']}\n민원 요약: {first['summary']}"
        ask(warm_state, warm_ids)
        rows = []
        for index, source in enumerate(retrieval_rows, 1):
            candidate_ids = [item["leaf_id"] for item in source["rankings"][str(args.dimension)]["top"][:k]]
            state = f"상담 경로: {source['channel']}\n민원 요약: {source['summary']}"
            start = time.perf_counter()
            predicted, answer, usage = ask(state, candidate_ids)
            rerank_latency_ms = (time.perf_counter() - start) * 1000
            rows.append({
                "id": source["id"], "channel": source["channel"], "slice": source["slice"],
                "summary": source["summary"], "expected_leaf": source.get("expected_leaf"),
                "acceptable_leaves": source["acceptable_leaves"],
                "needs_clarification": source["needs_clarification"],
                "retrieval_candidates": candidate_ids,
                "gold_retrieved": source.get("expected_leaf") in candidate_ids if source.get("expected_leaf") else None,
                "predicted_leaf": predicted,
                "predicted_path": leaves[predicted]["path"],
                "retrieval_latency_ms": source["query_latency_ms"],
                "rerank_latency_ms": rerank_latency_ms,
                "end_to_end_latency_ms": source["query_latency_ms"] + rerank_latency_ms,
                "answer": answer, "usage": usage,
            })
            if index % 10 == 0:
                print("k", k, index, "/", len(retrieval_rows), flush=True)

        strict = [row for row in rows if row["expected_leaf"]]
        retrieved = [row for row in strict if row["gold_retrieved"]]
        correct = sum(row["predicted_leaf"] == row["expected_leaf"] for row in strict)
        correct_retrieved = sum(row["predicted_leaf"] == row["expected_leaf"] for row in retrieved)
        summaries[str(k)] = {
            "n": len(strict), "gold_retrieved": len(retrieved),
            "retrieval_recall": len(retrieved) / len(strict),
            "correct": correct, "accuracy": correct / len(strict),
            "accuracy_given_gold_retrieved": correct_retrieved / len(retrieved) if retrieved else 0,
            "wilson95_accuracy": wilson(correct, len(strict)),
            "under_specified_n": sum(row["needs_clarification"] for row in rows),
            "under_specified_acceptable_hits": sum(
                row["predicted_leaf"] in row["acceptable_leaves"] for row in rows if row["needs_clarification"]
            ),
            "retrieval_latency_ms_p50": float(np.percentile([row["retrieval_latency_ms"] for row in rows], 50)),
            "rerank_latency_ms_p50": float(np.percentile([row["rerank_latency_ms"] for row in rows], 50)),
            "end_to_end_latency_ms_p50": float(np.percentile([row["end_to_end_latency_ms"] for row in rows], 50)),
            "end_to_end_latency_ms_p95": float(np.percentile([row["end_to_end_latency_ms"] for row in rows], 95)),
        }
        results[str(k)] = rows
        (out / f"rerank_k{k}_predictions.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = {
        "status": "complete", "model": args.model, "taxonomy": args.taxonomy,
        "retrieval": args.retrieval, "dimension": args.dimension, "ks": args.ks,
        "taxonomy_sha256": hashlib.sha256(taxonomy_path.read_bytes()).hexdigest(),
        "retrieval_sha256": hashlib.sha256(retrieval_path.read_bytes()).hexdigest(),
        "candidate_description": "Existing flat leaf name and description; no full-path format change.",
        "metrics": summaries,
        "scope": "Synthetic author-defined diagnostic; no independent human review; not a production benchmark.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "executed-run_retrieval_rerank.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
