"""Evaluate EmbeddingGemma 2 retrieval over complaint label descriptions."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

import numpy as np
import psutil
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]


def normalize(array):
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def summarize(rows, dim, ks):
    strict = [row for row in rows if row.get("expected_leaf")]
    ranks = [row["rankings"][str(dim)]["gold_rank"] for row in strict]
    return {
        "n": len(strict),
        "top1_correct": sum(rank == 1 for rank in ranks),
        "top1_accuracy": sum(rank == 1 for rank in ranks) / len(ranks),
        "recall_at_k": {str(k): sum(rank <= k for rank in ranks) / len(ranks) for k in ks},
        "mean_reciprocal_rank": sum(1 / rank for rank in ranks) / len(ranks),
        "query_latency_ms_p50": float(np.percentile([row["query_latency_ms"] for row in rows], 50)),
        "query_latency_ms_p95": float(np.percentile([row["query_latency_ms"] for row in rows], 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--taxonomy", default="datasets/complaints/taxonomy.json")
    parser.add_argument("--cases", default="datasets/complaints/cases.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dimensions", nargs="+", type=int, default=[128, 256, 768])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 8, 12])
    args = parser.parse_args()

    out = ROOT / args.out
    if out.exists():
        raise RuntimeError("Choose a fresh output directory; existing results are preserved")
    out.mkdir(parents=True)
    taxonomy_path, cases_path = ROOT / args.taxonomy, ROOT / args.cases
    taxonomy = json.loads(taxonomy_path.read_text("utf-8"))
    cases = json.loads(cases_path.read_text("utf-8"))
    leaves = {leaf["id"]: leaf for leaf in taxonomy["leaves"]}
    titles = {leaf_id: " > ".join(leaf["path"]) for leaf_id, leaf in leaves.items()}
    documents = [f"title: {titles[leaf_id]} | text: {leaves[leaf_id]['description']}" for leaf_id in leaves]
    leaf_ids = list(leaves)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    process = psutil.Process()
    load_start = time.perf_counter()
    model = SentenceTransformer(
        str(ROOT / args.model),
        device="cpu",
        config_kwargs={"vision_config": None, "audio_config": None},
    )
    load_seconds = time.perf_counter() - load_start
    rss_after_load = process.memory_info().rss

    document_start = time.perf_counter()
    document_full = np.asarray(model.encode(documents, normalize_embeddings=True, batch_size=16))
    document_seconds = time.perf_counter() - document_start
    rss_after_documents = process.memory_info().rss

    # Warm-up excluded from latency measurements.
    model.encode(cases[0]["summary"], prompt_name="SearchQuery", normalize_embeddings=True)
    rows = []
    for index, case in enumerate(cases, 1):
        start = time.perf_counter()
        query_full = np.asarray(model.encode(
            case["summary"], prompt_name="SearchQuery", normalize_embeddings=True
        )).reshape(1, -1)
        query_latency_ms = (time.perf_counter() - start) * 1000
        rankings = {}
        for dim in args.dimensions:
            query = normalize(query_full[:, :dim])[0]
            docs = normalize(document_full[:, :dim])
            scores = docs @ query
            order = np.argsort(-scores)
            ranked_ids = [leaf_ids[i] for i in order]
            gold_rank = ranked_ids.index(case["expected_leaf"]) + 1 if case.get("expected_leaf") else None
            rankings[str(dim)] = {
                "gold_rank": gold_rank,
                "top": [{"leaf_id": leaf_ids[i], "score": float(scores[i])}
                        for i in order[:max(args.ks)]],
            }
        rows.append(dict(case, query_latency_ms=query_latency_ms, rankings=rankings))
        if index % 10 == 0:
            print(index, "/", len(cases), flush=True)

    summary = {
        "status": "complete",
        "model": args.model,
        "model_revision": "914f7f89142e33e77833254d9c9b90c3cef7303b",
        "model_mode": "text-only; vision_config=None; audio_config=None",
        "taxonomy": args.taxonomy,
        "cases": args.cases,
        "taxonomy_sha256": hashlib.sha256(taxonomy_path.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "candidate_count": len(leaves),
        "dimensions": args.dimensions,
        "ks": args.ks,
        "document_format": "title: 대 > 중 > 소 | text: description",
        "query_prompt": "SearchQuery",
        "load_seconds": load_seconds,
        "document_encode_seconds": document_seconds,
        "rss_after_load_bytes": rss_after_load,
        "rss_after_document_encode_bytes": rss_after_documents,
        "metrics": {str(dim): summarize(rows, dim, args.ks) for dim in args.dimensions},
        "scope": "Synthetic author-defined diagnostic; no independent human review; not a production benchmark.",
    }
    (out / "predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "executed-run_embedding_retrieval.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
