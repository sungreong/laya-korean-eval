"""Rerank the eight paths produced by Kanana hierarchical Top-2 search."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import shutil
import time

import psutil
import torch
from transformers import AutoModel

from complaint_schema import build_questions
from run_kanana_hierarchy import apply_description_profile, build_indexes, summarize


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="evaluation/models/kanana-2-v-decide-3b-2611")
    parser.add_argument("--fixture", default="datasets/complaints")
    parser.add_argument("--source", required=True, help="Top-2 prediction JSON containing final_branches")
    parser.add_argument("--out", required=True)
    parser.add_argument("--description-profile")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_dir = ROOT / args.model
    fixture = ROOT / args.fixture
    source_path = ROOT / args.source
    out = ROOT / args.out
    if out.exists() and not args.resume:
        raise RuntimeError(f"Choose a fresh output directory or pass --resume: {out}")
    out.mkdir(parents=True, exist_ok=True)

    taxonomy_path = fixture / "taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_text("utf-8"))
    source_rows = json.loads(source_path.read_text("utf-8"))
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    leaves, majors, questions, _, _ = build_indexes(taxonomy)

    profile_path = ROOT / args.description_profile if args.description_profile else None
    profile = json.loads(profile_path.read_text("utf-8")) if profile_path else None
    questions = apply_description_profile(questions, leaves, majors, profile)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    load_seconds = time.perf_counter() - load_started

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")

    predictions_path = out / "predictions.json"
    rows = (
        json.loads(predictions_path.read_text("utf-8"))
        if predictions_path.exists() and args.resume
        else []
    )
    completed_ids = {row["id"] for row in rows}

    def candidate_text(leaf_id):
        leaf = leaves[leaf_id]
        base = questions["bottom"][leaf["middle_id"]][leaf_id]
        return f"{' > '.join(leaf['path'])} — {base}"

    def rerank(source_row):
        candidate_ids = [branch["leaf_id"] for branch in source_row["final_branches"]]
        if len(candidate_ids) != 8 or len(set(candidate_ids)) != 8:
            raise RuntimeError(f"Expected 8 unique branches for {source_row['id']}")
        criteria = {leaf_id: candidate_text(leaf_id) for leaf_id in candidate_ids}
        state = f"상담 경로: {source_row['channel']}\n민원 요약: {source_row['summary']}"
        question = {
            "final_route": {
                "type": "choice",
                "instructions": (
                    "민원 요약의 핵심 해결 요청을 기준으로 후보의 전체 대분류 > 중분류 > "
                    "소분류 경로와 설명을 직접 비교해 가장 적합한 최종 경로 하나를 선택하세요."
                ),
                "criteria": criteria,
            }
        }
        started = time.perf_counter()
        with torch.inference_mode():
            response = model.system_one(state=state, questions=question)
        latency_ms = (time.perf_counter() - started) * 1000
        answer = response["answers"]["final_route"]
        if set(answer["probabilities"]) != set(criteria):
            raise RuntimeError(f"Candidate IDs changed for {source_row['id']}")
        trace = {
            "stage": "final8_direct_rerank",
            "candidate_count": 8,
            "choice": answer["choice"],
            "probabilities": answer["probabilities"],
            "confidence": answer["confidence"],
            "latency_ms": latency_ms,
        }
        return answer["choice"], trace, latency_ms

    # Warm candidate encodings before measured requests, matching the earlier protocol.
    if source_rows:
        rerank(source_rows[0])
    rss_after_warmup = process.memory_info().rss

    for index, source_row in enumerate(source_rows, 1):
        if source_row["id"] in completed_ids:
            continue
        predicted_leaf, final_trace, rerank_latency_ms = rerank(source_row)
        row = {
            **source_row,
            "pre_rerank_predicted_leaf": source_row["predicted_leaf"],
            "pre_rerank_predicted_path": source_row["predicted_path"],
            "predicted_leaf": predicted_leaf,
            "predicted_path": leaves[predicted_leaf]["path"],
            "branch_latency_ms": source_row["latency_ms"],
            "rerank_latency_ms": rerank_latency_ms,
            "latency_ms": source_row["latency_ms"] + rerank_latency_ms,
            "trace": source_row["trace"] + [final_trace],
        }
        rows.append(row)
        save("predictions.json", rows)
        if index % 5 == 0:
            print(
                f"final_rerank {index}/{len(source_rows)} predicted={predicted_leaf} "
                f"rerank_latency_ms={rerank_latency_ms:.1f}",
                flush=True,
            )

    metrics = summarize(rows, leaves, "top2_branch")
    rerank_latencies = [row["rerank_latency_ms"] for row in rows]
    metrics["rerank_latency_ms_mean"] = sum(rerank_latencies) / len(rerank_latencies)
    summary = {
        "status": "complete",
        "model": "kakaocorp/kanana-2-v-decide-3b-2611",
        "model_revision": "1489062def4145ae3a0a9261fbac4f4d88a8c6d1",
        "scope": "Synthetic author-defined diagnostic; no independent human review; not a production benchmark.",
        "method": (
            "Reuse the final eight paths from per-branch Top-2 search, then make one additional "
            "Kanana choice call over all eight full paths."
        ),
        "source": args.source,
        "source_sha256": sha256(source_path),
        "fixture_sha256": {"taxonomy.json": sha256(taxonomy_path)},
        "description_profile": {
            "path": args.description_profile,
            "sha256": sha256(profile_path) if profile_path else None,
            "name": profile["profile"] if profile else "taxonomy-basic",
        },
        "environment": {
            "device": "cpu",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "threads": args.threads,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "load_seconds": load_seconds,
        "rss_after_warmup_bytes": rss_after_warmup,
        "metrics": metrics,
        "script_sha256": sha256(Path(__file__)),
    }
    save("summary.json", summary)
    shutil.copy2(Path(__file__), out / "executed-run_kanana_final_rerank.py")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
