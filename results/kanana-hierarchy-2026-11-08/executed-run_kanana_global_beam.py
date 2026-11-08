"""Evaluate a three-call global Top-2 hierarchy with Kanana V-Decide."""
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

from run_kanana_hierarchy import (
    apply_description_profile,
    build_indexes,
    summarize,
    top_keys,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="evaluation/models/kanana-2-v-decide-3b-2611")
    parser.add_argument("--fixture", default="datasets/complaints")
    parser.add_argument("--out", default="evaluation/kanana-2-v-decide-global-beam-cpu")
    parser.add_argument("--description-profile")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_dir = ROOT / args.model
    fixture = ROOT / args.fixture
    out = ROOT / args.out
    if out.exists() and not args.resume:
        raise RuntimeError(f"Choose a fresh output directory or pass --resume: {out}")
    out.mkdir(parents=True, exist_ok=True)

    taxonomy_path = fixture / "taxonomy.json"
    cases_path = fixture / "cases.json"
    taxonomy = json.loads(taxonomy_path.read_text("utf-8"))
    cases = json.loads(cases_path.read_text("utf-8"))
    if args.limit is not None:
        cases = cases[: args.limit]
    leaves, majors, questions, middle_to_major, middle_names = build_indexes(taxonomy)

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

    def ask(state, instructions, criteria, stage):
        question = {
            "route": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        }
        started = time.perf_counter()
        with torch.inference_mode():
            response = model.system_one(state=state, questions=question)
        latency_ms = (time.perf_counter() - started) * 1000
        answer = response["answers"]["route"]
        if set(answer["probabilities"]) != set(criteria):
            raise RuntimeError(f"Candidate IDs changed at {stage}")
        return {
            "stage": stage,
            "candidate_count": len(criteria),
            "choice": answer["choice"],
            "probabilities": answer["probabilities"],
            "confidence": answer["confidence"],
            "latency_ms": latency_ms,
        }

    def base_state(case):
        return f"상담 경로: {case['channel']}\n민원 요약: {case['summary']}"

    def predict(case):
        state = base_state(case)
        started = time.perf_counter()

        major_trace = ask(
            state,
            "민원 요약의 핵심 해결 요청을 기준으로 가장 적합한 대분류를 선택하세요.",
            questions["top"],
            "major3_global_top2",
        )
        major_ids = top_keys(major_trace["probabilities"], 2)

        middle_criteria = {}
        for major_id in major_ids:
            for middle_id, description in questions["middle"][major_id].items():
                middle_criteria[middle_id] = (
                    f"{majors[major_id]['name']} > {middle_names[middle_id]} — {description}"
                )
        middle_state = state + "\n1차 대분류 후보: " + ", ".join(
            majors[major_id]["name"] for major_id in major_ids
        )
        middle_trace = ask(
            middle_state,
            "대분류와 중분류의 전체 경로를 비교해 가장 적합한 대 > 중 경로를 선택하세요.",
            middle_criteria,
            "middle6_global_top2",
        )
        middle_ids = top_keys(middle_trace["probabilities"], 2)

        leaf_criteria = {}
        for middle_id in middle_ids:
            major_id = middle_to_major[middle_id]
            for leaf_id, description in questions["bottom"][middle_id].items():
                leaf_criteria[leaf_id] = (
                    f"{majors[major_id]['name']} > {middle_names[middle_id]} > "
                    f"{leaves[leaf_id]['path'][2]} — {description}"
                )
        leaf_state = middle_state + "\n2차 대·중 경로 후보: " + ", ".join(
            f"{majors[middle_to_major[middle_id]]['name']} > {middle_names[middle_id]}"
            for middle_id in middle_ids
        )
        leaf_trace = ask(
            leaf_state,
            "민원 원문과 후보의 전체 대 > 중 > 소 경로를 직접 비교해 최종 경로 하나를 선택하세요.",
            leaf_criteria,
            "leaf6_global_top1",
        )
        predicted_leaf = leaf_trace["choice"]

        if case["expected_leaf"] is None:
            survival = {
                "major_global_top2": None,
                "middle_global_top2": None,
                "leaf_final6": None,
            }
        else:
            expected = leaves[case["expected_leaf"]]
            survival = {
                "major_global_top2": expected["major_id"] in major_ids,
                "middle_global_top2": expected["middle_id"] in middle_ids,
                "leaf_final6": case["expected_leaf"] in leaf_criteria,
            }
        return (
            predicted_leaf,
            [major_trace, middle_trace, leaf_trace],
            {
                "major_top2": major_ids,
                "middle_top2": middle_ids,
                "final_leaf_candidates": list(leaf_criteria),
            },
            survival,
            (time.perf_counter() - started) * 1000,
        )

    # Warm the exact three-stage path before measured requests.
    if cases:
        predict(cases[0])
    rss_after_warmup = process.memory_info().rss

    predictions_path = out / "predictions.json"
    rows = (
        json.loads(predictions_path.read_text("utf-8"))
        if predictions_path.exists() and args.resume
        else []
    )
    completed_ids = {row["id"] for row in rows}
    for index, case in enumerate(cases, 1):
        if case["id"] in completed_ids:
            continue
        leaf_id, trace, pools, survival, latency_ms = predict(case)
        row = {
            **case,
            "predicted_leaf": leaf_id,
            "predicted_path": leaves[leaf_id]["path"],
            "expected_path": (
                leaves[case["expected_leaf"]]["path"]
                if case["expected_leaf"] is not None
                else None
            ),
            "latency_ms": latency_ms,
            "trace": trace,
            "candidate_pools": pools,
            "survival": survival,
        }
        rows.append(row)
        save("predictions.json", rows)
        if index % 5 == 0:
            print(
                f"global_beam {index}/{len(cases)} predicted={leaf_id} latency_ms={latency_ms:.1f}",
                flush=True,
            )

    metrics = summarize(rows, leaves, "global_beam")
    strict = [row for row in rows if not row["needs_clarification"]]
    metrics["candidate_survival"] = {
        key: {
            "count": sum(row["survival"][key] for row in strict),
            "rate": sum(row["survival"][key] for row in strict) / len(strict),
        }
        for key in ("major_global_top2", "middle_global_top2", "leaf_final6")
    }
    summary = {
        "status": "complete",
        "model": "kakaocorp/kanana-2-v-decide-3b-2611",
        "model_revision": "1489062def4145ae3a0a9261fbac4f4d88a8c6d1",
        "method": (
            "Three calls: major 3 -> global top-2; six full middle paths -> global top-2; "
            "six full leaf paths -> final top-1."
        ),
        "scope": "Synthetic author-defined diagnostic; no independent human review; not a production benchmark.",
        "fixture_sha256": {
            "taxonomy.json": sha256(taxonomy_path),
            "cases.json": sha256(cases_path),
        },
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
    shutil.copy2(Path(__file__), out / "executed-run_kanana_global_beam.py")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
