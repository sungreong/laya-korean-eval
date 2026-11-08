"""Evaluate Kanana with greedy predicted-prefix cascade and top-2 branching."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import statistics
import time

import psutil
import torch
from transformers import AutoModel

from complaint_schema import build_questions


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("greedy_cascade", "top2_branch")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * p / 100
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def wilson(correct, total, z=1.959963984540054):
    if total == 0:
        return [0.0, 0.0]
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [centre - margin, centre + margin]


def macro_f1(y_true, y_pred, labels):
    scores = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(scores) / len(scores)


def top_keys(probabilities, k):
    return sorted(probabilities, key=probabilities.get, reverse=True)[:k]


def build_indexes(taxonomy):
    leaves, majors, questions = build_questions(taxonomy)
    middle_to_major = {}
    middle_names = {}
    for major_id, major in majors.items():
        for middle in major["children"]:
            middle_to_major[middle["id"]] = major_id
            middle_names[middle["id"]] = middle["name"]
    return leaves, majors, questions, middle_to_major, middle_names


def apply_description_profile(questions, leaves, majors, profile):
    if profile is None:
        return questions
    for major_id, description in profile["major"].items():
        questions["top"][major_id] = f"{majors[major_id]['name']}: {description}"
    for major_id, major in majors.items():
        for middle in major["children"]:
            middle_id = middle["id"]
            questions["middle"][major_id][middle_id] = (
                f"{middle['name']}: {profile['middle'][middle_id]}"
            )
    suffix = profile["leaf_suffix"]
    for leaf_id, leaf in leaves.items():
        questions["bottom"][leaf["middle_id"]][leaf_id] = (
            f"{leaf['path'][2]}: {leaf['description']}. {suffix}"
        )
    return questions


def summarize(rows, leaves, method):
    strict = [row for row in rows if not row["needs_clarification"]]
    ambiguous = [row for row in rows if row["needs_clarification"]]

    def score(items):
        y_true = [row["expected_leaf"] for row in items]
        y_pred = [row["predicted_leaf"] for row in items]
        correct = sum(a == b for a, b in zip(y_true, y_pred))
        prefix = [
            sum(row["predicted_path"][:depth] == row["expected_path"][:depth] for row in items)
            for depth in (1, 2, 3)
        ]
        first_error = {"major": 0, "middle": 0, "leaf": 0}
        for row in items:
            for depth, name in enumerate(first_error):
                if row["predicted_path"][depth] != row["expected_path"][depth]:
                    first_error[name] += 1
                    break
        return {
            "n": len(items),
            "correct_paths": correct,
            "major_correct": prefix[0],
            "middle_path_correct": prefix[1],
            "exact_path_correct": prefix[2],
            "major_accuracy": prefix[0] / len(items),
            "middle_path_accuracy": prefix[1] / len(items),
            "exact_path_accuracy": prefix[2] / len(items),
            "leaf_macro_f1": macro_f1(y_true, y_pred, list(leaves)),
            "wilson95_exact": wilson(correct, len(items)),
            "first_error_level": first_error,
        }

    result = {
        "strict": score(strict),
        "by_slice": {
            name: score([row for row in strict if row["slice"] == name])
            for name in sorted({row["slice"] for row in strict})
        },
        "under_specified": {
            "n": len(ambiguous),
            "acceptable_hits": sum(row["predicted_leaf"] in row["acceptable_leaves"] for row in ambiguous),
            "note": "Forced-choice diagnostic; excluded from single-answer accuracy.",
        },
        "latency_ms_p50": percentile([row["latency_ms"] for row in rows], 50),
        "latency_ms_p95": percentile([row["latency_ms"] for row in rows], 95),
        "model_calls_mean": statistics.mean(len(row["trace"]) for row in rows),
        "prediction_distribution": Counter(row["predicted_leaf"] for row in rows),
    }
    if method == "top2_branch":
        result["candidate_survival"] = {
            key: {
                "count": sum(row["survival"][key] for row in strict),
                "rate": sum(row["survival"][key] for row in strict) / len(strict),
            }
            for key in ("major_top2", "middle_path_top2", "leaf_path_top2", "final8")
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="evaluation/models/kanana-2-v-decide-3b-2611")
    parser.add_argument("--fixture", default="datasets/complaints")
    parser.add_argument("--out", default="evaluation/kanana-2-v-decide-hierarchy-cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--description-profile")
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
    description_profile_path = ROOT / args.description_profile if args.description_profile else None
    description_profile = (
        json.loads(description_profile_path.read_text("utf-8"))
        if description_profile_path
        else None
    )
    questions = apply_description_profile(
        questions, leaves, majors, description_profile
    )

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
    rss_after_load = process.memory_info().rss

    instructions = {
        "major": "민원 요약의 핵심 요청을 기준으로 가장 적합한 대분류를 선택하세요.",
        "middle": "민원 요약과 앞 단계 대분류 context를 기준으로 가장 적합한 중분류를 선택하세요.",
        "leaf": "민원 요약과 앞 단계 대·중분류 context를 기준으로 가장 적합한 소분류를 선택하세요.",
    }

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")

    summary_path = out / "summary.json"
    if summary_path.exists() and args.resume:
        summary = json.loads(summary_path.read_text("utf-8"))
        summary["status"] = "running"
    else:
        summary = {
            "status": "running",
            "model": "kakaocorp/kanana-2-v-decide-3b-2611",
            "model_local_path": args.model,
            "model_revision": "1489062def4145ae3a0a9261fbac4f4d88a8c6d1",
            "fixture": args.fixture,
            "fixture_sha256": {
                "taxonomy.json": sha256(taxonomy_path),
                "cases.json": sha256(cases_path),
            },
            "description_profile": {
                "path": args.description_profile,
                "sha256": sha256(description_profile_path) if description_profile_path else None,
                "name": description_profile["profile"] if description_profile else "taxonomy-basic",
            },
            "scope": "Synthetic author-defined diagnostic; no independent human review; not a production benchmark.",
            "environment": {
                "device": "cpu",
                "dtype": "bfloat16",
                "attention": "sdpa",
                "threads": args.threads,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "container_memory_limit_bytes": (
                    Path("/sys/fs/cgroup/memory.max").read_text().strip()
                    if Path("/sys/fs/cgroup/memory.max").exists()
                    else None
                ),
            },
            "counts": {
                "major": len(majors),
                "middle": len(middle_to_major),
                "leaf": len(leaves),
                "cases": len(cases),
                "strict": sum(not case["needs_clarification"] for case in cases),
            },
            "methods": {},
        }
    summary.update({
        "script_sha256": sha256(Path(__file__)),
        "load_seconds_latest_process": load_seconds,
        "rss_after_load_bytes_latest_process": rss_after_load,
        "scoring": {
            "greedy_cascade": "Predicted top-1 parent is injected as context at the next stage.",
            "top2_branch": "Keep top-2 at all three stages; choose the final leaf with max sum of stage log probabilities.",
            "caveat": "Softmax probabilities from different candidate sets are not calibrated to be directly comparable.",
        },
    })
    save("summary.json", summary)
    (out / "executed-run_kanana_hierarchy.py").write_bytes(Path(__file__).read_bytes())

    def ask(state, criteria, stage_kind, stage_name):
        question = {
            "route": {
                "type": "choice",
                "instructions": instructions[stage_kind],
                "criteria": criteria,
            }
        }
        started = time.perf_counter()
        with torch.inference_mode():
            response = model.system_one(state=state, questions=question)
        latency_ms = (time.perf_counter() - started) * 1000
        answer = response["answers"]["route"]
        if set(answer["probabilities"]) != set(criteria):
            raise RuntimeError(f"Candidate IDs changed at {stage_name}")
        return {
            "stage": stage_name,
            "candidate_count": len(criteria),
            "choice": answer["choice"],
            "probabilities": answer["probabilities"],
            "confidence": answer["confidence"],
            "latency_ms": latency_ms,
        }

    def base_state(case):
        return f"상담 경로: {case['channel']}\n민원 요약: {case['summary']}"

    def greedy_predict(case):
        state = base_state(case)
        started = time.perf_counter()
        major_trace = ask(state, questions["top"], "major", "major3")
        major_id = major_trace["choice"]
        middle_state = state + f"\n이전 단계 선택 대분류: {majors[major_id]['name']}"
        middle_trace = ask(middle_state, questions["middle"][major_id], "middle", "middle3")
        middle_id = middle_trace["choice"]
        leaf_state = middle_state + f"\n이전 단계 선택 중분류: {middle_names[middle_id]}"
        leaf_trace = ask(leaf_state, questions["bottom"][middle_id], "leaf", "leaf3")
        leaf_id = leaf_trace["choice"]
        return leaf_id, [major_trace, middle_trace, leaf_trace], None, None, time.perf_counter() - started

    def top2_predict(case):
        state = base_state(case)
        started = time.perf_counter()
        traces = []
        major_trace = ask(state, questions["top"], "major", "major3_top2")
        traces.append(major_trace)
        major_ids = top_keys(major_trace["probabilities"], 2)
        branches = []
        middle_ids = []
        for major_id in major_ids:
            major_state = state + f"\n현재 가정한 대분류 후보: {majors[major_id]['name']}"
            middle_trace = ask(
                major_state,
                questions["middle"][major_id],
                "middle",
                f"middle3_under_{major_id}_top2",
            )
            traces.append(middle_trace)
            for middle_id in top_keys(middle_trace["probabilities"], 2):
                middle_ids.append(middle_id)
                leaf_state = (
                    major_state
                    + f"\n현재 가정한 중분류 후보: {middle_names[middle_id]}"
                )
                leaf_trace = ask(
                    leaf_state,
                    questions["bottom"][middle_id],
                    "leaf",
                    f"leaf3_under_{middle_id}_top2",
                )
                traces.append(leaf_trace)
                for leaf_id in top_keys(leaf_trace["probabilities"], 2):
                    stage_probabilities = {
                        "major": major_trace["probabilities"][major_id],
                        "middle": middle_trace["probabilities"][middle_id],
                        "leaf": leaf_trace["probabilities"][leaf_id],
                    }
                    score = sum(math.log(max(value, 1e-12)) for value in stage_probabilities.values())
                    branches.append({
                        "major_id": major_id,
                        "middle_id": middle_id,
                        "leaf_id": leaf_id,
                        "stage_probabilities": stage_probabilities,
                        "log_probability_sum": score,
                    })
        branches.sort(key=lambda branch: branch["log_probability_sum"], reverse=True)
        if len(branches) != 8 or len({branch["leaf_id"] for branch in branches}) != 8:
            raise RuntimeError(f"Expected 8 unique final branches, got {len(branches)}")
        final_leaf_ids = [branch["leaf_id"] for branch in branches]
        if case["expected_leaf"] is None:
            survival = {key: None for key in (
                "major_top2", "middle_path_top2", "leaf_path_top2", "final8"
            )}
        else:
            expected_leaf = leaves[case["expected_leaf"]]
            expected_major = expected_leaf["major_id"]
            expected_middle = expected_leaf["middle_id"]
            survival = {
                "major_top2": expected_major in major_ids,
                "middle_path_top2": expected_major in major_ids and expected_middle in middle_ids,
                "leaf_path_top2": case["expected_leaf"] in final_leaf_ids,
                "final8": case["expected_leaf"] in final_leaf_ids,
            }
        return branches[0]["leaf_id"], traces, branches, survival, time.perf_counter() - started

    predictors = {
        "greedy_cascade": greedy_predict,
        "top2_branch": top2_predict,
    }

    # Fill static candidate caches before measured requests.
    warm_case = cases[0]
    for method in args.methods:
        predictors[method](warm_case)
    summary["rss_after_warmup_bytes_latest_process"] = process.memory_info().rss
    save("summary.json", summary)

    for method in args.methods:
        predictions_path = out / f"{method}_predictions.json"
        rows = (
            json.loads(predictions_path.read_text("utf-8"))
            if predictions_path.exists() and args.resume
            else []
        )
        completed_ids = {row["id"] for row in rows}
        for index, case in enumerate(cases, 1):
            if case["id"] in completed_ids:
                continue
            leaf_id, trace, branches, survival, elapsed_seconds = predictors[method](case)
            row = {
                **case,
                "predicted_leaf": leaf_id,
                "predicted_path": leaves[leaf_id]["path"],
                "expected_path": (
                    leaves[case["expected_leaf"]]["path"]
                    if case["expected_leaf"] is not None
                    else None
                ),
                "latency_ms": elapsed_seconds * 1000,
                "trace": trace,
            }
            if branches is not None:
                row["final_branches"] = branches
                row["survival"] = survival
            rows.append(row)
            save(f"{method}_predictions.json", rows)
            if index % 5 == 0:
                print(
                    f"{method} {index}/{len(cases)} predicted={leaf_id} "
                    f"latency_ms={row['latency_ms']:.1f}",
                    flush=True,
                )
        summary["methods"][method] = summarize(rows, leaves, method)
        save("summary.json", summary)
        print(json.dumps({method: summary["methods"][method]}, ensure_ascii=False, indent=2), flush=True)

    if all(method in summary["methods"] for method in args.methods):
        summary["status"] = "complete"
    save("summary.json", summary)
    print("FINISHED", flush=True)


if __name__ == "__main__":
    main()
