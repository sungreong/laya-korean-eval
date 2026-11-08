"""Publish and compare Kanana flat, cascade, and top-2 hierarchy runs."""
from pathlib import Path
import argparse
import json
import shutil


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text("utf-8"))


def strict_map(rows):
    return {row["id"]: row for row in rows if not row["needs_clarification"]}


def paired(reference, candidate):
    counts = {"both_correct": 0, "candidate_only": 0, "reference_only": 0, "both_wrong": 0}
    for case_id, ref in reference.items():
        cand = candidate[case_id]
        ref_ok = ref["predicted_leaf"] == ref["expected_leaf"]
        cand_ok = cand["predicted_leaf"] == cand["expected_leaf"]
        if ref_ok and cand_ok:
            counts["both_correct"] += 1
        elif cand_ok:
            counts["candidate_only"] += 1
        elif ref_ok:
            counts["reference_only"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def compact_errors(rows):
    result = []
    for row in rows.values():
        if row["predicted_leaf"] == row["expected_leaf"]:
            continue
        item = {
            "id": row["id"],
            "slice": row["slice"],
            "summary": row["summary"],
            "expected_leaf": row["expected_leaf"],
            "expected_path": row["expected_path"],
            "predicted_leaf": row["predicted_leaf"],
            "predicted_path": row["predicted_path"],
            "latency_ms": row["latency_ms"],
        }
        if "survival" in row:
            item["survival"] = row["survival"]
            item["top_branches"] = row["final_branches"][:3]
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat", default="evaluation/kanana-2-v-decide-flat27-cpu")
    parser.add_argument("--basic", default="evaluation/kanana-2-v-decide-hierarchy-basic-cpu")
    parser.add_argument("--enriched", default="evaluation/kanana-2-v-decide-hierarchy-enriched-cpu")
    parser.add_argument("--basic-rerank", default="evaluation/kanana-2-v-decide-final-rerank-basic-cpu")
    parser.add_argument("--enriched-rerank", default="evaluation/kanana-2-v-decide-final-rerank-enriched-cpu")
    parser.add_argument("--out", default="results/kanana-hierarchy-2026-11-08")
    args = parser.parse_args()

    flat_dir = ROOT / args.flat
    basic_dir = ROOT / args.basic
    enriched_dir = ROOT / args.enriched
    basic_rerank_dir = ROOT / args.basic_rerank
    enriched_rerank_dir = ROOT / args.enriched_rerank
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    flat = strict_map(load(flat_dir / "predictions.json"))
    runs = {}
    for profile, directory in (("basic", basic_dir), ("enriched", enriched_dir)):
        summary = load(directory / "summary.json")
        if summary["status"] != "complete":
            raise RuntimeError(f"Incomplete run: {directory}")
        for method in ("greedy_cascade", "top2_branch"):
            rows = strict_map(load(directory / f"{method}_predictions.json"))
            name = f"{profile}_{method}"
            runs[name] = {
                "rows": rows,
                "metrics": summary["methods"][method],
                "paired_vs_flat": paired(flat, rows),
            }
    for profile, directory in (
        ("basic", basic_rerank_dir),
        ("enriched", enriched_rerank_dir),
    ):
        summary = load(directory / "summary.json")
        if summary["status"] != "complete":
            raise RuntimeError(f"Incomplete run: {directory}")
        rows = strict_map(load(directory / "predictions.json"))
        name = f"{profile}_top2_final_rerank"
        runs[name] = {
            "rows": rows,
            "metrics": summary["metrics"],
            "paired_vs_flat": paired(flat, rows),
        }

    comparison = {
        "scope": "Same 81 strict synthetic complaint cases; zero-shot; CPU Docker; no independent review.",
        "flat27": {
            "correct": sum(row["predicted_leaf"] == row["expected_leaf"] for row in flat.values()),
            "n": len(flat),
        },
        "methods": {
            name: {
                "strict": run["metrics"]["strict"],
                "latency_ms_p50": run["metrics"]["latency_ms_p50"],
                "latency_ms_p95": run["metrics"]["latency_ms_p95"],
                "model_calls_mean": run["metrics"]["model_calls_mean"],
                "candidate_survival": run["metrics"].get("candidate_survival"),
                "paired_vs_flat": run["paired_vs_flat"],
            }
            for name, run in runs.items()
        },
        "top2_failure_decomposition": {},
        "final_rerank_change": {},
        "description_change": {},
        "limits": [
            "The taxonomy has 27 leaves, not 1,000; this tests hierarchy mechanics, not 1,000-label scale.",
            "Stage softmax probabilities from different candidate sets are summed in log space without a separate calibration split.",
            "The enriched descriptions were authored after observing the task design, but not by copying test-case answers.",
        ],
    }

    for profile in ("basic", "enriched"):
        run = runs[f"{profile}_top2_branch"]
        rows = run["rows"]
        survived = [row for row in rows.values() if row["survival"]["final8"]]
        comparison["top2_failure_decomposition"][profile] = {
            "gold_dropped_before_final8": len(rows) - len(survived),
            "gold_survived_final8": len(survived),
            "correct_given_gold_survived": sum(
                row["predicted_leaf"] == row["expected_leaf"] for row in survived
            ),
            "wrong_score_selection_given_gold_survived": sum(
                row["predicted_leaf"] != row["expected_leaf"] for row in survived
            ),
        }
        reranked = runs[f"{profile}_top2_final_rerank"]["rows"]
        comparison["final_rerank_change"][profile] = {
            **paired(rows, reranked),
            "gold_survived_final8": len(survived),
            "correct_given_gold_survived": sum(
                row["predicted_leaf"] == row["expected_leaf"]
                for row in reranked.values()
                if row["survival"]["final8"]
            ),
            "wrong_rerank_selection_given_gold_survived": sum(
                row["predicted_leaf"] != row["expected_leaf"]
                for row in reranked.values()
                if row["survival"]["final8"]
            ),
        }

    for method in ("greedy_cascade", "top2_branch"):
        basic = runs[f"basic_{method}"]["rows"]
        enriched = runs[f"enriched_{method}"]["rows"]
        comparison["description_change"][method] = paired(basic, enriched)

    (out / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), "utf-8"
    )
    for name, run in runs.items():
        (out / f"errors-{name}.json").write_text(
            json.dumps(compact_errors(run["rows"]), ensure_ascii=False, indent=2), "utf-8"
        )

    copies = {
        basic_dir / "summary.json": out / "basic-summary.json",
        basic_dir / "greedy_cascade_predictions.json": out / "basic-greedy-predictions.json",
        basic_dir / "top2_branch_predictions.json": out / "basic-top2-predictions.json",
        enriched_dir / "summary.json": out / "enriched-summary.json",
        enriched_dir / "greedy_cascade_predictions.json": out / "enriched-greedy-predictions.json",
        enriched_dir / "top2_branch_predictions.json": out / "enriched-top2-predictions.json",
        basic_rerank_dir / "summary.json": out / "basic-final-rerank-summary.json",
        basic_rerank_dir / "predictions.json": out / "basic-final-rerank-predictions.json",
        enriched_rerank_dir / "summary.json": out / "enriched-final-rerank-summary.json",
        enriched_rerank_dir / "predictions.json": out / "enriched-final-rerank-predictions.json",
        basic_dir / "executed-run_kanana_hierarchy.py": out / "executed-run_kanana_hierarchy.py",
        basic_rerank_dir / "executed-run_kanana_final_rerank.py": out / "executed-run_kanana_final_rerank.py",
    }
    for source, destination in copies.items():
        shutil.copy2(source, destination)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
