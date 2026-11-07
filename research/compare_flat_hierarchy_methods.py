import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stage_metrics(predictions, leaf_meta, prediction_fn):
    strict = [row for row in predictions if not row["needs_clarification"]]
    major = middle = exact = 0
    for row in strict:
        gold = leaf_meta[row["expected_leaf"]]
        predicted_leaf = prediction_fn(row)
        predicted = leaf_meta[predicted_leaf]
        major += predicted["major_id"] == gold["major_id"]
        middle += predicted["middle_id"] == gold["middle_id"]
        exact += predicted_leaf == row["expected_leaf"]
    n = len(strict)
    return {
        "n": n,
        "major_correct": major,
        "major_accuracy": major / n,
        "middle_path_correct": middle,
        "middle_path_accuracy": middle / n,
        "exact_path_correct": exact,
        "exact_path_accuracy": exact / n,
    }


def measured_method(model, method, metrics, note):
    return {
        "model": model,
        "method": method,
        **metrics,
        "note": note,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/method-comparison-2026-11-08/summary.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out

    taxonomy = load(root / "datasets/complaints/taxonomy.json")
    leaf_meta = {row["id"]: row for row in taxonomy["leaves"]}
    kanana = load(root / "results/kanana-2-v-decide-2026-11-08/predictions.json")
    embedding = load(
        root
        / "results/embeddinggemma2-retrieval-2026-10-07/retrieval-main27/predictions.json"
    )
    hierarchy_root = root / "results/hierarchical-prefix-2026-10-07"
    laya = load(hierarchy_root / "laya/test/summary.json")
    kobert = load(hierarchy_root / "kobert/test/summary.json")
    beam = load(root / "results/beam12-2026-10-07/laya/beam_analysis.json")

    rows = [
        measured_method(
            "Kanana-2 V-Decide",
            "flat27 zero-shot",
            stage_metrics(kanana, leaf_meta, lambda row: row["predicted_leaf"]),
            "27개 전체 경로와 설명을 한 번에 비교",
        ),
        measured_method(
            "EmbeddingGemma 2 256d",
            "retrieval top-1",
            stage_metrics(
                embedding,
                leaf_meta,
                lambda row: row["rankings"]["256"]["top"][0]["leaf_id"],
            ),
            "27개 설명 문서 중 cosine top-1 검색",
        ),
    ]

    for model_name, summary in (("LAYA", laya), ("KoBERT [MASK]", kobert)):
        for method_name in ("flat27", "cascade", "cascade_prefix"):
            strict = summary["methods"][method_name]["strict"]
            rows.append(
                measured_method(
                    model_name,
                    method_name,
                    {
                        "n": strict["n"],
                        "major_correct": round(strict["major_accuracy"] * strict["n"]),
                        "major_accuracy": strict["major_accuracy"],
                        "middle_path_correct": round(
                            strict["middle_path_accuracy"] * strict["n"]
                        ),
                        "middle_path_accuracy": strict["middle_path_accuracy"],
                        "exact_path_correct": strict["correct_paths"],
                        "exact_path_accuracy": strict["exact_path_accuracy"],
                    },
                    "cascade_prefix는 예측한 상위 경로를 다음 단계 입력 prefix로 전달",
                )
            )

    report = {
        "scope": "Existing-run comparison on the same author-created synthetic fixture; not a newly controlled benchmark.",
        "fixture_sha256": laya["fixture_sha256"],
        "dataset": {
            "taxonomy": "3 major × 3 middle per major × 3 leaf per middle = 27 leaf labels",
            "major_labels": [row["name"] for row in taxonomy["majors"]],
            "strict_cases": 81,
            "strict_slices": {"explicit": 27, "implicit": 27, "mixed_context": 27},
            "under_specified_cases_excluded_from_accuracy": 9,
            "provenance": taxonomy["notes"],
        },
        "methods": rows,
        "beam12_posthoc": {
            "candidate_survival": beam["beam_candidate_survival"],
            "full_path_rerank": beam["beam12_full_path_rerank"],
            "flat27_scores_restricted_to_beam12": beam[
                "flat27_scores_restricted_to_beam12"
            ],
        },
        "comparability_limits": [
            "문항과 taxonomy는 같지만 model size, training recipe, thread 수, cache와 framework는 같지 않다.",
            "Kanana와 EmbeddingGemma에는 동일 fixture의 cascade 실행이 없어 flat 결과만 비교했다.",
            "합성 문장과 label 설명을 같은 작성자가 만들어 실제 민원보다 의미 경계가 선명할 수 있다.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
