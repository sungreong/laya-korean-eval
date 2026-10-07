"""Separate beam candidate recall from final reranking accuracy."""
from pathlib import Path
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam", required=True)
    parser.add_argument("--flat", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    beam_path = Path(args.beam)
    flat_path = Path(args.flat)
    beam = [row for row in json.loads(beam_path.read_text("utf-8")) if row.get("expected_leaf")]
    flat = {row["id"]: row for row in json.loads(flat_path.read_text("utf-8")) if row.get("expected_leaf")}

    survived = 0
    final_correct = 0
    restricted_correct = 0
    rescued_vs_flat = 0
    lost_vs_flat = 0
    rank_counts = {}
    for row in beam:
        candidates = row["trace"][-1]["answer"]["probabilities"]
        gold = row["expected_leaf"]
        if gold in candidates:
            survived += 1
            ranked = sorted(candidates, key=candidates.get, reverse=True)
            rank = ranked.index(gold) + 1
            rank_counts[str(rank)] = rank_counts.get(str(rank), 0) + 1
        final_correct += row["predicted_leaf"] == gold

        flat_row = flat[row["id"]]
        flat_probs = flat_row["trace"][0]["answer"]["probabilities"]
        restricted = max(candidates, key=lambda leaf_id: flat_probs[leaf_id])
        restricted_correct += restricted == gold
        rescued_vs_flat += restricted == gold and flat_row["predicted_leaf"] != gold
        lost_vs_flat += restricted != gold and flat_row["predicted_leaf"] == gold

    n = len(beam)
    result = {
        "scope": "81 single-answer synthetic complaint cases; post-hoc diagnostics, not a new benchmark.",
        "beam_predictions": str(beam_path).replace("\\", "/"),
        "flat_predictions": str(flat_path).replace("\\", "/"),
        "strict_cases": n,
        "beam_candidate_survival": {"count": survived, "rate": survived / n},
        "beam12_full_path_rerank": {
            "correct": final_correct,
            "accuracy": final_correct / n,
            "accuracy_given_gold_survived": final_correct / survived,
        },
        "flat27_scores_restricted_to_beam12": {
            "correct": restricted_correct,
            "accuracy": restricted_correct / n,
            "rescued_vs_unrestricted_flat27": rescued_vs_flat,
            "lost_vs_unrestricted_flat27": lost_vs_flat,
        },
        "gold_rank_within_final_beam": rank_counts,
        "interpretation": [
            "The beam retained the gold leaf in most cases, but the unseen 12-way full-path reranking format failed.",
            "Restricting the existing flat-27 scores to beam candidates also did not beat unrestricted flat-27.",
            "Top-k candidate recall and top-1 end-to-end accuracy must be reported separately.",
        ],
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
