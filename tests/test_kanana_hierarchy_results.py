import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "kanana-hierarchy-2026-11-08"


class KananaHierarchyPublishedResultsTest(unittest.TestCase):
    def test_comparison_counts_and_candidate_survival(self):
        comparison = json.loads((RESULTS / "comparison.json").read_text("utf-8"))
        methods = comparison["methods"]

        self.assertEqual(comparison["flat27"], {"correct": 75, "n": 81})
        self.assertEqual(methods["basic_greedy_cascade"]["strict"]["correct_paths"], 73)
        self.assertEqual(methods["basic_top2_branch"]["strict"]["correct_paths"], 75)
        self.assertEqual(methods["enriched_greedy_cascade"]["strict"]["correct_paths"], 72)
        self.assertEqual(methods["enriched_top2_branch"]["strict"]["correct_paths"], 74)
        self.assertEqual(methods["basic_top2_final_rerank"]["strict"]["correct_paths"], 79)
        self.assertEqual(methods["enriched_top2_final_rerank"]["strict"]["correct_paths"], 79)
        self.assertEqual(methods["basic_top2_final_rerank"]["model_calls_mean"], 8)
        self.assertEqual(
            methods["basic_top2_branch"]["candidate_survival"]["final8"]["count"],
            80,
        )
        self.assertEqual(
            comparison["top2_failure_decomposition"]["basic"],
            {
                "gold_dropped_before_final8": 1,
                "gold_survived_final8": 80,
                "correct_given_gold_survived": 75,
                "wrong_score_selection_given_gold_survived": 5,
            },
        )
        self.assertEqual(
            comparison["final_rerank_change"]["basic"],
            {
                "both_correct": 74,
                "candidate_only": 5,
                "reference_only": 1,
                "both_wrong": 1,
                "gold_survived_final8": 80,
                "correct_given_gold_survived": 79,
                "wrong_rerank_selection_given_gold_survived": 1,
            },
        )

    def test_prediction_trace_shape(self):
        for profile in ("basic", "enriched"):
            rows = json.loads(
                (RESULTS / f"{profile}-top2-predictions.json").read_text("utf-8")
            )
            self.assertEqual(len(rows), 90)
            self.assertEqual(sum(not row["needs_clarification"] for row in rows), 81)
            for row in rows:
                self.assertEqual(len(row["trace"]), 7)
                self.assertEqual(len(row["final_branches"]), 8)
                self.assertEqual(len({branch["leaf_id"] for branch in row["final_branches"]}), 8)

    def test_final_rerank_trace_shape(self):
        for profile in ("basic", "enriched"):
            rows = json.loads(
                (RESULTS / f"{profile}-final-rerank-predictions.json").read_text("utf-8")
            )
            self.assertEqual(len(rows), 90)
            for row in rows:
                self.assertEqual(len(row["trace"]), 8)
                self.assertEqual(row["trace"][-1]["stage"], "final8_direct_rerank")
                self.assertEqual(row["trace"][-1]["candidate_count"], 8)


if __name__ == "__main__":
    unittest.main()
