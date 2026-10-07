from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "hierarchical-prefix-2026-10-07"


class PublishedHierarchicalResultTests(unittest.TestCase):
    def test_published_runs_are_complete_and_use_predicted_prefix_method(self):
        for model in ("laya", "kobert"):
            summary = json.loads((RESULTS / model / "test" / "summary.json").read_text("utf-8"))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["methods"]["cascade_prefix"]["strict"]["n"], 81)

    def test_compact_report_records_data_separation_and_prefix_protocol(self):
        report = json.loads((RESULTS / "comparison.json").read_text("utf-8"))
        self.assertEqual(report["training_data_separation"]["exact_text_overlap"], 0)
        self.assertIn("Gold parent", report["prefix_protocol"]["training"])
        self.assertIn("Predicted parent", report["prefix_protocol"]["inference"])
        self.assertEqual(report["main_test_exact_path_accuracy_percent"]["laya"]["flat27"], 34.57)
        self.assertEqual(report["main_test_exact_path_accuracy_percent"]["kobert_mask"]["cascade_prefix"], 16.05)

    def test_unseen_results_cover_all_three_novelty_levels(self):
        for model in ("laya", "kobert"):
            summary = json.loads((RESULTS / model / "unseen" / "summary.json").read_text("utf-8"))
            novelty = summary["methods"]["cascade_prefix"]["by_novelty"]
            self.assertEqual(set(novelty), {"new_leaf", "new_middle", "new_major"})
            self.assertTrue(all(item["n"] == 9 for item in novelty.values()))


if __name__ == "__main__":
    unittest.main()
