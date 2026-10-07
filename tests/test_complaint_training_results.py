from pathlib import Path
import hashlib
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "complaint-training-2026-10-07"


class ComplaintTrainingResultTests(unittest.TestCase):
    def read(self, *parts):
        return json.loads((RESULTS.joinpath(*parts)).read_text(encoding="utf-8"))

    def test_laya_training_and_holdout_results(self):
        train = self.read("laya", "train", "summary.json")
        test = self.read("laya", "test", "summary.json")
        self.assertEqual(train["status"], "complete")
        self.assertEqual(train["selected_epoch"], 5)
        self.assertEqual(len(train["epoch_history"]), 5)
        self.assertEqual(test["methods"]["flat27"]["strict"]["correct_paths"], 26)
        self.assertEqual(test["methods"]["cascade"]["strict"]["correct_paths"], 19)
        executed = RESULTS / "laya" / "train" / "executed-train_complaints.py"
        self.assertEqual(hashlib.sha256(executed.read_bytes()).hexdigest(), train["script_sha256"])

    def test_kobert_used_real_mask_markers(self):
        train = self.read("kobert", "train", "summary.json")
        test = self.read("kobert", "test", "summary.json")
        self.assertEqual(train["status"], "complete")
        self.assertEqual(train["settings"]["head_layers"], 0)
        self.assertEqual(train["selected_epoch"], 1)
        self.assertEqual(test["methods"]["flat27"]["strict"]["correct_paths"], 4)
        self.assertEqual(test["methods"]["cascade"]["strict"]["correct_paths"], 5)
        rows = self.read("kobert", "test", "flat27_predictions.json")
        for row in rows:
            for trace in row["trace"]:
                self.assertEqual(trace["candidate_count"], len(trace["marker_token_ids"]))
                self.assertTrue(trace["marker_token_ids"])
                self.assertTrue(all(value == trace["mask_token_id"] == 4 for value in trace["marker_token_ids"]))

    def test_paired_comparison_matches_predictions(self):
        comparison = self.read("comparison.json")
        laya = comparison["laya_domain_vs_base"]
        self.assertEqual((laya["reference_correct"], laya["candidate_correct"]), (20, 26))
        self.assertAlmostEqual(laya["accuracy_delta"], 6 / 81)
        self.assertEqual(laya["discordant"], {"reference_only": 0, "candidate_only": 6})


if __name__ == "__main__":
    unittest.main()
