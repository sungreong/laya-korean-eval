import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Beam12ResultTests(unittest.TestCase):
    def test_published_beam_result_separates_recall_and_accuracy(self):
        folder = ROOT / "results" / "beam12-2026-10-07" / "laya"
        summary = json.loads((folder / "summary.json").read_text("utf-8"))
        analysis = json.loads((folder / "beam_analysis.json").read_text("utf-8"))
        self.assertEqual(summary["methods"]["beam12"]["strict"]["correct_paths"], 7)
        self.assertEqual(analysis["beam_candidate_survival"]["count"], 62)
        self.assertEqual(analysis["beam12_full_path_rerank"]["correct"], 7)
        self.assertEqual(analysis["flat27_scores_restricted_to_beam12"]["correct"], 26)


if __name__ == "__main__":
    unittest.main()
