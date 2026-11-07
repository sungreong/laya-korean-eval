import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "method-comparison-2026-11-08" / "summary.json"


class MethodComparisonResultsTest(unittest.TestCase):
    def test_stage_counts_and_main_ranking(self):
        report = json.loads(SUMMARY.read_text(encoding="utf-8"))
        rows = {(row["model"], row["method"]): row for row in report["methods"]}

        self.assertEqual(report["dataset"]["strict_cases"], 81)
        self.assertEqual(rows[("Kanana-2 V-Decide", "flat27 zero-shot")]["exact_path_correct"], 75)
        self.assertEqual(rows[("EmbeddingGemma 2 256d", "retrieval top-1")]["exact_path_correct"], 58)
        self.assertEqual(rows[("LAYA", "flat27")]["exact_path_correct"], 28)
        self.assertEqual(rows[("LAYA", "cascade_prefix")]["exact_path_correct"], 19)
        self.assertEqual(rows[("KoBERT [MASK]", "cascade_prefix")]["exact_path_correct"], 13)

    def test_beam_recall_is_not_top1_accuracy(self):
        report = json.loads(SUMMARY.read_text(encoding="utf-8"))
        beam = report["beam12_posthoc"]

        self.assertEqual(beam["candidate_survival"]["count"], 62)
        self.assertEqual(beam["full_path_rerank"]["correct"], 7)
        self.assertGreater(
            beam["candidate_survival"]["rate"],
            beam["full_path_rerank"]["accuracy"],
        )


if __name__ == "__main__":
    unittest.main()
