import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "kanana-2-v-decide-2026-11-08"


class KananaPublishedResultsTest(unittest.TestCase):
    def test_published_summary_and_fixture_provenance(self):
        summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["counts"], {"candidates": 27, "cases": 90, "strict": 81})
        self.assertEqual(summary["metrics"]["strict"]["correct"], 75)
        self.assertAlmostEqual(summary["metrics"]["strict"]["accuracy"], 75 / 81)
        self.assertEqual(
            summary["fixture_sha256"]["cases.json"],
            "ab63eb300e010c0602123f60a8b5428dfa2b5db1eb2cfb68aeeb9e6ddfd4cfb2",
        )

    def test_same_data_comparisons(self):
        report = json.loads((RESULTS / "comparison.json").read_text(encoding="utf-8"))
        comparisons = report["comparisons"]

        self.assertEqual(report["n"], 81)
        self.assertEqual(comparisons["LAYA base flat27"]["baseline_correct"], 20)
        self.assertEqual(comparisons["LAYA complaint-trained flat27"]["baseline_correct"], 26)
        self.assertEqual(comparisons["EmbeddingGemma 2 256d top1"]["baseline_correct"], 58)
        self.assertEqual(comparisons["EmbeddingGemma 2 256d top1"]["kanana_correct"], 75)


if __name__ == "__main__":
    unittest.main()
