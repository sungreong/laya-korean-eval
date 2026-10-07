import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "embeddinggemma2-retrieval-2026-10-07"


def load(*parts):
    return json.loads((RESULTS.joinpath(*parts)).read_text(encoding="utf-8"))


class EmbeddingGemma2PublishedResultsTest(unittest.TestCase):
    def test_main27_retrieval_counts(self):
        summary = load("retrieval-main27", "summary.json")
        metric = summary["metrics"]["256"]
        self.assertEqual(summary["model_revision"], "914f7f89142e33e77833254d9c9b90c3cef7303b")
        self.assertEqual(summary["candidate_count"], 27)
        self.assertEqual(metric["n"], 81)
        self.assertEqual(metric["top1_correct"], 58)
        self.assertEqual(metric["recall_at_k"]["5"], 80 / 81)
        self.assertEqual(metric["recall_at_k"]["8"], 1.0)

    def test_main27_rerank_counts(self):
        summary = load("rerank-main27", "summary.json")
        top3 = summary["metrics"]["3"]
        self.assertEqual(top3["n"], 81)
        self.assertEqual(top3["gold_retrieved"], 74)
        self.assertEqual(top3["correct"], 41)
        self.assertAlmostEqual(top3["accuracy_given_gold_retrieved"], 41 / 74)

    def test_unseen_results(self):
        retrieval = load("retrieval-unseen36", "summary.json")
        rerank = load("rerank-unseen36", "summary.json")
        self.assertEqual(retrieval["candidate_count"], 36)
        self.assertEqual(retrieval["metrics"]["256"]["top1_correct"], 16)
        self.assertEqual(retrieval["metrics"]["256"]["recall_at_k"]["8"], 1.0)
        self.assertEqual(rerank["metrics"]["3"]["correct"], 15)


if __name__ == "__main__":
    unittest.main()
