"""Check mathematical edge cases and recompute all published sentiment scores."""
import json, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
from metrics import metrics, wilson

class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        rows=[dict(expected=x,predicted=x,confidence=1.,probabilities={x:1.,y:0.},latency_ms=10.) for x,y in [('A','B'),('B','A')]]
        m=metrics(rows)
        self.assertEqual(m['accuracy'],1.)
        self.assertEqual(m['macro_f1'],1.)
        self.assertEqual(m['ece10_top_label'],0.)
        self.assertEqual(m['brier_sum'],0.)
        self.assertEqual(m['coverage_at_0.9'],1.)
    def test_confident_errors_and_empty_coverage(self):
        rows=[dict(expected='A',predicted='B',confidence=.8,probabilities={'A':.2,'B':.8},latency_ms=1.)]
        m=metrics(rows)
        self.assertAlmostEqual(m['ece10_top_label'],.8)
        self.assertAlmostEqual(m['brier_sum'],1.28)
        self.assertIsNone(m['accuracy_at_0.9'])
    def test_wilson_endpoints(self):
        lo,hi=wilson(0,200);self.assertAlmostEqual(lo,0.);self.assertGreater(hi,0.)
        lo,hi=wilson(200,200);self.assertLess(lo,1.);self.assertAlmostEqual(hi,1.)
    def test_recompute_published_scores(self):
        folder=ROOT/'results/pilot-2026-10-05'
        summary=json.loads((folder/'summary.json').read_text())
        for stage,filename in [('tfidf_logreg','baseline')]+[(s,s) for s in ['base_ko','base_en','finetuned_raw','finetuned_calibrated']]:
            with self.subTest(stage=stage):
                actual=metrics(json.loads((folder/(filename+'_predictions.json')).read_text()))
                for key in ['accuracy','macro_f1','ece10_top_label','brier_sum','latency_ms_p50','latency_ms_p95']:
                    self.assertAlmostEqual(actual[key],summary['stages'][stage][key],places=10)

if __name__=='__main__':unittest.main()
