import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research'))
from metrics import metrics


class FollowupResultTests(unittest.TestCase):
    def test_scaling_selection_and_published_metrics(self):
        folder = ROOT / 'results/scaling-2026-10-07'
        summary = json.loads((folder / 'summary.json').read_text('utf-8'))
        self.assertEqual(summary['status'], 'complete')
        chosen = max(summary['epoch_history'], key=lambda x: (x['validation_accuracy'], -x['validation_nll']))
        self.assertEqual(summary['selected_epoch'], chosen['epoch'])
        self.assertEqual(hashlib.sha256((folder / 'executed-run_scaling.py').read_bytes()).hexdigest(), summary['script_sha256'])
        for name, expected in summary['stages'].items():
            rows = json.loads((folder / (name + '_predictions.json')).read_text('utf-8'))
            actual = metrics(rows)
            for key in ['accuracy', 'macro_f1', 'brier_sum']:
                self.assertAlmostEqual(actual[key], expected[key], places=10)

    def test_hierarchy_reports_match_paths_and_original_fixture(self):
        fixtures = ROOT / 'datasets/complaints'
        cases = {x['id']: x for x in json.loads((fixtures / 'cases.json').read_text('utf-8'))}
        leaves = {x['id']: x for x in json.loads((fixtures / 'taxonomy.json').read_text('utf-8'))['leaves']}
        for model in ['base', 'trained']:
            folder = ROOT / 'results/complaints-2026-10-07' / model
            summary = json.loads((folder / 'summary.json').read_text('utf-8'))
            self.assertEqual(summary['status'], 'complete')
            self.assertEqual(hashlib.sha256((folder / 'executed-run_complaints.py').read_bytes()).hexdigest(), summary['script_sha256'])
            for name, digest in summary['fixture_sha256'].items():
                self.assertEqual(hashlib.sha256((fixtures / name).read_bytes()).hexdigest(), digest)
            for method, expected in summary['methods'].items():
                rows = json.loads((folder / (method + '_predictions.json')).read_text('utf-8'))
                self.assertEqual(len(rows), 90)
                for row in rows:
                    self.assertEqual(row['summary'], cases[row['id']]['summary'])
                    self.assertEqual(row['expected_leaf'], cases[row['id']]['expected_leaf'])
                    self.assertEqual(row['predicted_path'], leaves[row['predicted_leaf']]['path'])
                strict = [r for r in rows if not r['needs_clarification']]
                self.assertEqual(len(strict), 81)
                for depth, name in [(1, 'major_accuracy'), (2, 'middle_path_accuracy'), (3, 'exact_path_accuracy')]:
                    actual = sum(r['predicted_path'][:depth] == r['expected_path'][:depth] for r in strict) / 81
                    self.assertAlmostEqual(actual, expected['strict'][name])
                ambiguous = [r for r in rows if r['needs_clarification']]
                self.assertEqual(len(ambiguous), 9)
                self.assertEqual(sum(r['predicted_leaf'] in r['acceptable_leaves'] for r in ambiguous), expected['under_specified']['acceptable_path_hits'])
