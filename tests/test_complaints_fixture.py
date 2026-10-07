import json
from pathlib import Path
import unittest
from collections import Counter

FIXTURE = Path(__file__).resolve().parents[1] / 'datasets/complaints'


class ComplaintFixtureTests(unittest.TestCase):
    def test_tree_is_three_by_three_by_three(self):
        tree = json.loads((FIXTURE / 'taxonomy.json').read_text('utf-8'))
        self.assertEqual(len(tree['majors']), 3)
        leaves = {x['id']: x for x in tree['leaves']}
        self.assertEqual(len(leaves), 27)
        referenced = []
        for major in tree['majors']:
            self.assertEqual(len(major['children']), 3)
            for middle in major['children']:
                self.assertEqual(len(middle['children']), 3)
                for key in middle['children']:
                    self.assertEqual(leaves[key]['major_id'], major['id'])
                    self.assertEqual(leaves[key]['middle_id'], middle['id'])
                    self.assertEqual(leaves[key]['path'][:2], [major['name'], middle['name']])
                    referenced.append(key)
        self.assertEqual(Counter(referenced), Counter(leaves.keys()))

    def test_balanced_difficulty_and_explicit_ambiguity(self):
        cases = json.loads((FIXTURE / 'cases.json').read_text('utf-8'))
        leaves = {x['id'] for x in json.loads((FIXTURE / 'taxonomy.json').read_text('utf-8'))['leaves']}
        self.assertEqual(len(cases), 90)
        self.assertEqual(len({x['id'] for x in cases}), 90)
        self.assertEqual(len({x['summary'] for x in cases}), 90)
        self.assertEqual(Counter(x['channel'] for x in cases), {'콜센터': 45, '인터넷 상담': 45})
        for difficulty in ['explicit', 'implicit', 'mixed_context']:
            subset = [x for x in cases if x['slice'] == difficulty]
            self.assertEqual(Counter(x['expected_leaf'] for x in subset), Counter(leaves))
        for case in cases:
            self.assertTrue(set(case['acceptable_leaves']) <= leaves)
            if case['needs_clarification']:
                self.assertIsNone(case['expected_leaf'])
                self.assertGreater(len(case['acceptable_leaves']), 1)
                self.assertTrue(case['clarification_needed'])
            else:
                self.assertEqual(case['acceptable_leaves'], [case['expected_leaf']])

    def test_training_data_is_balanced_diverse_and_disjoint(self):
        training = FIXTURE / 'training'
        train = [json.loads(x) for x in (training / 'train.jsonl').read_text('utf-8').splitlines()]
        validation = [json.loads(x) for x in (training / 'validation.jsonl').read_text('utf-8').splitlines()]
        heldout = json.loads((FIXTURE / 'cases.json').read_text('utf-8'))
        self.assertEqual((len(train), len(validation)), (729, 243))
        def labels(rows):
            return Counter(max(r['gold']['route']['probabilities'], key=r['gold']['route']['probabilities'].get) for r in rows)
        self.assertEqual(set(labels(train).values()), {27})
        self.assertEqual(set(labels(validation).values()), {9})
        train_text, validation_text = {r['state'] for r in train}, {r['state'] for r in validation}
        self.assertFalse(train_text & validation_text)
        self.assertFalse((train_text | validation_text) & {'민원 요약: ' + r['summary'] for r in heldout})
        self.assertEqual(len(train_text) + len(validation_text), 972)
