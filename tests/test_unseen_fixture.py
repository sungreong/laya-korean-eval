from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UnseenFixtureTests(unittest.TestCase):
    def test_unseen_labels_are_absent_from_training_and_text_is_disjoint(self):
        fixture = ROOT / "datasets" / "complaints" / "unseen"
        taxonomy = json.loads((fixture / "taxonomy.json").read_text("utf-8"))
        cases = json.loads((fixture / "cases.json").read_text("utf-8"))
        new_ids = {leaf["id"] for leaf in taxonomy["leaves"] if leaf.get("novelty")}
        self.assertEqual(new_ids, {f"L{i}" for i in range(28, 37)})
        self.assertEqual({case["expected_leaf"] for case in cases}, new_ids)
        self.assertEqual(len(cases), 27)
        training = "\n".join((ROOT / "datasets" / "complaints" / "training" / name).read_text("utf-8")
                              for name in ["train.jsonl", "validation.jsonl"])
        self.assertTrue(all(case["summary"] not in training for case in cases))

    def test_three_novelty_levels_are_balanced(self):
        cases = json.loads((ROOT / "datasets" / "complaints" / "unseen" / "cases.json").read_text("utf-8"))
        counts = {kind: sum(case["novelty"] == kind for case in cases)
                  for kind in ["new_leaf", "new_middle", "new_major"]}
        self.assertEqual(counts, {"new_leaf": 9, "new_middle": 9, "new_major": 9})


if __name__ == "__main__":
    unittest.main()
