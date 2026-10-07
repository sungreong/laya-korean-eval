from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "complaints" / "hierarchical-training"


class HierarchicalTrainingDataTests(unittest.TestCase):
    def read(self, name):
        return [json.loads(line) for line in (DATA / name).read_text("utf-8").splitlines() if line.strip()]

    def test_four_task_views_per_source(self):
        for name, sources in [("train.jsonl", 729), ("validation.jsonl", 243)]:
            rows = self.read(name)
            self.assertEqual(len(rows), sources * 4)
            grouped = {}
            for row in rows:
                grouped.setdefault(row["metadata"]["source_id"], set()).add(row["metadata"]["stage"])
            self.assertEqual(len(grouped), sources)
            self.assertTrue(all(stages == {"flat27", "major3", "middle3", "leaf3"} for stages in grouped.values()))

    def test_candidate_counts_and_descriptions(self):
        expected_counts = {"flat27": 27, "major3": 3, "middle3": 3, "leaf3": 3}
        for row in self.read("train.jsonl"):
            stage = row["metadata"]["stage"]
            criteria = row["questions"]["route"]["criteria"]
            self.assertEqual(len(criteria), expected_counts[stage])
            self.assertTrue(all(":" in description for description in criteria.values()))
            self.assertEqual(sum(value > 0 for value in row["gold"]["route"]["probabilities"].values()), 1)

    def test_balanced_training_subset_rule(self):
        rows = self.read("train.jsonl")
        stage_for_style = {0: "major3", 1: "middle3", 2: "leaf3"}
        selected = [row for row in rows
                    if row["metadata"]["stage"] == stage_for_style[(row["metadata"]["style_index"] - 1) % 3]
                    or (row["metadata"]["stage"] == "flat27" and (row["metadata"]["style_index"] - 1) % 3 == 0)]
        self.assertEqual(len(selected), 972)
        counts = {stage: sum(row["metadata"]["stage"] == stage for row in selected)
                  for stage in ["flat27", "major3", "middle3", "leaf3"]}
        self.assertEqual(counts, {"flat27": 243, "major3": 243, "middle3": 243, "leaf3": 243})
        self.assertEqual(len({row["state"] for row in selected}), 729)

    def test_prefix_training_rows_include_gold_parent_names(self):
        prefix_dir = ROOT / "datasets" / "complaints" / "hierarchical-prefix-training"
        rows = [json.loads(line) for line in (prefix_dir / "train.jsonl").read_text("utf-8").splitlines() if line.strip()]
        for row in rows:
            stage = row["metadata"]["stage"]
            if stage == "middle3":
                self.assertIn("이전 단계 선택 대분류:", row["state"])
            elif stage == "leaf3":
                self.assertIn("이전 단계 선택 대분류:", row["state"])
                self.assertIn("이전 단계 선택 중분류:", row["state"])


if __name__ == "__main__":
    unittest.main()
