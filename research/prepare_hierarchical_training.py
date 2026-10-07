"""Expand each complaint example into flat, major, middle, and leaf tasks."""
from pathlib import Path
import hashlib
import json
import random

from complaint_schema import build_questions, question


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "complaints" / "training"
DEST = ROOT / "datasets" / "complaints" / "hierarchical-training"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def make_row(source, stage, criteria, expected):
    return {
        "id": f"{source['id']}-{stage}",
        "state": source["state"],
        "questions": question(criteria),
        "gold": {"route": {"probabilities": {key: float(key == expected) for key in criteria}}},
        "metadata": {
            **source["metadata"],
            "source_id": source["id"],
            "stage": stage,
            "candidate_count": len(criteria),
            "hierarchical_supervision": True,
        },
    }


def expand(rows, leaves, questions):
    expanded = []
    for source in rows:
        leaf_id = max(source["gold"]["route"]["probabilities"], key=source["gold"]["route"]["probabilities"].get)
        leaf = leaves[leaf_id]
        major_id, middle_id = leaf["major_id"], leaf["middle_id"]
        expanded.extend([
            make_row(source, "flat27", questions["flat"], leaf_id),
            make_row(source, "major3", questions["top"], major_id),
            make_row(source, "middle3", questions["middle"][major_id], middle_id),
            make_row(source, "leaf3", questions["bottom"][middle_id], leaf_id),
        ])
    return expanded


def main():
    taxonomy = json.loads((ROOT / "datasets" / "complaints" / "taxonomy.json").read_text("utf-8"))
    leaves, _, questions = build_questions(taxonomy)
    DEST.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, seed in [("train", 20261008), ("validation", 20261009)]:
        source_rows = read_jsonl(SOURCE / f"{split}.jsonl")
        rows = expand(source_rows, leaves, questions)
        random.Random(seed).shuffle(rows)
        assert len(rows) == len(source_rows) * 4
        assert {r["metadata"]["stage"] for r in rows} == {"flat27", "major3", "middle3", "leaf3"}
        assert all(sum(v > 0 for v in r["gold"]["route"]["probabilities"].values()) == 1 for r in rows)
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path = DEST / f"{split}.jsonl"
        path.write_text(payload, encoding="utf-8", newline="\n")
        outputs[split] = {
            "source_rows": len(source_rows),
            "expanded_rows": len(rows),
            "rows_per_source": 4,
            "sha256_utf8_lf": hashlib.sha256(payload.encode()).hexdigest(),
        }
    manifest = {
        "generator": "research/prepare_hierarchical_training.py",
        "source": "datasets/complaints/training",
        "tasks": {
            "flat27": "27 leaf names plus natural-language descriptions",
            "major3": "3 major names plus their middle-category lists",
            "middle3": "3 middle names plus their leaf-category lists under the gold major",
            "leaf3": "3 leaf names plus natural-language descriptions under the gold middle",
        },
        "teacher_forcing": "Middle and leaf training use the gold parent path; inference uses the predicted parent path.",
        "outputs": outputs,
        "limitations": [
            "No new unique complaint text; four task views are derived from each source example.",
            "No independent human review.",
            "Gold-parent teacher forcing does not expose the model to recovery from a wrong predicted parent.",
        ],
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
