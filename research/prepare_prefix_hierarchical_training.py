"""Create hierarchy-task rows that explicitly include the gold parent path as a prefix."""
from pathlib import Path
import hashlib
import json
import random

from complaint_schema import build_questions, question


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "complaints" / "training"
DEST = ROOT / "datasets" / "complaints" / "hierarchical-prefix-training"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def make_row(source, stage, criteria, expected, prefix_lines):
    state = source["state"]
    if prefix_lines:
        state += "\n" + "\n".join(prefix_lines)
    return {
        "id": f"{source['id']}-{stage}",
        "state": state,
        "questions": question(criteria),
        "gold": {"route": {"probabilities": {key: float(key == expected) for key in criteria}}},
        "metadata": {
            **source["metadata"],
            "source_id": source["id"],
            "stage": stage,
            "candidate_count": len(criteria),
            "prefix_conditioned": bool(prefix_lines),
        },
    }


def expand(rows, leaves, majors, questions):
    expanded = []
    for source in rows:
        leaf_id = max(source["gold"]["route"]["probabilities"], key=source["gold"]["route"]["probabilities"].get)
        leaf = leaves[leaf_id]
        major_id, middle_id = leaf["major_id"], leaf["middle_id"]
        major_name = majors[major_id]["name"]
        middle_name = next(node["name"] for node in majors[major_id]["children"] if node["id"] == middle_id)
        expanded.extend([
            make_row(source, "flat27", questions["flat"], leaf_id, []),
            make_row(source, "major3", questions["top"], major_id, []),
            make_row(source, "middle3", questions["middle"][major_id], middle_id,
                     [f"이전 단계 선택 대분류: {major_name}"]),
            make_row(source, "leaf3", questions["bottom"][middle_id], leaf_id,
                     [f"이전 단계 선택 대분류: {major_name}", f"이전 단계 선택 중분류: {middle_name}"]),
        ])
    return expanded


def main():
    taxonomy = json.loads((ROOT / "datasets" / "complaints" / "taxonomy.json").read_text("utf-8"))
    leaves, majors, questions = build_questions(taxonomy)
    DEST.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split, seed in [("train", 20261010), ("validation", 20261011)]:
        source_rows = read_jsonl(SOURCE / f"{split}.jsonl")
        rows = expand(source_rows, leaves, majors, questions)
        random.Random(seed).shuffle(rows)
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (DEST / f"{split}.jsonl").write_text(payload, encoding="utf-8", newline="\n")
        outputs[split] = {"source_rows": len(source_rows), "expanded_rows": len(rows),
                          "sha256_utf8_lf": hashlib.sha256(payload.encode()).hexdigest()}
    manifest = {
        "generator": "research/prepare_prefix_hierarchical_training.py",
        "source": "datasets/complaints/training",
        "prefix": {
            "major3": "none",
            "middle3": "gold major name",
            "leaf3": "gold major and middle names",
            "inference": "predicted major and middle names",
        },
        "teacher_forcing": "Training prefixes use gold parents; inference prefixes use predicted parents.",
        "outputs": outputs,
        "limitations": ["Wrong-parent recovery is impossible because later candidate sets follow the predicted path.",
                        "No new unique complaint text; task views derive from the separate synthetic training corpus."],
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
