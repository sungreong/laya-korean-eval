from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results" / "embeddinggemma2-retrieval-2026-10-07"

SOURCES = {
    "retrieval-main27": ROOT / "evaluation" / "embeddinggemma2-main27-r3",
    "retrieval-main36": ROOT / "evaluation" / "embeddinggemma2-main36",
    "retrieval-unseen36": ROOT / "evaluation" / "embeddinggemma2-unseen36",
    "rerank-main27": ROOT / "evaluation" / "embeddinggemma2-rerank-main27",
    "rerank-main36": ROOT / "evaluation" / "embeddinggemma2-rerank-main36",
    "rerank-unseen36": ROOT / "evaluation" / "embeddinggemma2-rerank-unseen36",
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    index: dict[str, object] = {"experiment_date": "2026-10-07", "artifacts": {}}
    for name, source in SOURCES.items():
        target = DEST / name
        target.mkdir(exist_ok=True)
        copied: list[str] = []
        for src in sorted(source.glob("*.json")) + sorted(source.glob("executed*.py")):
            shutil.copy2(src, target / src.name)
            copied.append(src.name)
        index["artifacts"][name] = copied

    (DEST / "manifest.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
