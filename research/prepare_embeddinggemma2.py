"""Download one pinned EmbeddingGemma 2 snapshot for offline Docker evaluation."""
from pathlib import Path
import hashlib
import json

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
DEST = ROOT / "evaluation" / "models" / f"embeddinggemma-2-{REVISION[:12]}"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO_ID,
        revision=REVISION,
        local_dir=DEST,
        ignore_patterns=["*.md", ".gitattributes"],
    )
    files = []
    for path in sorted(p for p in DEST.rglob("*") if p.is_file()):
        files.append({
            "path": path.relative_to(DEST).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    manifest = {"repo_id": REPO_ID, "revision": REVISION, "files": files,
                "total_bytes": sum(item["bytes"] for item in files)}
    (DEST / "download-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"destination": str(DEST), "files": len(files),
                      "total_bytes": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
