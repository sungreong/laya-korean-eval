"""Download the pinned Kanana-2-v-decide snapshot into the mounted workspace."""
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "kakaocorp/kanana-2-v-decide-3b-2611"
REVISION = "1489062def4145ae3a0a9261fbac4f4d88a8c6d1"
TARGET = ROOT / "evaluation/models/kanana-2-v-decide-3b-2611"

TARGET.mkdir(parents=True, exist_ok=True)
path = snapshot_download(repo_id=REPO, revision=REVISION, local_dir=TARGET)
print(path, flush=True)
