"""Measure first-request and cache-hit CPU latency for Kanana-2-v-decide."""
from pathlib import Path
import json
import time

import psutil
import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "evaluation/models/kanana-2-v-decide-3b-2611"
OUT = ROOT / "evaluation/kanana-2-v-decide-cold-cpu"

taxonomy = json.loads((ROOT / "datasets/complaints/taxonomy.json").read_text("utf-8"))
cases = json.loads((ROOT / "datasets/complaints/cases.json").read_text("utf-8"))
criteria = {leaf["id"]: " > ".join(leaf["path"]) + ": " + leaf["description"] for leaf in taxonomy["leaves"]}
question = {"route": {"type": "choice", "instructions": "민원 요약을 읽고 가장 적합한 하나의 대분류 > 중분류 > 소분류 경로를 선택하세요.", "criteria": criteria}}

torch.set_num_threads(8)
torch.set_num_interop_threads(1)
process = psutil.Process()
t0 = time.perf_counter()
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa")
model.eval()
load_seconds = time.perf_counter() - t0

def state(case):
    return f"상담 경로: {case['channel']}\n민원 요약: {case['summary']}"

with torch.inference_mode():
    t0 = time.perf_counter()
    first = model.system_one(state=state(cases[0]), questions=question)
    first_seconds = time.perf_counter() - t0
    rss_after_first = process.memory_info().rss
    t0 = time.perf_counter()
    second = model.system_one(state=state(cases[1]), questions=question)
    second_seconds = time.perf_counter() - t0
    rss_after_second = process.memory_info().rss

summary = {
    "status": "complete",
    "model": "kakaocorp/kanana-2-v-decide-3b-2611",
    "device": "cpu",
    "dtype": "bfloat16",
    "threads": 8,
    "candidate_count": 27,
    "load_seconds": load_seconds,
    "first_request_seconds_including_27_candidate_embeddings": first_seconds,
    "second_request_seconds_candidate_cache_hit": second_seconds,
    "rss_after_first_bytes": rss_after_first,
    "rss_after_second_bytes": rss_after_second,
    "first_choice": first["answers"]["route"]["choice"],
    "second_choice": second["answers"]["route"]["choice"],
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
