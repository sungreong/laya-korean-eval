"""Continue LAYA head training with flat and three hierarchical complaint tasks."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "laya"))
sys.path.insert(0, str(ROOT / "research"))
os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import laya
from laya.train import TrainConfig, items_from_rows, save_checkpoint, train_model


DATA = ROOT / "datasets" / "complaints" / "hierarchical-training"
STAGES = ("flat27", "major3", "middle3", "leaf3")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def score(rows):
    grouped = {}
    for stage in STAGES:
        items = [row for row in rows if row["stage"] == stage]
        correct = sum(row["expected"] == row["predicted"] for row in items)
        nll = float(np.mean([-math.log(max(row["probabilities"][row["expected"]], 1e-9)) for row in items]))
        grouped[stage] = {"n": len(items), "correct": correct, "accuracy": correct / len(items), "nll": nll}
    hierarchy = [grouped[s]["accuracy"] for s in ("major3", "middle3", "leaf3")]
    grouped["hierarchy_macro_accuracy"] = float(np.mean(hierarchy))
    grouped["all_task_macro_accuracy"] = float(np.mean([grouped[s]["accuracy"] for s in STAGES]))
    grouped["all_task_mean_nll"] = float(np.mean([grouped[s]["nll"] for s in STAGES]))
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--base-model", default="evaluation/complaint-training-diverse-e5/selected-model")
    parser.add_argument("--data", default="datasets/complaints/hierarchical-training")
    parser.add_argument("--out", default="evaluation/complaint-hierarchical-balanced-e2")
    args = parser.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise RuntimeError("Choose a fresh output directory")
    out.mkdir(parents=True)
    data_dir = ROOT / args.data
    available_train = read_jsonl(data_dir / "train.jsonl")
    validation = read_jsonl(data_dir / "validation.jsonl")
    stage_for_style = {0: "major3", 1: "middle3", 2: "leaf3"}
    train = [row for row in available_train
             if row["metadata"]["stage"] == stage_for_style[(row["metadata"]["style_index"] - 1) % 3]
             or (row["metadata"]["stage"] == "flat27" and (row["metadata"]["style_index"] - 1) % 3 == 0)]
    assert len(available_train) == 2916 and len(train) == 972 and len(validation) == 972
    assert not {r["state"] for r in train} & {r["state"] for r in validation}

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(43)
    np.random.seed(43)
    torch.manual_seed(43)
    agent = laya.load(str(ROOT / args.base_model), device="cpu", compile=False)
    agent.temperature = [1.0, 1.0, 1.0]
    agent.temperature_by_options = {}
    items, skipped = items_from_rows(agent.tok, train, 2048, 1536)
    if skipped or len(items) != len(train):
        raise RuntimeError(f"Training conversion failed: {skipped}")

    summary = {
        "status": "running",
        "base_model": args.base_model,
        "data": args.data,
        "available_task_views": len(available_train),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "unique_train_states": len({r["state"] for r in train}),
        "tasks": list(STAGES),
        "sampling": "Each of 729 unique states supplies one balanced hierarchy stage; 243 also supply flat27 rehearsal.",
        "epochs_planned": args.epochs,
        "selected_epoch": None,
        "selection": "highest teacher-forced hierarchy macro accuracy, then all-task mean NLL",
        "settings": {
            "freeze_encoder": True,
            "loss": "soft-ce",
            "head_lr": 5e-5,
            "micro_batch": 4,
            "grad_accum": 4,
            "max_len": 2048,
            "head_max_len": 1536,
            "seed": 43,
            "device": "cpu",
            "threads": 4,
        },
        "epoch_history": [],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "executed-train_hierarchical_complaints.py").write_bytes(Path(__file__).read_bytes())

    def save():
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def evaluate(epoch):
        agent.model.eval()
        rows = []
        for i, item in enumerate(validation, 1):
            answer = agent.predict(item["state"], item["questions"], max_len=2048, head_max_len=1536)["answers"]["route"]
            expected = max(item["gold"]["route"]["probabilities"], key=item["gold"]["route"]["probabilities"].get)
            rows.append({
                "id": item["id"],
                "stage": item["metadata"]["stage"],
                "expected": expected,
                "predicted": answer["choice"],
                "probabilities": answer["probabilities"],
            })
            if i % 108 == 0:
                print("validation", epoch, i, "/", len(validation), flush=True)
        measured = score(rows)
        (out / f"validation_epoch_{epoch}_predictions.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return measured

    save()
    best = (-1.0, -float("inf"))
    best_state = None
    start = time.perf_counter()

    def after_epoch(epoch, loss):
        nonlocal best, best_state
        rng_state = torch.random.get_rng_state()
        measured = evaluate(epoch + 1)
        key = (measured["hierarchy_macro_accuracy"], -measured["all_task_mean_nll"])
        if key > best:
            best = key
            summary["selected_epoch"] = epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in agent.model.state_dict().items()
                          if not name.startswith("encoder.")}
            torch.save(best_state, out / "best-head-fp32.pt")
        summary["epoch_history"].append({"epoch": epoch + 1, "train_loss": loss, "validation": measured})
        save()
        print("epoch", epoch + 1, json.dumps(measured, ensure_ascii=False), flush=True)
        agent.model.train()
        agent.model.encoder.eval()
        torch.random.set_rng_state(rng_state)

    cfg = TrainConfig(
        epochs=args.epochs,
        micro_batch=4,
        grad_accum=4,
        head_lr=5e-5,
        loss="soft-ce",
        shuffle_options=("choice",),
        freeze_encoder=True,
        seed=43,
        amp=False,
        gradient_checkpointing=False,
        log_every=100,
        max_len=2048,
        head_max_len=1536,
    )
    losses = train_model(agent.model, agent.tok, items, cfg, torch.device("cpu"), 2048, 1536, on_epoch_end=after_epoch)
    summary["training_and_validation_seconds"] = time.perf_counter() - start
    summary["epoch_losses"] = losses
    agent.model.load_state_dict(best_state, strict=False)
    agent.model.eval()
    checkpoint = out / "selected-model"
    config = dict(agent.cfg, temperature=[1.0, 1.0, 1.0], fine_tuned=True, max_len=2048, head_max_len=1536)
    config.pop("temperature_by_options", None)
    save_checkpoint(agent.model, agent.tok, config, str(checkpoint))
    del agent, best_state
    gc.collect()
    loaded = laya.load(str(checkpoint), device="cpu", compile=False)
    del loaded
    gc.collect()
    summary["selected_checkpoint"] = str(checkpoint.relative_to(ROOT))
    summary["export_dtype"] = "float16"
    summary["status"] = "complete"
    summary["total_seconds"] = time.perf_counter() - start
    save()
    print("FINISHED selected epoch", summary["selected_epoch"], flush=True)


if __name__ == "__main__":
    main()
