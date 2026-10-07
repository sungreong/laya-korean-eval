"""Continue KoBERT real-[MASK] scorer training on flat and hierarchical tasks."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

import numpy as np
import torch
from kobert_mask_model import collate, load_trained, pack_one, save_head


DATA = ROOT / "datasets" / "complaints" / "hierarchical-training"
STAGES = ("flat27", "major3", "middle3", "leaf3")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def target(row):
    probabilities = row["gold"]["route"]["probabilities"]
    return max(probabilities, key=probabilities.get)


def summarize(predictions):
    result = {}
    for stage in STAGES:
        rows = [row for row in predictions if row["stage"] == stage]
        correct = sum(row["expected"] == row["predicted"] for row in rows)
        nll = float(np.mean([-math.log(max(row["probabilities"][row["expected"]], 1e-9)) for row in rows]))
        result[stage] = {"n": len(rows), "correct": correct, "accuracy": correct / len(rows), "nll": nll}
    result["hierarchy_macro_accuracy"] = float(np.mean([result[s]["accuracy"] for s in ("major3", "middle3", "leaf3")]))
    result["all_task_macro_accuracy"] = float(np.mean([result[s]["accuracy"] for s in STAGES]))
    result["all_task_mean_nll"] = float(np.mean([result[s]["nll"] for s in STAGES]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--base-checkpoint", default="evaluation/kobert-mask-direct-e5/selected-model")
    parser.add_argument("--out", default="evaluation/kobert-mask-hierarchical-balanced-e2")
    args = parser.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise RuntimeError("Choose a fresh output directory")
    out.mkdir(parents=True)
    available_train = read_jsonl(DATA / "train.jsonl")
    validation = read_jsonl(DATA / "validation.jsonl")
    stage_for_style = {0: "major3", 1: "middle3", 2: "leaf3"}
    train = [row for row in available_train
             if row["metadata"]["stage"] == stage_for_style[(row["metadata"]["style_index"] - 1) % 3]
             or (row["metadata"]["stage"] == "flat27" and (row["metadata"]["style_index"] - 1) % 3 == 0)]
    assert len(available_train) == 2916 and len(train) == 972 and len(validation) == 972

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    random.seed(43)
    np.random.seed(43)
    torch.manual_seed(43)
    model_dir = ROOT / "evaluation" / "models" / "kobert-base-v1"
    model, base_config = load_trained(model_dir, ROOT / args.base_checkpoint)
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    tokenizer = model.tokenizer
    model.to(torch.device("cpu"))
    parameters = list(model.head.parameters()) + list(model.scorer.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.head_lr, weight_decay=0.01)
    updates = math.ceil(math.ceil(len(train) / 8) / 2) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=args.head_lr / 100)

    summary = {
        "status": "running",
        "model": "skt/kobert-base-v1 real-[MASK] shared scorer",
        "base_checkpoint": args.base_checkpoint,
        "base_config": base_config,
        "data": "datasets/complaints/hierarchical-training",
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
            "batch_size": 8,
            "grad_accum": 2,
            "head_lr": args.head_lr,
            "max_length": 512,
            "seed": 43,
            "device": "cpu",
            "threads": 4,
        },
        "epoch_history": [],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "executed-train_hierarchical_kobert_mask.py").write_bytes(Path(__file__).read_bytes())

    def save():
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def evaluate(epoch):
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(validation), 8):
                chunk = validation[start:start + 8]
                packed, targets = [], []
                for row in chunk:
                    q = row["questions"]["route"]
                    item = pack_one(tokenizer, row["state"], q["instructions"], q["criteria"])
                    packed.append(item)
                    targets.append(item["keys"].index(target(row)))
                batch = collate(packed, tokenizer.pad_token_id, targets)
                logits = model(**{key: value for key, value in batch.items() if key != "targets"})
                probs = torch.softmax(logits, -1).cpu().numpy()
                for row, item, values, expected_index in zip(chunk, packed, probs, targets):
                    predicted_index = int(values.argmax())
                    predictions.append({
                        "id": row["id"],
                        "stage": row["metadata"]["stage"],
                        "expected": item["keys"][expected_index],
                        "predicted": item["keys"][predicted_index],
                        "probabilities": dict(zip(item["keys"], map(float, values))),
                    })
                if (start // 8 + 1) % 20 == 0:
                    print("validation", epoch, min(start + 8, len(validation)), "/", len(validation), flush=True)
        measured = summarize(predictions)
        (out / f"validation_epoch_{epoch}_predictions.json").write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return measured

    save()
    summary["validation_before_training"] = evaluate(0)
    save()
    print("validation before training", json.dumps(summary["validation_before_training"], ensure_ascii=False), flush=True)
    best = (-1.0, -float("inf"))
    best_state = None
    start_time = time.perf_counter()
    rng = random.Random(43)
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.encoder.eval()
        order = list(range(len(train)))
        random.Random(43 + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        steps = 0
        for batch_number, start in enumerate(range(0, len(order), 8), 1):
            packed, targets = [], []
            for index in order[start:start + 8]:
                row = train[index]
                q = row["questions"]["route"]
                permutation = list(range(len(q["criteria"])))
                rng.shuffle(permutation)
                item = pack_one(tokenizer, row["state"], q["instructions"], q["criteria"], order=permutation)
                packed.append(item)
                targets.append(item["keys"].index(target(row)))
            batch = collate(packed, tokenizer.pad_token_id, targets)
            logits = model(**{key: value for key, value in batch.items() if key != "targets"})
            loss = torch.nn.functional.cross_entropy(logits, batch["targets"])
            (loss / 2).backward()
            loss_sum += float(loss.detach())
            steps += 1
            if batch_number % 2 == 0 or start + 8 >= len(order):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batch_number % 50 == 0:
                print("epoch", epoch, "batch", batch_number, "loss", round(float(loss.detach()), 4), flush=True)
        measured = evaluate(epoch)
        mean_loss = loss_sum / steps
        key = (measured["hierarchy_macro_accuracy"], -measured["all_task_mean_nll"])
        if key > best:
            best = key
            summary["selected_epoch"] = epoch
            best_state = model.trainable_state()
            torch.save(best_state, out / "best-head-fp32.pt")
        summary["epoch_history"].append({"epoch": epoch, "train_loss": mean_loss, "validation": measured})
        save()
        print("epoch", epoch, "mean", mean_loss, "validation", json.dumps(measured, ensure_ascii=False), flush=True)
    model.load_state_dict(best_state, strict=False)
    checkpoint = out / "selected-model"
    save_head(model, checkpoint, {"base_revision": base_config["base_revision"], "max_length": 512, "hierarchical_joint": True})
    del model
    reloaded, _ = load_trained(model_dir, checkpoint)
    reloaded.eval()
    summary["status"] = "complete"
    summary["selected_checkpoint"] = str(checkpoint.relative_to(ROOT))
    summary["total_seconds"] = time.perf_counter() - start_time
    save()
    print("FINISHED selected epoch", summary["selected_epoch"], flush=True)


if __name__ == "__main__":
    main()
