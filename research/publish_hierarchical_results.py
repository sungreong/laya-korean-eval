"""Publish the small hierarchical/prefix experiment without model checkpoints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BLOG = ROOT.parent / "laya-blog-2026-10-05"
OUT = ROOT / "results" / "hierarchical-prefix-2026-10-07"


RUNS = {
    "laya/hierarchy-train": ROOT / "evaluation/complaint-hierarchical-balanced-e2",
    "laya/prefix-train": ROOT / "evaluation/complaint-hierarchical-prefix-e1",
    "laya/test": ROOT / "evaluation/complaints-hierarchical-prefix-final",
    "laya/unseen": ROOT / "evaluation/unseen-laya-hierarchical-prefix",
    "laya/base-unseen": ROOT / "evaluation/unseen-laya-base",
    "kobert/hierarchy-train": ROOT / "evaluation/kobert-mask-hierarchical-balanced-e2",
    "kobert/prefix-train": ROOT / "evaluation/kobert-mask-hierarchical-prefix-e1",
    "kobert/test": ROOT / "evaluation/kobert-mask-hierarchical-prefix-final",
    "kobert/unseen": ROOT / "evaluation/unseen-kobert-hierarchical-prefix",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def copy_run(name: str, source: Path) -> None:
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    for file in source.glob("*.json"):
        shutil.copy2(file, target / file.name)
    for file in source.glob("executed-*.py"):
        shutil.copy2(file, target / file.name)


def acc(summary: dict, method: str) -> float:
    return 100 * summary["methods"][method]["strict"]["exact_path_accuracy"]


def main() -> None:
    missing = [str(path) for path in RUNS.values() if not (path / "summary.json").exists()]
    if missing:
        raise SystemExit(f"Missing completed result summaries: {missing}")

    for name, source in RUNS.items():
        copy_run(name, source)

    laya_test = read_json(RUNS["laya/test"] / "summary.json")
    kobert_test = read_json(RUNS["kobert/test"] / "summary.json")
    laya_unseen = read_json(RUNS["laya/unseen"] / "summary.json")
    kobert_unseen = read_json(RUNS["kobert/unseen"] / "summary.json")
    laya_train = read_json(RUNS["laya/prefix-train"] / "summary.json")
    kobert_train = read_json(RUNS["kobert/prefix-train"] / "summary.json")

    comparison = {
        "scope": "Author-created synthetic diagnostic; one seed; frozen encoders; not a production benchmark.",
        "training_data_separation": {
            "unique_train_states": 729,
            "unique_validation_states": 243,
            "main_test_cases": 90,
            "unseen_test_cases": 27,
            "exact_text_overlap": 0,
        },
        "prefix_protocol": {
            "training": "Gold parent labels are injected as prefixes (teacher forcing).",
            "inference": "Predicted parent labels are injected as prefixes; no gold label is available.",
            "risk": "A wrong parent removes the correct child path from later candidate sets.",
        },
        "main_test_exact_path_accuracy_percent": {
            "laya": {method: round(acc(laya_test, method), 2) for method in laya_test["methods"]},
            "kobert_mask": {method: round(acc(kobert_test, method), 2) for method in kobert_test["methods"]},
        },
        "teacher_forced_validation_percent": {
            "laya": {k: round(100 * v["accuracy"], 2) for k, v in laya_train["epoch_history"][-1]["validation"].items() if isinstance(v, dict)},
            "kobert_mask": {k: round(100 * v["accuracy"], 2) for k, v in kobert_train["epoch_history"][-1]["validation"].items() if isinstance(v, dict)},
        },
        "unseen_exact_path_accuracy_percent": {
            model: {
                method: {
                    novelty: round(100 * metrics["exact_path_accuracy"], 2)
                    for novelty, metrics in data["methods"][method]["by_novelty"].items()
                }
                for method in ("flat27", "cascade", "cascade_prefix")
            }
            for model, data in (("laya", laya_unseen), ("kobert_mask", kobert_unseen))
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), dpi=160)
    colors = ["#2563EB", "#14B8A6", "#F97316"]

    labels = ["27개 일괄", "순차", "순차+prefix"]
    methods = ["flat27", "cascade", "cascade_prefix"]
    for ax, title, summary in ((axes[0], "LAYA 독립 시험", laya_test), (axes[1], "KoBERT [MASK] 독립 시험", kobert_test)):
        values = [acc(summary, method) for method in methods]
        bars = ax.bar(labels, values, color=colors, width=0.68)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("전체 경로 정확도 (%)")
        ax.set_ylim(0, 42)
        ax.grid(axis="y", alpha=0.2)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=3, fontsize=9)

    novelty = ["새 소분류", "새 중분류", "새 대분류"]
    novelty_keys = ["new_leaf", "new_middle", "new_major"]
    x = range(3)
    width = 0.36
    laya_values = [100 * laya_unseen["methods"]["cascade_prefix"]["by_novelty"][key]["exact_path_accuracy"] for key in novelty_keys]
    kobert_values = [100 * kobert_unseen["methods"]["cascade_prefix"]["by_novelty"][key]["exact_path_accuracy"] for key in novelty_keys]
    b1 = axes[2].bar([i - width / 2 for i in x], laya_values, width, label="LAYA", color="#2563EB")
    b2 = axes[2].bar([i + width / 2 for i in x], kobert_values, width, label="KoBERT [MASK]", color="#F97316")
    axes[2].set_title("학습에서 제외한 유형 · 순차+prefix", fontweight="bold")
    axes[2].set_ylabel("전체 경로 정확도 (%)")
    axes[2].set_xticks(list(x), novelty)
    axes[2].set_ylim(0, 42)
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend(frameon=False)
    axes[2].bar_label(b1, labels=[f"{v:.1f}%" for v in laya_values], padding=3, fontsize=8)
    axes[2].bar_label(b2, labels=[f"{v:.1f}%" for v in kobert_values], padding=3, fontsize=8)

    fig.suptitle("계층 학습과 predicted-prefix 평가 결과", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.02, "합성 자료 · 단일 seed · encoder 동결. 정확도는 생산 성능을 뜻하지 않는다.", ha="center", color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))

    original = BLOG / "assets/09-hierarchical-prefix-results.png"
    optimized = BLOG / "assets/09-hierarchical-prefix-results.webp"
    fig.savefig(original, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    with Image.open(original) as image:
        image.save(optimized, "WEBP", quality=85, method=6)
        dimensions = list(image.size)

    before = original.stat().st_size
    after = optimized.stat().st_size
    manifest_path = BLOG / "assets/image-manifest.json"
    manifest = read_json(manifest_path)
    manifest = [item for item in manifest if item["original"] != original.name]
    manifest.append({
        "original": original.name,
        "optimized": optimized.name,
        "original_dimensions": dimensions,
        "display_dimensions": dimensions,
        "before_bytes": before,
        "after_bytes": after,
        "reduction_percent": round(100 * (before - after) / before, 2),
        "quality": 85,
    })
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(original, OUT / "hierarchical-prefix-results.png")
    shutil.copy2(optimized, OUT / "hierarchical-prefix-results.webp")
    print(json.dumps({"published": str(OUT), "image": str(optimized), "before": before, "after": after}, ensure_ascii=False))


if __name__ == "__main__":
    main()
