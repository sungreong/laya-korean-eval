"""Plot complaint-routing training curves and create an optimized WebP copy."""
from pathlib import Path
import argparse
import json


def load_history(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status") != "complete":
        raise RuntimeError(f"Training is not complete: {path}")
    return data["epoch_history"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--laya", required=True)
    parser.add_argument("--kobert", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from PIL import Image

    runs = {
        "LAYA": load_history(args.laya),
        "KoBERT [MASK]": load_history(args.kobert),
    }
    colors = {"LAYA": "#246BFD", "KoBERT [MASK]": "#EF6C35"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), dpi=160)
    fields = [
        ("train_loss", "Training loss", None),
        ("accuracy", "Validation accuracy", (0, 1)),
        ("nll", "Validation NLL", None),
    ]
    for label, history in runs.items():
        epochs = [row["epoch"] for row in history]
        for ax, (field, title, ylim) in zip(axes, fields):
            values = [row[field] if field == "train_loss" else row["validation"][field] for row in history]
            ax.plot(epochs, values, marker="o", linewidth=2.2, label=label, color=colors[label])
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_xticks(epochs)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=.22)
    axes[1].set_ylabel("Score")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False)
    fig.suptitle("27-way Korean complaint routing: training curves", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    webp = out.with_suffix(".webp")
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with Image.open(png) as image:
        image.save(webp, "WEBP", quality=85, method=6)
    before, after = png.stat().st_size, webp.stat().st_size
    with Image.open(webp) as image:
        width, height = image.size
    print(json.dumps({
        "png": str(png), "webp": str(webp), "dimensions": [width, height],
        "png_bytes": before, "webp_bytes": after,
        "reduction_percent": round((1 - after / before) * 100, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
