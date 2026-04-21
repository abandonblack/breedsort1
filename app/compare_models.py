from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from app.experiment_utils import build_test_loader, evaluate_topk, load_model_from_checkpoint
from app.model_resnet34 import build_model as build_resnet34
from app.model_seresnet34 import build_model as build_seresnet34


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对比 ResNet34 与 SE-ResNet34 的训练曲线与测试集指标")
    parser.add_argument("--data-dir", type=Path, default=Path("data/oxford_iiit_pet"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-download", action="store_true")

    parser.add_argument("--se-ckpt", type=Path, default=Path("artifacts/seresnet34.pth"))
    parser.add_argument("--se-history", type=Path, default=Path("artifacts/seresnet34_history.json"))
    parser.add_argument("--plain-ckpt", type=Path, default=Path("artifacts/resnet34_plain.pth"))
    parser.add_argument("--plain-history", type=Path, default=Path("artifacts/resnet34_plain_history.json"))

    parser.add_argument("--loss-fig", type=Path, default=Path("artifacts/compare_train_loss.png"))
    parser.add_argument("--valacc-fig", type=Path, default=Path("artifacts/compare_val_acc.png"))
    parser.add_argument("--test-fig", type=Path, default=Path("artifacts/compare_test_topk.png"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/compare_report.json"))
    return parser.parse_args()


def _load_history(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"训练历史文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["history"]


def plot_training_curves(se_history: dict, plain_history: dict, loss_fig: Path, valacc_fig: Path) -> None:
    loss_fig.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(se_history["epochs"], se_history["train_loss"], label="SE-ResNet34", linewidth=2)
    plt.plot(plain_history["epochs"], plain_history["train_loss"], label="ResNet34", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss 对比")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(loss_fig, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(se_history["epochs"], se_history["val_acc"], label="SE-ResNet34", linewidth=2)
    plt.plot(plain_history["epochs"], plain_history["val_acc"], label="ResNet34", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Val Acc 对比")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(valacc_fig, dpi=180)
    plt.close()


def plot_test_bar(topk_se: dict[int, float], topk_plain: dict[int, float], fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    labels = ["Top1", "Top3"]
    x = [0, 1]
    width = 0.34
    se_vals = [topk_se[1], topk_se[3]]
    plain_vals = [topk_plain[1], topk_plain[3]]

    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], se_vals, width=width, label="SE-ResNet34")
    plt.bar([i + width / 2 for i in x], plain_vals, width=width, label="ResNet34")
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Official Test Top1 / Top3 对比")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    se_history = _load_history(args.se_history)
    plain_history = _load_history(args.plain_history)
    plot_training_curves(se_history, plain_history, args.loss_fig, args.valacc_fig)

    test_loader, _ = build_test_loader(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        no_download=args.no_download,
    )

    se_model, _ = load_model_from_checkpoint(args.se_ckpt, build_seresnet34, device)
    plain_model, _ = load_model_from_checkpoint(args.plain_ckpt, build_resnet34, device)

    topk_se = evaluate_topk(se_model, test_loader, device, topk=(1, 3))
    topk_plain = evaluate_topk(plain_model, test_loader, device, topk=(1, 3))
    plot_test_bar(topk_se, topk_plain, args.test_fig)

    report = {
        "dataset": "Oxford-IIIT Pet official test split",
        "se_resnet34": {"top1": topk_se[1], "top3": topk_se[3]},
        "resnet34_plain": {"top1": topk_plain[1], "top3": topk_plain[3]},
        "figures": {
            "train_loss": str(args.loss_fig),
            "val_acc": str(args.valacc_fig),
            "test_topk": str(args.test_fig),
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== 官方 Test 结果 ==")
    print(f"SE-ResNet34: top1={topk_se[1]:.4f}, top3={topk_se[3]:.4f}")
    print(f"ResNet34:    top1={topk_plain[1]:.4f}, top3={topk_plain[3]:.4f}")
    print(f"报告已保存: {args.report}")


if __name__ == "__main__":
    main()
