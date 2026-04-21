from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.experiment_utils import DATASET_CATALOG, build_train_val_loaders, save_json, set_seed, train_model
from app.model_seresnet34 import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 SE-ResNet34（Oxford-IIIT Pet）")
    parser.add_argument("--data-dir", type=Path, default=Path("data/oxford_iiit_pet"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/seresnet34.pth"))
    parser.add_argument("--history-path", type=Path, default=Path("artifacts/seresnet34_history.json"))
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"使用数据集：{DATASET_CATALOG['oxford_iiit_pet']['name']}")
    train_loader, val_loader, class_names, train_size, _ = build_train_val_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        workers=args.workers,
        seed=args.seed,
        no_download=args.no_download,
    )

    model = build_model(num_classes=len(class_names)).to(device)
    result = train_model(
        model=model,
        model_name="seresnet34",
        class_names=class_names,
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=train_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        device=device,
        save_path=args.save_path,
    )

    meta = {
        "arch": "seresnet34",
        "dataset_preset": "oxford_iiit_pet",
        "dataset_info": DATASET_CATALOG["oxford_iiit_pet"],
        "classes": class_names,
        "best_val_acc": result["best_val_acc"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "history": result["history"],
    }
    save_json(meta, args.history_path)
    print(f"训练完成，最佳验证准确率: {result['best_val_acc']:.4f}")
    print(f"权重已保存: {args.save_path}")
    print(f"训练历史已保存: {args.history_path}")


if __name__ == "__main__":
    main()
