from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DATASET_CATALOG = {
    "oxford_iiit_pet": {
        "name": "Oxford-IIIT Pet Dataset",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "description": "37 个猫狗品种，官方标注质量高，适合品种分类。",
    }
}


TransformPair = tuple[transforms.Compose, transforms.Compose]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_transforms(image_size: int) -> TransformPair:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def _class_name_from_path(image_path: str) -> str:
    image_name = Path(image_path).stem
    return "_".join(image_name.split("_")[:-1]).lower()


def build_class_names(dataset: datasets.OxfordIIITPet) -> list[str]:
    id_to_name: dict[int, str] = {}
    for image_path, label in zip(dataset._images, dataset._labels):
        class_index = int(label) - 1
        if class_index not in id_to_name:
            id_to_name[class_index] = _class_name_from_path(image_path)
    return [id_to_name[idx] for idx in sorted(id_to_name.keys())]


def build_train_val_loaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    val_split: float,
    workers: int,
    seed: int,
    no_download: bool,
) -> tuple[DataLoader, DataLoader, list[str], int, int]:
    train_transform, eval_transform = build_transforms(image_size)

    full_train = datasets.OxfordIIITPet(
        root=str(data_dir),
        split="trainval",
        target_types="category",
        transform=train_transform,
        download=not no_download,
    )
    val_base = datasets.OxfordIIITPet(
        root=str(data_dir),
        split="trainval",
        target_types="category",
        transform=eval_transform,
        download=False,
    )
    class_names = build_class_names(full_train)
    if len(class_names) < 2:
        raise ValueError("至少需要两个品种类别进行训练")

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)

    all_indices = torch.randperm(len(full_train), generator=generator).tolist()
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    train_set = Subset(full_train, train_indices)
    val_set = Subset(val_base, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, class_names, train_size, val_size


def build_test_loader(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    no_download: bool,
) -> tuple[DataLoader, list[str]]:
    _, eval_transform = build_transforms(image_size)
    test_set = datasets.OxfordIIITPet(
        root=str(data_dir),
        split="test",
        target_types="category",
        transform=eval_transform,
        download=not no_download,
    )
    class_names = build_class_names(test_set)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, class_names


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_topk(model: nn.Module, loader: DataLoader, device: torch.device, topk: tuple[int, ...] = (1, 3)) -> dict[int, float]:
    model.eval()
    correct = {k: 0 for k in topk}
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            maxk = max(topk)
            _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            target = labels.view(1, -1).expand_as(pred)
            match = pred.eq(target)
            for k in topk:
                correct[k] += match[:k].any(dim=0).sum().item()
            total += labels.size(0)
    return {k: correct[k] / max(total, 1) for k in topk}


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def train_model(
    model: nn.Module,
    model_name: str,
    class_names: list[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    device: torch.device,
    save_path: Path,
) -> dict:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {"epochs": [], "train_loss": [], "val_loss": [], "val_acc": []}
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(train_size, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{model_name}] Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr={scheduler.get_last_lr()[0]:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "arch": model_name,
                    "model_state_dict": model.state_dict(),
                    "classes": class_names,
                    "image_size": getattr(model, "image_size", 224),
                    "dataset_preset": "oxford_iiit_pet",
                    "best_val_acc": best_acc,
                },
                save_path,
            )

    return {"best_val_acc": best_acc, "history": history}


def load_model_from_checkpoint(
    checkpoint_path: Path,
    builder: Callable[[int], nn.Module],
    device: torch.device,
) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    model = builder(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
