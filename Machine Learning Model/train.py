import argparse
import os
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Azure ML run context (for Metrics tab logging)
from azureml.core import Run

run = Run.get_context()


def stratified_split_indices(targets, val_split=0.2, seed=42):
    """
    Returns (train_indices, val_indices) with approximately stratified class proportions.
    Ensures at least 1 sample per class in train when possible.
    """
    g = torch.Generator().manual_seed(seed)

    # group indices by class
    class_to_indices = defaultdict(list)
    for idx, y in enumerate(targets):
        class_to_indices[int(y)].append(idx)

    train_idx, val_idx = [], []

    for cls, idxs in class_to_indices.items():
        idxs_tensor = torch.tensor(idxs)
        perm = idxs_tensor[torch.randperm(len(idxs_tensor), generator=g)].tolist()

        # how many go to val for this class
        n_cls = len(perm)
        n_val = int(round(n_cls * val_split))

        # guardrails for tiny classes
        if n_cls == 1:
            # If only 1 sample exists, keep it in train (val would be impossible to evaluate meaningfully)
            n_val = 0
        else:
            # Keep at least 1 in train if possible
            n_val = min(n_val, n_cls - 1)

        val_part = perm[:n_val]
        train_part = perm[n_val:]

        val_idx.extend(val_part)
        train_idx.extend(train_part)

    # shuffle final lists (optional, but keeps things mixed)
    train_idx = torch.tensor(train_idx)
    val_idx = torch.tensor(val_idx)

    train_idx = train_idx[torch.randperm(len(train_idx), generator=g)].tolist()
    val_idx = val_idx[torch.randperm(len(val_idx), generator=g)].tolist()

    return train_idx, val_idx


def compute_class_counts(subset, num_classes):
    """
    subset is a torch.utils.data.Subset whose underlying dataset is ImageFolder.
    We rely on subset.dataset.targets and subset.indices.
    """
    targets = subset.dataset.targets
    counts = [0] * num_classes
    for i in subset.indices:
        counts[int(targets[i])] += 1
    return counts


def log_class_counts(prefix, classes, counts):
    """
    Prints and logs counts to AzureML metrics.
    """
    print(f"{prefix} counts:")
    for cls_name, c in zip(classes, counts):
        print(f"  {cls_name}: {c}")
        # log as individual metrics so you can see them in Metrics tab
        run.log(f"{prefix}_count_{cls_name}", int(c))


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    img_size: int = 224,
    val_split: float = 0.2,
    seed: int = 42,
):
    """
    Expects folder layout:
      data_dir/
        OK/
        WARNING/
        REPLACE/
    """
    # Stronger but still lightweight augmentation
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )

    # Build two datasets pointing at the same folder, but with different transforms.
    # This avoids mutating a shared dataset and accidentally changing train/val transforms.
    train_base = datasets.ImageFolder(root=data_dir, transform=train_tfms)
    val_base = datasets.ImageFolder(root=data_dir, transform=eval_tfms)

    n_total = len(train_base)
    if n_total == 0:
        raise RuntimeError(
            f"No images found under {data_dir}. Verify OK/WARNING/REPLACE subfolders."
        )

    targets = train_base.targets  # same ordering as val_base
    train_idx, val_idx = stratified_split_indices(targets, val_split=val_split, seed=seed)

    # Fallback if something weird happens (e.g., val ends up empty)
    if len(val_idx) == 0 and n_total > 1:
        val_idx = [train_idx.pop()]  # move one sample to val

    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(val_base, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, train_base.classes


def set_backbone_trainable(model, trainable: bool):
    """
    Freeze/unfreeze all layers except the classifier head.
    ResNet18 head is model.fc.
    """
    for p in model.parameters():
        p.requires_grad = trainable

    # Always keep the head trainable
    for p in model.fc.parameters():
        p.requires_grad = True


def unfreeze_last_block(model):
    """
    Unfreeze layer4 (last ResNet block) + fc for fine-tuning.
    """
    # Keep everything frozen first, then unfreeze last block
    for p in model.parameters():
        p.requires_grad = False

    for p in model.layer4.parameters():
        p.requires_grad = True

    for p in model.fc.parameters():
        p.requires_grad = True


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Mounted dataset folder (uri_folder)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)  # lower default helps overfitting
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--output_dir", default="./outputs")
    ap.add_argument("--seed", type=int, default=42)

    # Freeze schedule:
    ap.add_argument("--freeze_epochs", type=int, default=2, help="Train only head for N epochs")
    ap.add_argument("--finetune_last_block", action="store_true", help="Unfreeze layer4 after freeze_epochs")
    args = ap.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise RuntimeError(f"data_dir not found or not a directory: {data_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Data dir: {data_dir}")

    # Log basic run config so it’s visible in run details
    run.log("epochs", int(args.epochs))
    run.log("batch_size", int(args.batch_size))
    run.log("lr", float(args.lr))
    run.log("img_size", int(args.img_size))
    run.log("val_split", float(args.val_split))
    run.log("seed", int(args.seed))
    run.log("freeze_epochs", int(args.freeze_epochs))
    run.log("finetune_last_block", 1 if args.finetune_last_block else 0)
    run.log("device_is_cuda", 1 if device == "cuda" else 0)

    train_loader, val_loader, classes = get_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    num_classes = len(classes)
    print(f"Classes: {classes} (num_classes={num_classes})")
    run.log("num_classes", int(num_classes))
    run.log("num_images_total", int(len(train_loader.dataset) + len(val_loader.dataset)))
    run.log("num_images_train", int(len(train_loader.dataset)))
    run.log("num_images_val", int(len(val_loader.dataset)))

    # Print/log class distributions
    train_counts = compute_class_counts(train_loader.dataset, num_classes)
    val_counts = compute_class_counts(val_loader.dataset, num_classes)
    log_class_counts("train", classes, train_counts)
    log_class_counts("val", classes, val_counts)

    # Transfer learning: ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Start with head-only training
    set_backbone_trainable(model, trainable=False)

    # Optimizer should only see trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_val_acc = 0.0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        # At epoch freeze_epochs+1, optionally fine-tune last block
        if epoch == args.freeze_epochs + 1:
            if args.finetune_last_block:
                print("Unfreezing last block (layer4) + fc for fine-tuning.")
                unfreeze_last_block(model)
            else:
                print("Unfreezing full backbone for fine-tuning.")
                set_backbone_trainable(model, trainable=True)

            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.lr * 0.3,  # smaller LR when fine-tuning
                weight_decay=1e-4,
            )
            run.log("finetune_started_epoch", int(epoch))

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

        # --- Azure ML Metrics logging (shows under Job > Metrics) ---
        run.log("train_loss", float(tr_loss))
        run.log("train_accuracy", float(tr_acc))
        run.log("val_loss", float(va_loss))
        run.log("val_accuracy", float(va_acc))
        run.log("epoch", int(epoch))

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "img_size": args.img_size,
                },
                ckpt_path,
            )
            print(f"Saved new best checkpoint: {ckpt_path} (val_acc={best_val_acc:.3f})")

    print(f"Done. Best val acc: {best_val_acc:.3f}")
    print(f"Checkpoint: {ckpt_path}")
    run.log("best_val_accuracy", float(best_val_acc))


if __name__ == "__main__":
    main()
