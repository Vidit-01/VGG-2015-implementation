import argparse
import importlib.util
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import time

from utils.dataset import TinyImageNetTrain, TinyImageNetVal


# ------------------------------
# Load transforms dynamically
# ------------------------------
def load_transforms(path):
    spec = importlib.util.spec_from_file_location("transform_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.train_transform, module.val_transform


# ------------------------------
# Load model dynamically
# ------------------------------
def load_model(model_path, num_classes):
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VGG(num_classes=num_classes)


# ------------------------------
# Dataloaders
# ------------------------------
def get_dataloaders(batch_size, num_workers, transforms_path, distributed):
    train_transform, val_transform = load_transforms(transforms_path)

    train_set = TinyImageNetTrain(transform=train_transform)
    val_set = TinyImageNetVal(transform=val_transform)

    # Sampler for DDP
    train_sampler = DistributedSampler(train_set) if distributed else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler


# ------------------------------
# Train one epoch
# ------------------------------
def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs, scaler):
    model.train()
    total_loss = 0
    start_time = time.time()

    progress = tqdm(
        enumerate(loader), total=len(loader),
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        leave=False
    )

    for batch_idx, (x, y) in progress:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        # AMP
        with torch.cuda.amp.autocast(enabled=True):
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


# ------------------------------
# Validation
# ------------------------------
def evaluate(model, loader, device, epoch, total_epochs):
    model.eval()
    correct, total = 0, 0

    progress = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Epoch {epoch}/{total_epochs} [val]",
        leave=False
    )

    with torch.no_grad():
        for batch_idx, (x, y) in progress:
            x, y = x.to(device), y.to(device)

            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=0.0005)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results/")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--transforms", default=os.path.join("utils", "transforms_baseline.py"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --------------------
    # DDP detection
    # --------------------
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        print(f"[DDP] Using GPU {local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Single Process] Using device: {device}")

    # --------------------
    # Model
    # --------------------
    model = load_model(args.model_path, args.num_classes).to(device)

    # If not running DDP, fallback to DataParallel automatically
    if not distributed and torch.cuda.device_count() > 1:
        print(f"🔀 Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Wrap model in DDP
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    # --------------------
    # Data
    # --------------------
    train_loader, val_loader, train_sampler = get_dataloaders(
        args.bs, args.workers, args.transforms, distributed
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.wd
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3,
        threshold=1e-4, min_lr=1e-6
    )

    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0
    es_counter = 0
    es_patience = 3

    # --------------------
    # Training loop
    # --------------------
    for epoch in range(1, args.epochs + 1):

        if distributed:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model, train_loader, optimizer,
            criterion, device, epoch, args.epochs, scaler
        )

        val_acc = evaluate(model, val_loader, device, epoch, args.epochs)

        if not distributed or dist.get_rank() == 0:
            print(f"\nEpoch {epoch}: loss={train_loss:.4f} acc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {"state_dict": model.state_dict(), "num_classes": args.num_classes},
                    os.path.join(args.out, "best.pth")
                )
                print("💾 Best model saved!")

            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_acc)
            new_lr = optimizer.param_groups[0]["lr"]

            if new_lr < old_lr:
                es_counter += 1
            else:
                es_counter = max(es_counter - 1, 0)

            if es_counter >= es_patience:
                print("🛑 Early Stopping Triggered")
                break

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
