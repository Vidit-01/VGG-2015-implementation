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
# Dynamic import helpers
# ------------------------------
def load_transforms(path):
    spec = importlib.util.spec_from_file_location("transform_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.train_transform, module.val_transform


def load_model(model_path, num_classes):
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VGG(num_classes=num_classes)


# ------------------------------
# Data Loaders
# ------------------------------
def get_dataloaders(batch_size, num_workers, transforms_path, distributed):
    train_transform, val_transform = load_transforms(transforms_path)

    train_set = TinyImageNetTrain(transform=train_transform)
    val_set = TinyImageNetVal(transform=val_transform)

    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, train_sampler


# ------------------------------
# TRAIN ONE EPOCH
# ------------------------------
def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs, scaler, rank=0, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0, 0, 0

    # safe autocast for PyTorch 1.13
    if device.type == "cuda":
        autocast = torch.cuda.amp.autocast
    else:
        from contextlib import nullcontext
        autocast = nullcontext

    progress = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", dynamic_ncols=True, leave=False) if rank == 0 else loader

    for x, y in progress:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            out = model(x)
            loss = criterion(out, y)

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if rank == 0:
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return total_loss / len(loader), correct / total


# ------------------------------
# VALIDATION
# ------------------------------
def evaluate(model, loader, device, rank=0):

    model.eval()
    total_loss, correct, total = 0, 0, 0
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        autocast = torch.cuda.amp.autocast
    else:
        from contextlib import nullcontext
        autocast = nullcontext

    progress = tqdm(loader, desc="Val", dynamic_ncols=True, leave=False) if rank == 0 else loader

    with torch.no_grad():
        for x, y in progress:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with autocast():
                out = model(x)
                loss = criterion(out, y)

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

            if rank == 0:
                progress.set_postfix(acc=f"{correct/total:.4f}")

    return correct / total, total_loss / len(loader)


# ------------------------------
# MAIN TRAINING LOOP
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0001)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results/")
    parser.add_argument("--transforms", default=os.path.join("utils", "transforms_baseline.py"))
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Device / DDP
    distributed = "RANK" in os.environ
    rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"[Device] {device}")

    # Load model
    model = load_model(args.model_path, args.num_classes)

    # ⚠️ IMPORTANT FIX:
    # DataParallel FIRST, then torch.compile
    if not distributed and torch.cuda.device_count() > 1:
        if rank == 0:
            print(f"🔀 Using DataParallel ({torch.cuda.device_count()} GPUs)")
        model = nn.DataParallel(model)

    if hasattr(torch, "compile"):
        if rank == 0:
            print("🚀 Compiling model...")
        model = torch.compile(model)

    model = model.to(device)

    # Load data
    train_loader, val_loader, train_sampler = get_dataloaders(
        args.bs, args.workers, args.transforms, distributed
    )

    # Optimizer, Loss, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[args.warmup_epochs]
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Training
    best_acc = 0
    best_epoch = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):

        if distributed:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args.epochs, scaler, rank, args.grad_clip
        )

        val_acc, val_loss = evaluate(model, val_loader, device, rank)

        scheduler.step()

        if rank == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f}")
            print(f"  Val   Loss {val_loss:.4f} | Val   Acc {val_acc:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch

                ckpt = {
                    "epoch": epoch,
                    "state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "args": vars(args)
                }

                torch.save(ckpt, os.path.join(args.out, "best.pth"))
                print(f"💾 Saved best model (acc={best_acc:.4f})")

    if rank == 0:
        print(f"\n🎉 Finished! Best acc = {best_acc:.4f} at epoch {best_epoch}")
        print(f"Total time = { (time.time() - start)/60:.2f} min\n")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
