import argparse
import importlib.util
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm  # Fixed: Using standard tqdm instead of notebook
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
    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True  # Better for batch norm stability
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size * 2,  # Can use larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return train_loader, val_loader, train_sampler


# ------------------------------
# Train one epoch
# ------------------------------
def train_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs, scaler, rank=0, grad_clip=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Only show progress bar on rank 0 (or single GPU)
    if rank == 0:
        progress = tqdm(
            loader,
            desc=f"Train Epoch {epoch}/{total_epochs}",
            dynamic_ncols=True,
            leave=False
        )
    else:
        progress = loader

    for x, y in progress:
        # Non-blocking transfer for better overlap
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

        # Mixed precision training
        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = criterion(out, y)

        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        total_loss += loss.item()
        with torch.no_grad():
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        if rank == 0:
            progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

    avg_loss = total_loss / len(loader)
    avg_acc = correct / total
    
    return avg_loss, avg_acc


# ------------------------------
# Validation
# ------------------------------
def evaluate(model, loader, device, rank=0):
    model.eval()
    correct, total = 0, 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    if rank == 0:
        progress = tqdm(
            loader,
            desc="Validation",
            dynamic_ncols=True,
            leave=False
        )
    else:
        progress = loader

    with torch.no_grad():
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Use AMP for validation too
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = criterion(out, y)
            
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()

            if rank == 0:
                progress.set_postfix({'acc': f'{correct/total:.4f}'})

    return correct / total, total_loss / len(loader)


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)  # Increased for VGG
    parser.add_argument("--wd", type=float, default=0.0001)  # Reduced weight decay
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results/")
    parser.add_argument("--transforms", default=os.path.join("utils", "transforms_baseline.py"))
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # --------------------
    # DDP detection
    # --------------------
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = 0
    world_size = 1

    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        if rank == 0:
            print(f"[DDP] Using {world_size} GPUs")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if rank == 0:
            print(f"[Single Process] Using device: {device}")

    # --------------------
    # Model
    # --------------------
    model = load_model(args.model_path, args.num_classes).to(device)

    # Compile model for better performance (PyTorch 2.0+)
    if hasattr(torch, 'compile') and not distributed:
        if rank == 0:
            print("🚀 Compiling model with torch.compile...")
        model = torch.compile(model)

    # DataParallel fallback for non-DDP multi-GPU
    if not distributed and torch.cuda.device_count() > 1:
        if rank == 0:
            print(f"🔀 Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Wrap model in DDP
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            gradient_as_bucket_view=True  # Performance optimization
        )

    # --------------------
    # Data
    # --------------------
    train_loader, val_loader, train_sampler = get_dataloaders(
        args.bs, args.workers, args.transforms, distributed
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization

    # AdamW is generally better than Adam
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.wd,
        betas=(0.9, 0.999)
    )

    # Cosine annealing with warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=args.warmup_epochs
    )
    
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - args.warmup_epochs,
        eta_min=1e-6
    )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_epochs]
    )

    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0
    best_epoch = 0
    start_time = time.time()

    # --------------------
    # Training loop
    # --------------------
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        if distributed:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer,
            criterion, device, epoch, args.epochs, scaler, rank, args.grad_clip
        )

        val_acc, val_loss = evaluate(model, val_loader, device, rank)

        # Step scheduler
        scheduler.step()

        if rank == 0:
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{args.epochs} | Time: {epoch_time:.2f}s | LR: {current_lr:.6f}")
            print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                
                # Save checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "state_dict": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    "num_classes": args.num_classes,
                    "optimizer": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "args": vars(args)
                }
                
                torch.save(checkpoint, os.path.join(args.out, "best.pth"))
                print(f"💾 Best model saved! (Acc: {best_acc:.4f})")
            
            print(f"Best: {best_acc:.4f} at epoch {best_epoch}")
            print(f"{'='*70}")

    if rank == 0:
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()