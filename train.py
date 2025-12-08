import argparse
import importlib.util
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from utils.dataset import trainset,testset


def load_model(model_path, num_classes):
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VGG(num_classes=num_classes)

def get_dataloaders(batch_size, num_workers):
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def train_epoch(model,optimizer,criterion,train_loader,device):
    model.train()
    start_time = time.time()
    progress = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        leave=False
    )
    total, correct, total_loss = 0, 0, 0
    for batch_idx,(x, y) in progress:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
        elapsed = time.time() - start_time
        batches_done = batch_idx + 1
        batches_total = len(train_loader)
        eta = elapsed / batches_done * (batches_total - batches_done)

        progress.set_postfix({
            "batch": f"{batches_done}/{batches_total}",
            "loss": f"{loss.item():.4f}",
            "eta": f"{eta:.1f}s"
        })
    return total_loss / len(train_loader), correct / total

def evaluate(model, test_loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default="results/")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n✅ Using device: {device}")

    model = load_model(args.model_path, args.num_classes).to(device)

    train_loader, val_loader = get_dataloaders(
        args.bs, args.workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        threshold=1e-4,
        min_lr=1e-6,
    )
    es_patience = 3
    es_counter = 0
    best_acc = 0

    for epoch in range(11,20):
        train_loss, train_acc = train_epoch(model,optimizer,criterion,train_loader,device)
        val_acc = evaluate(model,val_loader,device)
        print(f"Epoch {epoch+1}: Loss={train_loss:.3f} | Train Acc={train_acc:.3f} | Test Acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "num_classes": args.num_classes
            }, os.path.join(args.out, "best.pth"))
            print("Saved new best model!")
            improved = True
        else:
            improved = False

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr < old_lr:
            es_counter += 1
            print(f"LR reduced: {old_lr} → {new_lr} (plateau count = {es_counter})")
        else:
            es_counter = max(es_counter - 1, 0)

        if es_counter >= es_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}!")
            break

    print(f"\nTraining complete! Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
