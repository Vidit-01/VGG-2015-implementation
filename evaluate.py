import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import time
from tqdm import tqdm

def load_model(model_path, num_classes=10):
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VGG(num_classes=num_classes)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        start_time = time.time()
        progress = tqdm(enumerate(loader))
        for batch_idx,(x, y) in progress:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            elapsed = time.time() - start_time
            batches_done = batch_idx + 1
            batches_total = len(loader)
            eta = elapsed / batches_done * (batches_total - batches_done)

            progress.set_postfix({
                "batch": f"{batches_done}/{batches_total}",
                "eta": f"{eta:.1f}s"
            })
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model .py file")
    parser.add_argument("checkpoint", help="Path to saved .pth file")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model = load_model(args.model_path).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    print("Loaded checkpoint!")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))

    ])

    val_set = datasets.CIFAR10(root="data/cifar10", train=False,
                               transform=transform, download=True)

    val_loader = DataLoader(val_set, batch_size=64,
                            shuffle=False, num_workers=2)

    acc = evaluate(model, val_loader, device)
    print(f"\nAccuracy: {acc*100:.2f}%")



if __name__ == "__main__":
    main()
