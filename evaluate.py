import argparse
import importlib.util
import torch
from torch.utils.data import DataLoader

from utils.dataset import TinyImageNetVal
from utils.transforms import val_transform


def load_model(model_path, num_classes):
    spec = importlib.util.spec_from_file_location("model_module", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VGG(num_classes=num_classes)


def evaluate(model, loader, device):
    model.eval()
    top1_correct, top5_correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)  # [batch, num_classes]

            # ---- TOP-1 ----
            top1_preds = logits.argmax(dim=1)
            top1_correct += (top1_preds == y).sum().item()

            # ---- TOP-5 ----
            top5_preds = torch.topk(logits, k=5, dim=1).indices  # [batch, 5]
            top5_correct += top5_preds.eq(y.unsqueeze(1)).any(dim=1).sum().item()

            total += y.size(0)

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    return top1_acc, top5_acc



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model .py file")
    parser.add_argument("checkpoint", help="Path to saved .pth file")
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # auto device fallback
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\nUsing device: {device}")

    # load model architecture
    model = load_model(args.model_path, args.num_classes).to(device)

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded checkpoint from: {args.checkpoint}")

    # build validation dataloader
    val_set = TinyImageNetVal(transform=val_transform)
    val_loader = DataLoader(val_set,
                            batch_size=args.bs,
                            shuffle=False,
                            num_workers=args.workers)

    # evaluate
    acc1,acc5 = evaluate(model, val_loader, device)

    print(f"\nValidation Accuracy \nTOP-1: {acc1 * 100:.2f}%\nTOP-5: {acc5*100:.2f}")


if __name__ == "__main__":
    main()
