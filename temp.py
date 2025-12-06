import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

# ---------------------------
# 1. DATA
# ---------------------------
transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

# ---------------------------
# 2. SIMPLE MODEL (VGG-like)
# ---------------------------
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(8*8*128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyCNN().to(device)

# ---------------------------
# 3. LOSS + OPTIM
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# 4. TRAIN FUNCTION
# ---------------------------
def train_epoch():
    model.train()
    total, correct, total_loss = 0, 0, 0
    for x, y in train_loader:
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

    return total_loss / len(train_loader), correct / total

# ---------------------------
# 5. TEST FUNCTION
# ---------------------------
def test_epoch():
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

# ---------------------------
# 6. TRAIN LOOP
# ---------------------------
for epoch in range(10):
    train_loss, train_acc = train_epoch()
    test_acc = test_epoch()
    print(f"Epoch {epoch+1}: Loss={train_loss:.3f} | Train Acc={train_acc:.3f} | Test Acc={test_acc:.3f}")
