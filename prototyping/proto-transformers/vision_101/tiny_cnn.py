import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import argparse, ipdb, random
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--label_smoothing', type=float, default=0.1)
args = parser.parse_args()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Select 10% of the training data with balanced classes using StratifiedShuffleSplit
full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
targets = full_trainset.targets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=42)
subset_idx, _ = next(sss.split(torch.zeros(len(targets)), targets))
trainset = Subset(full_trainset, subset_idx)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

class TinyCNN(nn.Module):
    """BatchNorma and ReLU improve generalization."""
    def __init__(self, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

device = torch.device("mps" if torch.mps.is_available() else "cpu")
print(f"Using {str(device).upper()} for compute.")
net = TinyCNN(dropout=args.dropout).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in tqdm(range(args.epochs), desc="Epochs"):

    net.train()
    total_loss = 0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(trainloader)

    net.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
    test_accuracy = 100. * correct / len(testset)

    tqdm.write(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")