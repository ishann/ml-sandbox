import torch
from utils.utils import save_checkpoint
from tqdm import tqdm

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    # Wrap the loader with tqdm to show a progress bar
    with tqdm(loader, desc="Training", unit="batch") as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # Update the progress bar with the current loss
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))
    return running_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100. * correct / total

def train(model, trainloader, testloader, config, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print("Begin training.")
    for epoch in range(config.epochs):
        loss = train_one_epoch(model, trainloader, criterion, optimizer, device)
        acc = evaluate(model, testloader, device)
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {loss:.4f}, Test Acc: {acc:.2f}%")
        #save_checkpoint(model, optimizer, epoch, config.checkpoint_dir)
