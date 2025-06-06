from time import time as tick
from utils.data_loader import fetch_data_loaders
from utils.train import train
from utils.config import Config
import torch

def main():

    start_time = tick()

    # Config and setup.
    config_start_time = tick()
    config = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device={device} for tensor manipulation.")
    print(f"Config and device setup took {tick() - config_start_time:.2f} seconds.")

    # Dataloading.
    data_loading_start_time = tick()
    print("Setting up the dataloaders.")
    trainloader, testloader, num_classes = fetch_data_loaders(
        config.dataset, config.batch_size
    )
    print(f"Data loading took {tick() - data_loading_start_time:.2f} seconds.")

    # Model setup.
    model_setup_start_time = tick()
    print("Train.")

    if config.model=="ResNet18":
        from models.resnet import ResNet18
        model = ResNet18().to(device)
    elif config.model=="TinyViT":
        from models.vit import VisionTransformer
        model = VisionTransformer(num_classes=num_classes).to(device)
    print(f"Model setup took {tick() - model_setup_start_time:.2f} seconds.")

    # Learn.
    training_start_time = tick()
    train(model, trainloader, testloader, config, device)
    print(f"Training took {tick() - training_start_time:.2f} seconds.")

    print(f"Total execution time: {tick() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
