import time
from models.vit import VisionTransformer
from utils.data_loader import get_data_loaders
from utils.train import train
from utils.config import Config

import torch

from ipdb import set_trace as st


def main():
    # Start overall timing for the script
    start_time = time.time()

    # Timing for config and device setup
    config_start_time = time.time()
    config = Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} as the device for tensor manipulation.")
    print(f"Config and device setup took {time.time() - config_start_time:.2f} seconds.")

    # Timing for data loading
    data_loading_start_time = time.time()
    print("Setting up the dataloaders.")
    trainloader, testloader, num_classes = get_data_loaders(
        config.dataset, config.batch_size
    )
    print(f"Data loading took {time.time() - data_loading_start_time:.2f} seconds.")

    # Timing for model setup
    model_setup_start_time = time.time()
    print("Train.")
    model = VisionTransformer(num_classes=num_classes).to(device)
    print(f"Model setup took {time.time() - model_setup_start_time:.2f} seconds.")

    # Timing for training
    training_start_time = time.time()
    train(model, trainloader, testloader, config, device)
    print(f"Training took {time.time() - training_start_time:.2f} seconds.")

    # Overall script time
    print(f"Total script execution time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
