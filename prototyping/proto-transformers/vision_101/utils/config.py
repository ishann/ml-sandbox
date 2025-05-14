import argparse

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Vision Transformer Training")
        parser.add_argument(
            "--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"]
        )
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument(
            "--checkpoint_dir", type=str, default="./outputs/checkpoints"
        )
        args = parser.parse_args()

        # Set class attributes from the parsed arguments
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.lr = args.lr
        self.checkpoint_dir = args.checkpoint_dir
        self.num_workers=args.num_workers

        # Print the config values to stdout
        print("Configuration:")
        print(f"Dataset: {self.dataset}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of workers: {self.num_workers}")
        print(f"Epochs: {self.epochs}")
        print(f"Learning rate: {self.lr}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
