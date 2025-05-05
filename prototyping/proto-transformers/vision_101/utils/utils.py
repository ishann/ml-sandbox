import os
import torch


def compute_accuracy(outputs, labels):
    """
    Computes the classification accuracy given model outputs and ground-truth labels.

    Args:
        outputs (Tensor): Raw model outputs (logits), shape (B, num_classes).
        labels (Tensor): Ground-truth labels, shape (B,).

    Returns:
        float: Accuracy percentage (0-100).
    """
    # Get predicted class indices by taking argmax over class dimension
    _, preds = outputs.max(1)

    # Compare predictions to true labels and compute mean accuracy
    return (preds == labels).float().mean().item() * 100


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    """
    Saves a training checkpoint including model and optimizer state dicts.

    Args:
        model (nn.Module): The PyTorch model to save.
        optimizer (Optimizer): The optimizer used for training.
        epoch (int): Current epoch number (0-indexed).
        checkpoint_dir (str): Directory path to save the checkpoint.
    """
    # Ensure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Construct the checkpoint filename with epoch number (1-indexed)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")

    # Save model and optimizer state dicts, and epoch number
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
