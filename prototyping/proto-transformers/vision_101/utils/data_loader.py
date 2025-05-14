import torch
import torchvision
import torchvision.transforms as transforms

def fetch_data_loaders(dataset='cifar10', batch_size=64, num_workers=0):
    """
    Returns PyTorch DataLoaders for CIFAR-10 or CIFAR-100 datasets with standard preprocessing.

    Applies common data augmentation to the training set (random cropping and horizontal flipping)
    and normalization to both train and test sets.

    Args:
        dataset (str): Dataset to load, either 'cifar10' or 'cifar100' (case-insensitive).
        batch_size (int): Number of samples per batch in the DataLoader.

    Returns:
        trainloader (DataLoader): DataLoader for the training set.
        testloader (DataLoader): DataLoader for the test set.
        num_classes (int): Number of classes in the dataset (10 for CIFAR-10, 100 for CIFAR-100).
    
    NOTES:
    1. macOS uses the spawn method for multiprocessing by default (not fork).
       This leads to issues with PyTorch's multiprocessing.
       PyTorch disables num_workers>0 on MPS backend intentionally due to known instability/crashes.
    2. Setting num_workers=0 is faster than num_workers=1.
       On macOS, spawning a process for a single worker is costly.
       That process has to serialize the dataset object, potentially triggering overheads with
       shared memory and inter-process communication. Since PyTorch’s multiprocessing isn't
       fully compatible with MPS/macOS, performance may degrade or even silently fail. The slowdown is
       likely due to the cost of starting the subprocess + serialization overhead with no real gain.
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset.lower() == 'cifar100':
        DatasetClass = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        DatasetClass = torchvision.datasets.CIFAR10
        num_classes = 10

    trainset = DatasetClass(
        root="./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testset = DatasetClass(
        root="./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader, num_classes
