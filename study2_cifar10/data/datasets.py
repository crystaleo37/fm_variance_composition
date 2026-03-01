"""Data loaders for CIFAR-10 and ImageNet-32.

All images normalized to [-1, 1] for flow matching.
"""

import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

# On Windows, multiprocessing uses spawn which can't pickle lambdas.
# On Colab, only 2 CPU cores available.
_DEFAULT_WORKERS = 0 if os.name == "nt" else 2
_PIN_MEMORY = torch.cuda.is_available()


def _build_transform(train: bool) -> T.Compose:
    """Build image transform pipeline (picklable — no lambdas)."""
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip())
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # [0,1] -> [-1,1]
    return T.Compose(transforms)


def get_cifar10_loader(
    batch_size: int = 128,
    root: str = "./data/cifar10",
    train: bool = True,
    num_workers: int = _DEFAULT_WORKERS,
) -> DataLoader:
    """CIFAR-10 dataloader with images in [-1, 1] (standard for flow matching)."""
    dataset = torchvision.datasets.CIFAR10(
        root=root, train=train, download=True, transform=_build_transform(train),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=_PIN_MEMORY,
        drop_last=train,
    )


def get_imagenet32_loader(
    batch_size: int = 256,
    root: str = "./data/imagenet32",
    train: bool = True,
    num_workers: int = _DEFAULT_WORKERS,
) -> DataLoader:
    """ImageNet-32 dataloader with images in [-1, 1].

    Expects ImageNet-32x32 stored as an ImageFolder at root/train and root/val.
    Download from https://image-net.org/download-images (downsampled 32x32).
    """
    split = "train" if train else "val"
    dataset = torchvision.datasets.ImageFolder(
        root=f"{root}/{split}", transform=_build_transform(train),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=_PIN_MEMORY,
        drop_last=train,
    )


def get_dataloader(
    dataset: str,
    batch_size: int,
    train: bool = True,
    num_workers: int = _DEFAULT_WORKERS,
    data_root: str = "./data",
) -> DataLoader:
    """Dispatch to the appropriate dataset loader."""
    if dataset == "cifar10":
        return get_cifar10_loader(batch_size, f"{data_root}/cifar10", train, num_workers)
    elif dataset == "imagenet32":
        return get_imagenet32_loader(batch_size, f"{data_root}/imagenet32", train, num_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
