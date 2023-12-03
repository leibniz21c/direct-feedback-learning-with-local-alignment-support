import os
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import v2
from datasets import load_dataset


# CONSTANTS
_MNIST_TRAIN_NORMAL = 0.15, 0.3
_MNIST_TEST_NORMAL = 0.15, 0.3

_CIFAR10_TRAIN_NORMAL = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
_CIFAR10_TEST_NORMAL = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)

_CIFAR100_TRAIN_NORMAL = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
_CIFAR100_TEST_NORMAL = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)

_TINYIMAGENET_TRAIN_NORMAL = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
_TINYIMAGENET_TEST_NORMAL = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)


class TinyImageNet(Dataset):
    def __init__(self, train: bool = True, transform=None):
        if train:
            self._ds = load_dataset('Maysee/tiny-imagenet', split='train')
        else:
            self._ds = load_dataset('Maysee/tiny-imagenet', split='valid')
        self.transform = transform


    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self._ds[index]['image'].convert('RGB')), self._ds[index]['label']
        return self._ds[index]['image'].convert('RGB'), self._ds[index]['label']

    
    def __len__(self):
        return len(self._ds)
    

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def get_lr_decays(num_epochs: int, decays: tuple = (.5, .75, .89, .94)):
    return (
        int(num_epochs * 0.5), 
        int(num_epochs * 0.75),
        int(num_epochs * 0.89),
        int(num_epochs * 0.94),
    )


def get_loaders(dataset, root, batch_size, shuffle, num_workers):
    if dataset == 'MNIST':
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=_MNIST_TRAIN_NORMAL[0], 
                std=_MNIST_TRAIN_NORMAL[1],
            ),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=_MNIST_TEST_NORMAL[0], 
                std=_MNIST_TEST_NORMAL[1],
            ),
        ])
    elif dataset == 'CIFAR10':
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=_CIFAR10_TRAIN_NORMAL[0], 
                std=_CIFAR10_TRAIN_NORMAL[1],
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=_CIFAR10_TEST_NORMAL[0], 
                std=_CIFAR10_TEST_NORMAL[1],
            ),
        ])
    elif dataset == 'CIFAR100':
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            v2.RandAugment(),
            transforms.Normalize(
                mean=_CIFAR100_TRAIN_NORMAL[0], 
                std=_CIFAR100_TRAIN_NORMAL[1],
            ),
            Cutout(n_holes=1, length=14),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=_CIFAR100_TEST_NORMAL[0], 
                std=_CIFAR100_TEST_NORMAL[1],
            ),
        ])
    elif dataset == 'TINYIMAGENET':
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            v2.RandAugment(),
            transforms.Normalize(
                mean=_TINYIMAGENET_TRAIN_NORMAL[0], 
                std=_TINYIMAGENET_TRAIN_NORMAL[1],
            ),
            Cutout(n_holes=1, length=14),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=_TINYIMAGENET_TEST_NORMAL[0], 
                std=_TINYIMAGENET_TEST_NORMAL[1],
            ),
        ])
    else:
        raise NotImplementedError(f"{dataset} is not implemented.")
    
    if dataset == 'TINYIMAGENET':
        return (
            DataLoader(
                dataset=TinyImageNet(train=True, transform=train_transforms),
                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
            ),
            DataLoader(
                dataset=TinyImageNet(train=False, transform=test_transforms),
                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
            ),
        )
    return (
        DataLoader(
            dataset=getattr(datasets, dataset)(root=root, train=True, download=True, transform=train_transforms),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
        ),
        DataLoader(
            dataset=getattr(datasets, dataset)(root=root, train=False, download=True, transform=test_transforms),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
        ),
    )


def get_logger(result_path, exist_ok=False):
    os.makedirs(f"{result_path}", exist_ok=exist_ok)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s - %(levelname)s] : %(message)s")
    file_handler = logging.FileHandler(f"{result_path}/train_log.log")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_device(seed: int):
    device = torch.device(f"cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    return device


def logging_vars(logger, **kwargs):
    logger.info("VARIABLES : ")
    for key in kwargs:
        logger.info(f"\t{key} : {kwargs[key]}")