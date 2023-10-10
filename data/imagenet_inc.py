import os
from os import path as osp
import sys
sys.path.append('.')
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import (Onehot, encode_onehot, query_transform,
                            train_transform)
import copy
from loguru import logger
num_classes = 100
def load_data(root, batch_size, num_workers=8, is_original=True, num_origin_classes=50, num_inc_classes=50):
    """
    Load imagenet dataset

    Args:
        root (str): Path of imagenet dataset.
        batch_size (int): Number of samples in one batch.
        workers (int): Number of data loading threads.

    Returns:
        train_loader (torch.utils.data.DataLoader): Training dataset loader.
        query_loader (torch.utils.data.DataLoader): Query dataset loader.
        val_loader (torch.utils.data.DataLoader): Validation dataset loader.
    """

    category_list = list(range(0, num_classes, 2)) + list(range(1, num_classes, 2))
    cate2target = {category_list[i]:i for i in range(len(category_list))}
    if is_original:
        category_list = category_list[:num_origin_classes]
    else:
        category_list = category_list[num_origin_classes:num_origin_classes+num_inc_classes]
    
    # Data transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    query_retrieval_init_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    # Construct data loader
    train_dir = os.path.join(root, 'train')
    query_dir = os.path.join(root, 'query')
    retrieval_dir = os.path.join(root, 'database')

    train_dataset = ImagenetDataset(
        train_dir,
        transform=train_transform,
        category_list=category_list,
        cate2target=cate2target,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    query_dataset = ImagenetDataset(
        query_dir,
        transform=query_retrieval_init_transform,
        category_list=category_list,
        cate2target=cate2target,
    )

    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    retrieval_dataset = ImagenetDataset(
        retrieval_dir,
        transform=query_retrieval_init_transform,
        category_list=category_list,
        cate2target=cate2target,
    )

    retrieval_loader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, query_loader, retrieval_loader

class ImagenetDataset(Dataset):
    classes = None
    class_to_idx = None

    def __init__(self, root, transform=None, target_transform=None, category_list=None, cate2target=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        self.category_list = category_list

        # Assume file alphabet order is the class order
        if ImagenetDataset.class_to_idx is None:
            ImagenetDataset.classes, ImagenetDataset.class_to_idx = self._find_classes(root)

        for i, cl in enumerate(ImagenetDataset.classes):
            if ImagenetDataset.class_to_idx[cl] in self.category_list:
                cur_class = os.path.join(self.root, cl)
                files = os.listdir(cur_class)
                files = [os.path.join(cur_class, i) for i in files]
                self.data.extend(files)
                self.targets.extend([cate2target[ImagenetDataset.class_to_idx[cl]] for i in range(len(files))])
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.onehot_targets = encode_onehot(self.targets, 100)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target, 100)
        return img, target, item

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def get_onehot_targets(self):
        '''
        Return one-hot encoding targets.
        '''
        return torch.from_numpy(self.onehot_targets).float()

    def add_memory(self, exem_data, exem_labels):

        tmp = copy.deepcopy(exem_data)
        tmp.extend(self.data.tolist())
        self.data = np.array(tmp)
        tmp = copy.deepcopy(exem_labels)
        tmp.extend(self.targets.tolist())
        self.targets = np.array(tmp)
        self.onehot_targets = encode_onehot(self.targets, 100)
        logger.info("##### [add memory] #####")
        cat_dict = {}
        for cat in exem_labels:
            if cat not in cat_dict:
                cat_dict[cat] = 0
            cat_dict[cat] += 1
        logger.info(cat_dict)
        logger.info("########################")