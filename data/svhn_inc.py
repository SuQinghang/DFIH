'''
Copy from ShuZhao
Use fixed train, query and retrieval data
'''
import sys
sys.path.append('')
import os
import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from data.transform import train_transform, query_transform, Onehot, encode_onehot
from PIL import Image
import copy
from loguru import logger

# ori_category_list = [0, 1, 2, 3, 4, 5, 6, 9]
# inc_category_list = [7, 8]
num_classes = 10
def load_data(root, batch_size=64, num_workers=8, is_original=True, num_origin_classes=5, num_inc_classes=5):
    """
    Load cifar-10 dataset.

    Args
        root(str): Path of dataset.
        batch_size(int): Batch size.
        num_workers(int): Number of data loading workers.

    Returns
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """

    ori_category_list = list(range(10))[:num_origin_classes]
    inc_category_list = list(range(10))[num_origin_classes:num_origin_classes + num_inc_classes]
    if is_original:
        category_list = ori_category_list
    else:
        category_list = inc_category_list
    
    train_dataloader = DataLoader(
        ImagenetDataset(
            root=os.path.join(root, 'train'),
            transform=train_transform(),
            category_list=category_list,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    query_dataloader = DataLoader(
        ImagenetDataset(
            root=os.path.join(root, 'query'),
            transform=query_transform(),
            category_list=category_list,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    retrieval_dataloader = DataLoader(
        ImagenetDataset(
            root=os.path.join(root, 'retrieval'),
            transform=query_transform(),
            category_list=category_list,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, query_dataloader, retrieval_dataloader,


class ImagenetDataset(Dataset):
    classes = None
    class_to_idx = None

    def __init__(self, root, transform=None, target_transform=None, category_list=None):
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
            #! select cl from category_list
            idx = ImagenetDataset.class_to_idx[cl]
            if idx in self.category_list:
                cur_class = os.path.join(self.root, cl)
                files = os.listdir(cur_class)
                files = [os.path.join(cur_class, i) for i in files]
                self.data.extend(files)
                self.targets.extend([ImagenetDataset.class_to_idx[cl] for i in range(len(files))])
        #! set data to nparray
        self.data = np.array(self.data)
        self.targets = np.asarray(self.targets)
        self.onehot_targets = encode_onehot(self.targets, num_classes)
    
    def get_onehot_targets(self):
        return torch.from_numpy(self.onehot_targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
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

    def add_memory(self, exem_data, exem_labels):

        tmp = copy.deepcopy(exem_data)
        tmp.extend(self.data.tolist())
        self.data = np.array(tmp)
        tmp = copy.deepcopy(exem_labels)
        tmp.extend(self.targets.tolist())
        self.targets = np.array(tmp)
        self.onehot_targets = encode_onehot(self.targets, num_classes)
        logger.info("##### [add memory] #####")
        cat_dict = {}
        for cat in exem_labels:
            if cat not in cat_dict:
                cat_dict[cat] = 0
            cat_dict[cat] += 1
        logger.info(cat_dict)
        logger.info("########################")