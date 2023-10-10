import os
import sys
sys.path.append('')
import pickle
import numpy as np
import torch
from PIL import Image
from skimage import io
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import (Onehot, encode_onehot, query_transform,
                            train_transform)
import copy
from loguru import logger

num_classes = 100

def save_imgs(imgs, label, root, mode='query'):
    imgs_path = os.path.join(root, mode, str(label))
    if (not os.path.exists(imgs_path)):
        os.makedirs(imgs_path)

    for i in range(imgs.shape[0]):
        r = imgs[i, :1024].reshape(32, 32)
        g = imgs[i, 1024:2048].reshape(32, 32)
        b = imgs[i, 2048:3072].reshape(32, 32)
        img = np.dstack((r, g, b))

        img_path = os.path.join(imgs_path, '{}_{}_{}.png'.format(mode, label, i))
        io.imsave(img_path, img)

def split_train_query_retrieval():
    root = '/data2/suqinghang/Dataset/cifar-100/cifar-100-python'


    with open(os.path.join(root, 'train-data'), 'rb') as cifar100_train:
        train_data = pickle.load(cifar100_train, encoding='bytes')

    train_imgs = train_data['data'.encode()]
    train_labels = np.array(train_data['fine_labels'.encode()])

    with open(os.path.join(root, 'test-data'), 'rb') as cifar100_test:
        test_data = pickle.load(cifar100_test, encoding='bytes')

    test_imgs = test_data['data'.encode()]
    test_labels = np.array(test_data['fine_labels'.encode()])

    save_path = '/data2/suqinghang/Dataset/cifar-100'
    for l in range(100):
        query_imgs_i = test_imgs[test_labels==l]
        query_labels_i = test_labels[test_labels==l]

        train_and_database_imgs_i = train_imgs[train_labels==l]
        train_and_database_labels_i = train_labels[train_labels==l]
        save_imgs(query_imgs_i, l, save_path, mode='query')
        save_imgs(train_and_database_imgs_i, l, save_path, mode='retrieval')
        # save_imgs(itrain_and_database_imgs_i, labels_i[retrieval_index], save_path, mode='retrieval')

def load_data(root, batch_size=64, num_workers=8, is_original=True, num_origin_classes=50, num_inc_classes=50):
    """
    Load cifar-10 dataset.

    Args
        root(str): Path of dataset.
        batch_size(int): Batch size.
        num_workers(int): Number of data loading workers.

    Returns
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """

    category_list = list(range(0, num_classes, 2)) + list(range(1, num_classes, 2))
    cate2target = {category_list[i]:i for i in range(len(category_list))}
    if is_original:
        category_list = category_list[:num_origin_classes]
    else:
        category_list = category_list[num_origin_classes:num_origin_classes+num_inc_classes]
    
    train_dataloader = DataLoader(
        ImagenetDataset(
            root=os.path.join(root, 'retrieval'),
            transform=train_transform(),
            category_list=category_list,
            cate2target=cate2target,
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
            cate2target=cate2target,
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
            cate2target=cate2target,
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
            #! select cl from category_list
            idx = ImagenetDataset.class_to_idx[cl]
            if idx in self.category_list:
                cur_class = os.path.join(self.root, cl)
                files = os.listdir(cur_class)
                files = [os.path.join(cur_class, i) for i in files]
                self.data.extend(files)
                self.targets.extend([cate2target[ImagenetDataset.class_to_idx[cl]] for i in range(len(files))])
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
            target = self.target_transform(target, num_classes)
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