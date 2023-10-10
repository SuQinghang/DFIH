import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import DataLoader

def random(dataset, memory_size):
    gallery = deepcopy(dataset)

    # Collect exemplars
    cat_dict = {}
    for dat, tar in zip(gallery.data, gallery.targets):
        tar = int(tar)
        if tar not in cat_dict:
            cat_dict[tar] = []
        cat_dict[tar].append(dat)
    num_class = len(cat_dict.keys())
    mem_per_cls = memory_size // num_class
    exemplar_set = {}
    for class_id in cat_dict:
        current_images = np.array(cat_dict[class_id])
        sample_index = np.random.choice(len(current_images), mem_per_cls, replace=False)
        exemplar_set[class_id] = list(current_images[sample_index])
    return exemplar_set