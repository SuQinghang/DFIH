import sys
import torch

sys.path.append('')
import os

from PIL import Image, ImageFile
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import encode_onehot, query_transform, train_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True

def sample_dataloader(dataloader, num_samples, batch_size, root, dataset):
    """
    Sample data from dataloder.

    Args
        dataloader (torch.utils.data.DataLoader): Dataloader.
        num_samples (int): Number of samples.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        sample_index (int): Sample index.
        dataset(str): Dataset name.

    Returns
        sample_dataloader (torch.utils.data.DataLoader): Sample dataloader.
    """
    data = dataloader.dataset.data
    targets = dataloader.dataset.targets

    
    if isinstance(data, list):
        size = len(data)
    else:
        size = data.shape[0]
    sample_index = torch.randperm(size)[:num_samples]
    # import ipdb;ipdb.set_trace()
    # sample_index = torch.randperm(5000)[:num_samples]
    data = data[sample_index]
    targets = targets[sample_index]
    sample = wrap_data(data, targets, batch_size, root, dataset)

    return sample, sample_index

def wrap_data(data, targets, batch_size, root, dataset):
    """
    Wrap data into dataloader.

    Args
        data (np.ndarray): Data.
        targets (np.ndarray): Targets.
        batch_size (int): Batch size.
        root (str): Path of dataset.
        dataset(str): Dataset name.

    Returns
        dataloader (torch.utils.data.dataloader): Data loader.
    """
    class MyDataset(Dataset):
        def __init__(self, data, targets, root, dataset):
            self.data = data
            self.targets = targets
            self.root = root
            self.transform = train_transform()
            self.dataset = dataset
            if self.dataset == 'cifar-100' or self.dataset == 'imagenet':
                self.onehot_targets = encode_onehot(self.targets, 100)
            elif self.dataset == 'cifar-10' or self.dataset=='svhn':
                self.onehot_targets = encode_onehot(self.targets, 10)
            elif self.dataset == 'tinyimage200_g':
                self.onehot_targets = encode_onehot(self.targets, 200)
            else:
                self.onehot_targets = self.targets

        def __getitem__(self, index):
            img = Image.open(os.path.join(self.root, self.data[index])).convert('RGB')
            img = self.transform(img)
            return img, self.targets[index], index

        def __len__(self):
            return self.data.shape[0]

        def get_onehot_targets(self):
            """
            Return one-hot encoding targets.
            """
            return torch.from_numpy(self.onehot_targets).float()

    dataset = MyDataset(data, targets, root, dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    return dataloader

# if __name__=='__main__':
#     train_dataloader, query_dataloader, retrieval_dataloader = load_data(
#         dataset='cifar-100-inc',
#         root = '/data2/suqinghang/Dataset/cifar-100',
#         batch_size=128,
#         num_workers=8,
#         is_original=True,
#         num_origin_classes=50
#     )
#     print('len of train: ', len(train_dataloader.dataset))
#     print('len of query: ', len(query_dataloader.dataset))
#     print('len of retrieval: ', len(retrieval_dataloader.dataset))
