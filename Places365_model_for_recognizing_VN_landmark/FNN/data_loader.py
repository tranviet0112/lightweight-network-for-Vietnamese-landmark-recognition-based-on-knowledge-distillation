from torchvision import datasets, models, transforms
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

def data_loader(data_dir, batch_size = 20, valid_size = 0.2):
    
    def npy_loader(img_path):
        sample = torch.from_numpy(np.load(img_path))
        return sample
    dataset = datasets.DatasetFolder(
                                    root=data_dir,
                                    loader = npy_loader,
                                    extensions=('.npy'))
    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load

    num_train = len(dataset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)

    return train_loader, valid_loader, dataset.classes

