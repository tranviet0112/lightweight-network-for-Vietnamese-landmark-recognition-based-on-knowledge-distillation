from torchvision import datasets, models, transforms
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision

def data_loader(data_dir, batch_size = 20, valid_size = 0.2):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    # using random crops and horizontal flip for train set
    num_workers = 1
    train_transformer = transforms.Compose([
        transforms.Resize((224,224)), # anh chuan dau vao 224 x 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=train_transformer)

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
  


