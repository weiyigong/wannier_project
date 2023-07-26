import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import random_split


def get_train_test_dataloader(dataset, batch_size, train_ratio, seed, num_workers, pin_memory):
    n_data = len(dataset)
    train_split = int(n_data * train_ratio)

    dataset_train, dataset_val = random_split(
        dataset,
        [train_split, len(dataset) - train_split],
        generator=torch.Generator().manual_seed(seed)
    )

    kw = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': pin_memory}

    return DataLoader(dataset_train, **kw), DataLoader(dataset_val, **kw)
