import functools
import json
import os.path as osp
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import os
import torch
from numpy.linalg import norm
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

PATH = Path(__file__).parent


class CrystalGraphDataset(Dataset):
    def __init__(self, dataset_dir, dataset_ratio=1.0, radius=8.0, n_nbr=12):
        super().__init__()

        self.radius = radius
        self.n_nbr = n_nbr

        self.struc_dir = osp.join(dataset_dir, 'structures')
        self.ham_dir = osp.join(dataset_dir, 'hamiltonians')
        self.ids = os.listdir(self.struc_dir)

        self.ids = self.ids[:int(len(self.ids) * dataset_ratio)]

    def __len__(self):
        return len(self.ids)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        file = osp.join(self.struc_dir, self.ids[index])
        fid = self.ids[index].split('_')[1]

        with open(file, 'r') as f:
            cont = f.read()
        struc = Structure.from_str(cont, fmt='poscar')
        h_file = osp.join(self.ham_dir, 'hamiltonians_{}'.format(fid))
        hoppings = []
        with open(h_file, 'r') as f:
            for line in f:
                hoppings.append(float(line.split()[-2]))
        hoppings = np.array(hoppings)
        hoppings = np.around(hoppings, decimals=3)
        hoppings = torch.tensor(hoppings, dtype=torch.float)

        edge_index, edge_attr, lg_edge_index, lg_edge_attr, target, target_mask = self._get_edge_and_face(struc)
        target = hoppings[target]
        x = torch.tensor([s.specie.number - 1 for s in struc], dtype=torch.long)
        data = {'x': x, 'edge_index': edge_index, 'edge_attr': edge_attr, 'lg_edge_index': lg_edge_index,
                'lg_edge_attr': lg_edge_attr, 'target': target, 'target_mask': target_mask}
        return data, fid

    def _get_edge_and_face(self, struc):
        lg_edge_index, lg_edge_attr, edge_index, edge_attr, target, target_mask = [], [], [], [], [], []
        all_nbrs = [sorted(nbrs, key=lambda y: y.nn_distance)[:self.n_nbr] for nbrs in
                    struc.get_all_neighbors(self.radius)]

        lg_node_start_idx = 0
        for ctr_idx, nbrs in enumerate(all_nbrs):
            edge1, edge2 = [], []
            for i, (nbr1, nbr2) in enumerate(product(nbrs, nbrs)):
                f1 = struc.frac_coords[nbr1.index] - struc.frac_coords[ctr_idx] + nbr1.image
                f2 = struc.frac_coords[nbr2.index] - struc.frac_coords[ctr_idx] + nbr2.image
                m = struc.lattice.matrix
                edge1.append(f1 @ m)
                edge2.append(f2 @ m)

            edge1 = np.stack(edge1)
            edge2 = np.stack(edge2)
            cos = (edge1 * edge2).sum(-1) / (norm(edge1, axis=-1) * norm(edge2, axis=-1))
            lg_edge_attr.append(cos)

            for nbr in nbrs:
                edge_index.append([ctr_idx, nbr.index])
                edge_attr.append([nbr.nn_distance, * (nbr.coords - struc[ctr_idx].coords) / nbr.nn_distance])
                rows, mask = get_orb_rows(ctr_idx, nbr.index, nbr.image)
                target.append(rows)
                target_mask.append(mask)

            lg_node_idx = range(lg_node_start_idx, lg_node_start_idx + len(nbrs))
            lg_edge_index.extend(list(product(lg_node_idx, lg_node_idx)))

            lg_node_start_idx += len(nbrs)

        edge_index = np.array(edge_index).transpose()
        edge_attr = np.array(edge_attr)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        lg_edge_index = np.array(lg_edge_index).transpose()
        lg_edge_index = torch.tensor(lg_edge_index, dtype=torch.long)
        lg_edge_attr = np.hstack(lg_edge_attr)
        lg_edge_attr = torch.tensor(lg_edge_attr, dtype=torch.float)

        target = np.stack(target)
        target = torch.tensor(target, dtype=torch.long)
        target_mask = np.stack(target_mask)
        target_mask = torch.tensor(target_mask, dtype=torch.long)
        return edge_index, edge_attr, lg_edge_index, lg_edge_attr, target, target_mask


class CrystalGraphPreprocessedDataset(Dataset):
    def __init__(self, dataset_dir, dataset_ratio=1.0):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.ids = os.listdir(dataset_dir)
        self.ids = self.ids[:int(len(self.ids) * dataset_ratio)]

    def __len__(self):
        return len(self.ids)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        file = osp.join(self.dataset_dir, self.ids[index])

        data_dict = torch.load(file)
        data_dict['lg_edge_index_batch'] = data_dict.pop('lg_edge_index')
        return Data.from_dict(data_dict)


def get_orb_idx(i):
    if i < 32:
        return list(range(i * 5, i * 5 + 5))
    else:
        return list(range(160 + (i - 32) * 3, 160 + (i - 32) * 3 + 3))


IMAGE_IDX = {(-2, -2, 0): 0, (-2, -1, 0): 1, (-2, 0, 0): 2, (-1, -2, 0): 3, (-1, -1, 0): 4, (-1, 0, 0): 5,
             (-1, 1, 0): 6, (0, -2, 0): 7, (0, -1, 0): 8, (0, 0, 0): 9, (0, 1, 0): 10, (0, 2, 0): 11, (1, -1, 0): 12,
             (1, 0, 0): 13, (1, 1, 0): 14, (1, 2, 0): 15, (2, 0, 0): 16, (2, 1, 0): 17, (2, 2, 0): 18}


def get_orb_rows(ctr_idx, nbr_idx, image):
    indices = np.zeros(64, dtype=int)
    mask = np.zeros(64, dtype=int)
    if image in IMAGE_IDX:
        image_idx = IMAGE_IDX[image] * 352 * 352
        ctr_indices = get_orb_idx(ctr_idx)
        nbr_indices = get_orb_idx(nbr_idx)
        if len(ctr_indices) == 3:
            if len(nbr_indices) == 3:
                start = 0
            else:
                start = 9
        else:
            if len(nbr_indices) == 3:
                start = 24
            else:
                start = 39
        for i, (c, n) in enumerate(product(ctr_indices, nbr_indices)):
            idx = image_idx + c + n * 352
            indices[start + i] = idx
            mask[start + i] = 1

    return indices, mask


if __name__ == '__main__':
    dataset = CrystalGraphDataset('../dataset/MoS2_96', n_nbr=18)
    N = None


    def process_data(idx):
        d, fid = dataset[idx]
        torch.save(d, '../dataset/processed_dataset/{}.pt'.format(fid))


    if N is None:
        N = len(dataset)

    pool = Pool(processes=12)

    for _ in tqdm(pool.imap_unordered(process_data, range(N)), total=N):
        pass
