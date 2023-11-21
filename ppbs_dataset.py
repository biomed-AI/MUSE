#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Copyright (c) 2023, Sun Yat-sen Univeristy.
All rights reserved.

@author:   Jiahua Rao
@license:  BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact:  jiahua.rao@gmail.com
'''


import h5py
import Bio.PDB
import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.data import Data, InMemoryDataset

from utilities.ppbs_dataset_utils import read_labels, align_labels

import warnings
warnings.filterwarnings('ignore')


config_model = {
    "em": {'N0': 30, 'N1': 32},
    "sum": [
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
    ],
    "spl": {'N0': 32, 'N1': 32, 'Nh': 4},
    "dm": {'N0': 32, 'N1': 32, 'N2': 5}
}


list_datasets = [
    'train',
    'validation_70',
    'validation_homology',
    'validation_topology',
    'validation_none',
    'test_70',
    'test_homology',
    'test_topology',
    'test_none',
]



def collate_batch_features(batch_data, max_num_nn=64):
    # pack coordinates and charges
    X = torch.cat([data[0] for data in batch_data], dim=0)
    q = torch.cat([data[2] for data in batch_data], dim=0)

    # extract sizes
    sizes = torch.tensor([data[3].shape for data in batch_data])

    # pack nearest neighbors indices and residues masks
    ids_topk = torch.zeros((X.shape[0], max_num_nn), dtype=torch.long, device=X.device)
    M = torch.zeros(torch.Size(torch.sum(sizes, dim=0)), dtype=torch.float, device=X.device)
    for size, data in zip(torch.cumsum(sizes, dim=0), batch_data):
        # get indices of slice location
        ix1 = size[0]
        ix0 = ix1-data[3].shape[0]
        iy1 = size[1]
        iy0 = iy1-data[3].shape[1]
        # store data
        ids_topk[ix0:ix1, :data[1].shape[1]] = data[1]+ix0+1
        M[ix0:ix1,iy0:iy1] = data[3]

    return X, ids_topk, q, M


def collate_batch_data(batch_data):
    # collate sids
    sample_ids = [data[0] for data in batch_data]
    batch_data = [data[1:] for data in batch_data]

    # collate features
    X, ids_topk, q, M = collate_batch_features(batch_data)

    # collate targets
    y = torch.cat([data[4] for data in batch_data])

    X = torch.nan_to_num(X, nan=0)
    ids_topk = torch.nan_to_num(ids_topk, nan=0)
    q = torch.nan_to_num(q, nan=0)
    M = torch.nan_to_num(M, nan=0)
    y = torch.nan_to_num(y, nan=0)
    return sample_ids, X, ids_topk, q, M, y


class ProteinAtomDataset(InMemoryDataset):
    def __init__(self, args, config, split, **kwargs):
        self.args = args
        self.config = config
        self.kwargs = kwargs
        self.split = split

        self.inter_dataset_root = args.inter_data_dir
        self.dataset_name = list(self.config.inter_dataset_config.keys())[0]
        self.inter_dataset_config = self.config.inter_dataset_config[self.dataset_name]

        super().__init__(root=os.path.join(self.inter_dataset_root, self.dataset_name.replace('-', '_')))
        split_idx = list_datasets.index(self.split)
        self.protein_datasets = torch.load(self.processed_paths[split_idx])

    def __getitem__(self, idx):
        return self.protein_datasets[idx]

    def __len__(self):
        return len(self.protein_datasets)

    def collate_fn(self, data_list):
        return collate_batch_data(data_list)

    @property
    def processed_file_names(self):
        return [f"{dataset}_protein_structures.pt" for dataset in list_datasets]

    def process(self):
        return super().process()


class ProteinResidueDataset(InMemoryDataset):
    def __init__(self, args, config, split, embeddings=None, **kwargs):
        self.args = args
        self.config = config
        self.kwargs = kwargs
        self.split = split
        self.embeddings = embeddings

        self.inter_dataset_root = args.inter_data_dir
        self.dataset_name = list(self.config.inter_dataset_config.keys())[0]
        self.inter_dataset_config = self.config.inter_dataset_config[self.dataset_name]

        super().__init__(root=os.path.join(self.inter_dataset_root, self.dataset_name.replace('-', '_')))
        self.protein_datasets = torch.load(self.processed_paths[0])
        self.ppi_proteins = torch.load(self.processed_paths[1])
        self.ppi_pairs = torch.load(self.processed_paths[2])
    

        if embeddings is None:
            embeddings = [torch.randn(p.x.shape[0], 64) for p in self.protein_datasets]
        else:
            embeddings = self.embeddings[self.split]

        protein_dataset = []
        for idx, data in enumerate(self.protein_datasets):
            data.node_feat = embeddings[idx].cpu()
            protein_dataset.append(data)
        self.protein_datasets = protein_dataset

    def __getitem__(self, idx):
        if len(self.ppi_pairs[idx]) > 0:
            pos_ppi = random.sample(self.ppi_pairs[idx], 1)[0]
            return self.protein_datasets[idx], (self.ppi_proteins[pos_ppi[0]], self.ppi_proteins[pos_ppi[1]])
        else:
            return self.protein_datasets[idx], (None, None)

    def __len__(self):
        return len(self.protein_datasets)

    def collate_fn(self, data_list):
        graph_data_list = [data[0] for data in data_list]
        ppi_src_graphs = [data[1][0] for data in data_list if data[1][0] is not None]
        ppi_dst_graphs = [data[1][1] for data in data_list if data[1][1] is not None]
        return Batch.from_data_list(graph_data_list), \
               Batch.from_data_list(ppi_src_graphs), \
               Batch.from_data_list(ppi_dst_graphs)

    @property
    def processed_file_names(self):
        split_idx = list_datasets.index(self.split)
        dataset = list_datasets[split_idx]
        return [f"{dataset}_protein_graphs.pt", f"{dataset}_ppi_graphs.pt", f"{dataset}_ppi_pairs.pt"]

    def process(self):
        return super().process()
    

class PPIDataset(InMemoryDataset):
    def __init__(self, args, config, split, embeddings, **kwargs):
        self.args = args
        self.config = config
        self.kwargs = kwargs
        self.split = split
    
        self.inter_dataset_root = args.inter_data_dir
        self.dataset_name = list(self.config.inter_dataset_config.keys())[0]
        self.inter_dataset_config = self.config.inter_dataset_config[self.dataset_name]

        super().__init__(root=os.path.join(self.inter_dataset_root, self.dataset_name.replace('-', '_')))
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return os.path.join('ppi_data_processed.pt')

    @property
    def raw_file_names(self):
        return [f'train_ppi_pairs.pt']
    
    def process(self):
        add_inverse_edge = self.config.get('add_inverse_edge', True)

        # loading necessary files
        try:
            edge = pd.read_csv(os.path.join(self.raw_dir, 'edge.csv.gz'), compression='gzip', header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
            num_node_list = pd.read_csv(os.path.join(self.raw_dir, 'num-node-list.csv.gz'), compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list

        except FileNotFoundError:
            raise RuntimeError('No necessary file')

        print('Processing graphs...')
        graph_list = []
        num_node = num_node_list[0]

        graph = dict()
        if add_inverse_edge:
            ### duplicate edge
            duplicated_edge = np.repeat(edge, 2, axis = 1)
            duplicated_edge[0, 1::2] = duplicated_edge[1,0::2]
            duplicated_edge[1, 1::2] = duplicated_edge[0,0::2]
            graph['edge_index'] = duplicated_edge

        else:
            graph['edge_index'] = edge

        graph['num_nodes'] = num_node
        graph_list.append(graph)

        print('Converting graphs into PyG objects...')
        data = []
        for graph in tqdm(graph_list):
            g = Data()
            g.num_nodes = graph['num_nodes']
            g.edge_index = torch.from_numpy(graph['edge_index'])

            del graph['num_nodes']
            del graph['edge_index']

            if graph['node_feat'] is not None:
                g.x = torch.from_numpy(graph['node_feat'])
                del graph['node_feat']

            if graph['edge_label'] is not None:
                g.edge_attr = torch.from_numpy(graph['edge_label'])
                del graph['edge_label']
            data.append(g)

        data = data[0]
        data = data if self.pre_transform is None else self.pre_transform(data)

        print('Saving...')
        torch.save(self.collate([data]), self.processed_paths[0])