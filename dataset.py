#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author:  Jiahua Rao
@license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact: jiahua.rao@gmail.com
@time:    05/2023
'''


import os, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import AllChem
from gensim.models import Word2Vec, word2vec


import torch
from torch.utils.data import Dataset
from torch_sparse import SparseTensor
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.data import Data, InMemoryDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset

from utils import filter_invalid_smiles, replace_numpy_with_torchtensor, ppi_split_dataset
from utils import get_mol_features, get_mol2vec_features, molecule_collate_fn, protein_collate_fn
from features import mol_to_graph_data_obj


class MoleculeDataset(InMemoryDataset):
    r"""
    Molecule Dataset
    """
    def __init__(self, args, config, split, **kwargs):
        self.args = args
        self.config = config
        self.kwargs = kwargs
        self.split = split

        self.inter_dataset_root = args.inter_data_dir
        self.dataset_name = list(self.config.inter_dataset_config.keys())[0]
        self.inter_dataset_config = self.config.inter_dataset_config[self.dataset_name]
        self.add_inverse_edge = self.inter_dataset_config.get('add_inverse_edge', True)

        super(MoleculeDataset, self).__init__(root=os.path.join(self.inter_dataset_root, self.dataset_name.replace('-', '_')))
        self.data = torch.load(self.processed_paths[0])
        self.molecule_dataset = torch.load(self.processed_paths[1])

        self.splits = self.get_edge_idx_split()
        if self.add_inverse_edge and self.split == 'train':
            self.edges = torch.cat((self.data['edge_index'][:, self.splits[split]], self.data['edge_index'][:, self.splits[split]][[1, 0]]), dim=1).t()
            self.labels = torch.cat((self.data['edge_label'][self.splits[split]], self.data['edge_label'][self.splits[split]]), dim=0)
        else:
            self.edges = self.data['edge_index'][:, self.splits[split]].t()
            self.labels = self.data['edge_label'][self.splits[split]]

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index):
        row, col = self.edges[index]
        protein_row, protein_col = self.molecule_dataset[row], self.molecule_dataset[col]
        return {
            "edge": self.edges[index],
            "mol1_data": protein_row,
            "mol2_data": protein_col,
            "label": self.labels[index],
        }
    
    def collate_fn(self, batch):
        edges = torch.stack([torch.LongTensor(data["edge"]) for data in batch], dim=0)
        molecule1_data = Batch.from_data_list([data["mol1_data"] for data in batch])
        molecule2_data = Batch.from_data_list([data["mol2_data"] for data in batch])
        labels = torch.stack([torch.LongTensor(data["label"]) for data in batch],dim=0).to(dtype=torch.long)
        return {
            "edge": edges,
            "molecule1_data": molecule1_data,
            "molecule2_data": molecule2_data,
            "label": labels,
        }

    def get_edge_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.inter_dataset_config.split

        path = os.path.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(os.path.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
        test_idx = pd.read_csv(os.path.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]

        if os.path.exists(os.path.join(path, 'valid.csv.gz')):
            valid_idx = pd.read_csv(os.path.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
            return {'train': torch.LongTensor(train_idx), 'valid': torch.LongTensor(valid_idx), 'test': torch.LongTensor(test_idx)}
        else:
            return {'train': torch.LongTensor(train_idx), 'test': torch.LongTensor(test_idx)}

    @property
    def processed_file_names(self):
        return ['ddi_data_processed.pt', 'molecule_data_processed.pt']

    @property
    def raw_file_names(self):
        file_names = ['edge']
        return [file_name + '.csv.gz' for file_name in file_names]

    def process(self):
        # add inverse edges
        add_inverse_edge = self.add_inverse_edge

        # loading necessary files
        try:
            all_edge = pd.read_csv(os.path.join(self.raw_dir, 'all-edge.csv.gz'), compression='gzip', header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
            all_edge_label = pd.read_csv(os.path.join(self.raw_dir, 'all-edge-label.csv.gz'), compression='gzip', header = None).values.astype(np.int64) # (num_edge, 7) numpy array
            num_node_list = pd.read_csv(os.path.join(self.raw_dir, 'num-node-list.csv.gz'), compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list

        except FileNotFoundError:
            raise RuntimeError('No necessary file')
        
        graph = dict()
        print('Processing drug-drug interaction graph...')
        if add_inverse_edge:
            ### duplicate edge
            duplicated_edge = np.repeat(all_edge, 2, axis = 1)
            duplicated_edge[0, 1::2] = duplicated_edge[1,0::2]
            duplicated_edge[1, 1::2] = duplicated_edge[0,0::2]
            # graph['edge_index'] = duplicated_edge
            # graph['edge_label'] = all_edge_label
            graph['edge_index'] = torch.from_numpy(duplicated_edge)
            graph['edge_label'] = torch.from_numpy(np.repeat(all_edge_label, 2, axis=0))

        else:
            graph['edge_index'] = torch.from_numpy(all_edge)
            graph['edge_label'] = torch.from_numpy(all_edge_label)

        graph['num_nodes'] = num_node_list[0]

        print('Processing molecule graphs...')
        molecule_graph_list = self.process_molecules()

        print('Saving...')
        torch.save(graph, self.processed_paths[0])
        torch.save(molecule_graph_list, self.processed_paths[1])

    def process_molecules(self):
        self.dataset_path = os.path.join(self.root, self.inter_dataset_config.storage[0])
        self.smiles_col = self.inter_dataset_config.smiles_column
        if self.dataset_name.startswith('ogb'):
            molecle_dataset = PygGraphPropPredDataset(self.dataset_name.replace('_', '-'), root=self.inter_dataset_root)

        elif self.dataset_path[-4:] == '.csv':
            self.whole_data_df = pd.read_csv(self.dataset_path)
            # valid_smiles = filter_invalid_smiles(list(self.whole_data_df.loc[:,self.smiles_col]))
            # self.whole_data_df = self.whole_data_df[self.whole_data_df[self.smiles_col].isin(valid_smiles)].reset_index(drop=True)
            molecle_dataset = self.process_smiles()

        elif self.dataset_path[-3:] == '.gz':
            self.whole_data_df = pd.read_csv(self.dataset_path, compression='gzip', header=0)
            molecle_dataset = self.process_smiles()

        else:
            raise print(f"File Format must be in ['.csv', '.gz'] or in OGB-Benchmark")

        return molecle_dataset

    def process_smiles(self):
        smiles_list = self.whole_data_df[self.smiles_col]
        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

        dataset = []
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs_list[i]
            if rdkit_mol is None:
                continue
            data = mol_to_graph_data_obj(rdkit_mol)
            data.id = torch.tensor([i])
            data.smiles = smiles_list[i]
            dataset.append(data)
        return dataset


class ProteinDataset(InMemoryDataset):
    r"""
    Protein Dataset
    """
    def __init__(self, args, config, split, **kwargs):
        self.args = args
        self.config = config
        self.kwargs = kwargs
        self.split = split

        self.inter_dataset_root = args.inter_data_dir
        self.dataset_name = list(self.config.inter_dataset_config.keys())[0]
        self.inter_dataset_config = self.config.inter_dataset_config[self.dataset_name]
        self.add_inverse_edge = self.inter_dataset_config.get('add_inverse_edge', False)

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                              'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                              'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                              'N': 2, 'Y': 18, 'M': 12, 'U': 20}


        super(ProteinDataset, self).__init__(root=os.path.join(self.inter_dataset_root, self.dataset_name.replace('-', '_')))
        self.data = torch.load(self.processed_paths[0])
        self.protein_dataset = torch.load(self.processed_paths[1])

        self.splits = self.get_edge_idx_split()
        if self.add_inverse_edge and self.split == 'train':
            self.edges = torch.cat((self.data['edge_index'][:, self.splits[split]], self.data['edge_index'][:, self.splits[split]][[1, 0]]), dim=1).t()
            self.labels = torch.cat((self.data['edge_label'][self.splits[split]], self.data['edge_label'][self.splits[split]]), dim=0)
        else:
            self.edges = self.data['edge_index'][:, self.splits[split]].t()
            self.labels = self.data['edge_label'][self.splits[split]]

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index):
        row, col = self.edges[index]
        protein_row, protein_col = self.protein_dataset[row], self.protein_dataset[col]
        return {
            "edge": self.edges[index],
            "protein1_data": protein_row,
            "protein2_data": protein_col,
            "label": self.labels[index],
        }

    def collate_fn(self, batch):
        edges = torch.stack([torch.LongTensor(data["edge"]) for data in batch], dim=0)
        protein1_data = Batch.from_data_list([data["protein1_data"] for data in batch])
        protein2_data = Batch.from_data_list([data["protein2_data"] for data in batch])
        labels = torch.stack([torch.LongTensor(data["label"]) for data in batch],dim=0).to(dtype=torch.long)
        return {
            "edge": edges,
            "protein1_data": protein1_data,
            "protein2_data": protein2_data,
            "label": labels,
        }

    def get_edge_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.inter_dataset_config.split

        path = os.path.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(os.path.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
        test_idx = pd.read_csv(os.path.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]

        if os.path.exists(os.path.join(path, 'valid.csv.gz')):
            valid_idx = pd.read_csv(os.path.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
            return {'train': torch.LongTensor(train_idx), 'valid': torch.LongTensor(valid_idx), 'test': torch.LongTensor(test_idx)}
        else:
            return {'train': torch.LongTensor(train_idx), 'test': torch.LongTensor(test_idx)}

    @property
    def processed_file_names(self):
        return ['ppi_data_processed.pt', 'protein_data_processed.pt']

    @property
    def raw_file_names(self):
        file_names = ['edge']
        return [file_name + '.csv.gz' for file_name in file_names]

    def process(self):
        # add inverse edges
        add_inverse_edge = self.add_inverse_edge

        # loading necessary files
        try:
            all_edge = pd.read_csv(os.path.join(self.raw_dir, 'all-edge.csv.gz'), compression='gzip', header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
            all_edge_label = pd.read_csv(os.path.join(self.raw_dir, 'all-edge-label.csv.gz'), compression='gzip', header = None).values.astype(np.int64) # (num_edge, 7) numpy array
            num_node_list = pd.read_csv(os.path.join(self.raw_dir, 'protein-num-node-list.csv.gz'), compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list

        except FileNotFoundError:
            raise RuntimeError('No necessary file')


        try:
            protein_mapping = pd.read_csv(os.path.join(self.root, "mapping", "nodeidx2proteinid.csv"), header=0)
            protein_idx2protein = dict(zip(protein_mapping.node_idx, protein_mapping.Protein))
            protein_idx2sequence = dict(zip(protein_mapping.node_idx, protein_mapping.Sequence))

        except FileNotFoundError:
            raise RuntimeError('No necessary file')
        
        # self.process_protein()

        graph = dict()
        print('Processing protein-protein interaction graph...')
        if add_inverse_edge:
            ### duplicate edge
            duplicated_edge = np.repeat(all_edge, 2, axis = 1)
            duplicated_edge[0, 1::2] = duplicated_edge[1,0::2]
            duplicated_edge[1, 1::2] = duplicated_edge[0,0::2]
            graph['edge_index'] = torch.from_numpy(duplicated_edge)
            graph['edge_label'] = torch.from_numpy(np.repeat(all_edge_label, 2, axis=0))

        else:
            graph['edge_index'] = torch.from_numpy(all_edge)
            graph['edge_label'] = torch.from_numpy(all_edge_label)

        # print('Processing protein graphs...')
        # protein_graph_list = []
        # for idx, num_node in enumerate(tqdm(num_node_list, total=len(num_node_list))):
        #     g = Data()
        #     g.num_nodes = num_node
        #     ### handling edge
        #     g.edge_index = torch.from_numpy(np.array(self.protein_inter_graph_adj[idx]).T)
        #     ### handling node
        #     g.x = torch.from_numpy(self.protein_graph_feats[idx]).to(dtype=torch.float)
        #     protein_graph_list.append(g)

        print('Processing protein graphs...')
        protein_graph_list = self.process_protein_graph(list(protein_idx2protein.values()), [protein_idx2sequence[i] for i in protein_idx2protein.keys()])
        

        print('Saving...')
        torch.save(graph, self.processed_paths[0])
        torch.save(protein_graph_list, self.processed_paths[1])

    def process_protein_graph(self, protein_list, protein_seq_list):
        protein_graphs = []
        for idx, name in enumerate(tqdm(protein_list)):
            X = torch.load(self.raw_dir + "/pdb/" + name + ".tensor")
            seq = torch.tensor([self.letter_to_num[aa] for aa in protein_seq_list[idx]], dtype=torch.long)
            # prottrans_feat = torch.load(self.feature_path + "ProtTrans/" + name + ".tensor")
            dssp_feat = torch.load(self.raw_dir + '/dssp/' + name + ".tensor")
            # pre_computed_node_feat = torch.cat([prottrans_feat, dssp_feat], dim=-1)
            pre_computed_node_feat = dssp_feat

            X_ca = X[:, 1]
            edge_index = torch_geometric.nn.radius_graph(X_ca, r=15, loop=True, max_num_neighbors = 1000, num_workers = 4)
            graph_data = torch_geometric.data.Data(name=name, seq=seq, x=X, node_feat=pre_computed_node_feat,
                                                   edge_index=edge_index)
            protein_graphs.append(graph_data)
        return protein_graphs

    def process_protein(self):
        data_dir = os.path.join(self.inter_dataset_root, self.dataset_name.replace('-', '_'))
        raw_dir = os.path.join(data_dir, "raw")
        self.max_len = self.inter_dataset_config.get("max_len", 2000)
        self.pseq_path = os.path.join(raw_dir, self.inter_dataset_config.protein_seq_path)
        self.vec_path = os.path.join(raw_dir, self.inter_dataset_config.vec_path)

        protein_nodes = pd.read_csv(os.path.join(self.inter_dataset_root, self.inter_dataset_config.storage[0]))
        self.protein_name2idx = dict(zip(protein_nodes.Protein, protein_nodes.node_idx))
        self.protein_idx2name = dict(zip(protein_nodes.node_idx, protein_nodes.Protein))

        # aac: amino acid sequences
        pseq_dict = {}
        protein_len = []
        for line in open(self.pseq_path):
            line = line.strip().split('\t')
            if line[0] not in pseq_dict.keys():
                pseq_dict[line[0]] = line[1]
                protein_len.append(len(line[1]))

        print("protein num: {}".format(len(pseq_dict)))
        print("protein average length: {}".format(np.average(protein_len)))
        print("protein max & min length: {}, {}".format(np.max(protein_len), np.min(protein_len)))

        self.acid2vec = {}
        self.dim = None
        for line in open(self.vec_path):
            line = line.strip().split('\t')
            temp = np.array([float(x) for x in line[1].split()])
            self.acid2vec[line[0]] = temp
            if self.dim is None:
                self.dim = len(temp)
        print("acid vector dimension: {}".format(self.dim))

        self.pvec_dict = {}
        for p_name in tqdm(pseq_dict.keys()):
            temp_seq = pseq_dict[p_name]
            temp_vec = []
            for acid in temp_seq:
                temp_vec.append(self.acid2vec[acid])
            temp_vec = np.array(temp_vec)
            # temp_vec = self.embed_normal(temp_vec, self.dim)
            self.pvec_dict[p_name] = temp_vec
        
        self.protein_seq_emb_dict = {}  # node-idx: protein_seq_feature
        for name in self.protein_name2idx.keys():
            self.protein_seq_emb_dict[self.protein_name2idx[name]] = self.pvec_dict[name]

        protein_graph_feat_path = os.path.join(raw_dir, self.inter_dataset_config.protein_feats_path)
        protein_graph_adj_path = os.path.join(raw_dir, self.inter_dataset_config.protein_graph_adj_path)

        self.protein_graph_feats = torch.load(protein_graph_feat_path)
        self.protein_inter_graph_adj = np.load(protein_graph_adj_path, allow_pickle=True)

    def embed_normal(self, seq, dim):
        if len(seq) > self.max_len:
            return seq[:self.max_len]
        elif len(seq) < self.max_len:
            less_len = self.max_len - len(seq)
            return np.concatenate((seq, np.zeros((less_len, dim))))
        return seq


class DrugProteinDataset(InMemoryDataset):
    def __init__(self, args, config, split, **kwargs):
        self.args = args
        self.config = config
        self.kwargs = kwargs
        self.split = split

        self.inter_dataset_root = args.inter_data_dir
        self.dataset_name = list(self.config.inter_dataset_config.keys())[0]
        self.inter_dataset_config = self.config.inter_dataset_config[self.dataset_name]
        self.add_inverse_edge = self.inter_dataset_config.get('add_inverse_edge', True)

        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                              'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                              'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                              'N': 2, 'Y': 18, 'M': 12, 'U': 20}


        super(DrugProteinDataset, self).__init__(root=os.path.join(self.inter_dataset_root, self.dataset_name.replace('-', '_')))
        self.data = torch.load(self.processed_paths[0])
        self.protein_dataset = torch.load(self.processed_paths[1])
        self.molecule_dataset = torch.load(self.processed_paths[2])
        # self.molecule_dataset = [self.get_mol_features(compound.smiles) for compound in self.molecule_smiles_data]
        self.molecule_num = len(self.molecule_dataset)

        self.splits = self.get_edge_idx_split()
        if self.add_inverse_edge:
            self.edges = torch.cat((self.data['edge_index'][:, self.splits[split]], self.data['edge_index'][:, self.splits[split]][[1, 0]]), dim=1).t()
            self.labels = torch.cat((self.data['edge_label'][self.splits[split]], self.data['edge_label'][self.splits[split]]), dim=0)
        else:
            self.edges = self.data['edge_index'][:, self.splits[split]].t()
            self.labels = self.data['edge_label'][self.splits[split]]

    @property
    def processed_file_names(self):
        return ['dpi_data_processed.pt','protein_data_processed.pt', 'molecule_data_processed.pt']

    @property
    def raw_file_names(self):
        file_names = ['edge']
        return [file_name + '.csv.gz' for file_name in file_names]

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, index):
        row, col = self.edges[index]
        compound, protein = self.molecule_dataset[row], self.protein_dataset[col - self.molecule_num]
        return {
            "edge": self.edges[index],
            "compound_data": compound,
            "protein_data": protein,
            "label": self.labels[index],
        }

    def collate_fn(self, batch):
        edges = torch.stack([torch.LongTensor(data["edge"]) for data in batch], dim=0)
        labels = torch.stack([torch.LongTensor(data["label"]) for data in batch],dim=0).to(dtype=torch.long)
        batch_outs = {"edge": edges, "label": labels}

        compound_data = molecule_collate_fn([data["compound_data"] for data in batch])
        protein_data = protein_collate_fn([data["protein_data"] for data in batch])
        batch_outs = {**batch_outs, **compound_data}
        batch_outs = {**batch_outs, **protein_data}
        return batch_outs

    def get_mol_features(self, smiles):
        compound_node_features, compound_adj_matrix, _ = get_mol_features(smiles, self.inter_dataset_config.get("atom_dim", 34))
        compound_word_embedding = get_mol2vec_features(self.mol2vec_model, smiles)
        return {
            "compound_node_features": compound_node_features,
            "compound_adj_matrix": compound_adj_matrix,
            "compound_word_embedding": compound_word_embedding,
        }

    def get_edge_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.inter_dataset_config.split

        path = os.path.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(os.path.join(path, 'train.csv.gz'), compression='gzip', header=None).values.T[0]
        test_idx = pd.read_csv(os.path.join(path, 'test.csv.gz'), compression='gzip', header=None).values.T[0]

        if os.path.exists(os.path.join(path, 'valid.csv.gz')):
            valid_idx = pd.read_csv(os.path.join(path, 'valid.csv.gz'), compression='gzip', header=None).values.T[0]
            return {'train': torch.LongTensor(train_idx), 'valid': torch.LongTensor(valid_idx), 'test': torch.LongTensor(test_idx)}
        else:
            return {'train': torch.LongTensor(train_idx), 'test': torch.LongTensor(test_idx)}

    def process_protein_graph(self, protein_list, protein_seq_list):
        protein_graphs = []
        for idx, name in enumerate(tqdm(protein_list)):
            X = torch.load(self.raw_dir + "/pdb/" + name + ".tensor")
            seq = torch.tensor([self.letter_to_num[aa] for aa in protein_seq_list[idx]], dtype=torch.long)
            # prottrans_feat = torch.load(self.feature_path + "ProtTrans/" + name + ".tensor")
            # dssp_feat = torch.load(self.raw_dir + '/dssp/' + name + ".tensor")
            # pre_computed_node_feat = torch.cat([prottrans_feat, dssp_feat], dim=-1)
            # pre_computed_node_feat = dssp_feat
            pre_computed_node_feat = torch.from_numpy(self.protein_seq_emb_dict[idx])

            X_ca = X[:, 1]
            edge_index = torch_geometric.nn.radius_graph(X_ca, r=15, loop=True, max_num_neighbors = 1000, num_workers = 4)
            graph_data = torch_geometric.data.Data(name=name, seq=seq, x=X, node_feat=pre_computed_node_feat,
                                                   edge_index=edge_index)
            assert X_ca.shape[0] == pre_computed_node_feat.shape[0], f"Shape: {X_ca.shape[0]} != {pre_computed_node_feat.shape[0]}"
            graph_data.num_nodes = X_ca.shape[0]
            protein_graphs.append(graph_data)
        return protein_graphs

    def process(self):
        # add inverse edges
        add_inverse_edge = self.add_inverse_edge

        # loading necessary files
        try:
            all_edge = pd.read_csv(os.path.join(self.raw_dir, 'all-edge.csv.gz'), compression='gzip', header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
            all_edge_label = pd.read_csv(os.path.join(self.raw_dir, 'all-edge-label.csv.gz'), compression='gzip', header = None).values.astype(np.int64) # (num_edge, 7) numpy array
            # dpi_num_edge_list = pd.read_csv(os.path.join(self.raw_dir, 'all-num-edge-list.csv.gz'), compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_edge, ) python list
            # protein_num_node_list = pd.read_csv(os.path.join(self.raw_dir, 'protein-num-node-list.csv.gz'), compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_graph, ) protein list

        except FileNotFoundError:
            raise RuntimeError('No necessary file')

        graph = dict()
        print('Processing drug-protein interaction graph...')
        if add_inverse_edge:
            ### duplicate edge
            duplicated_edge = np.repeat(all_edge, 2, axis = 1)
            duplicated_edge[0, 1::2] = duplicated_edge[1,0::2]
            duplicated_edge[1, 1::2] = duplicated_edge[0,0::2]
            graph['edge_index'] = torch.from_numpy(duplicated_edge)
            graph['edge_label'] = torch.from_numpy(np.repeat(all_edge_label, 2, axis=0))

        else:
            graph['edge_index'] = torch.from_numpy(all_edge)
            graph['edge_label'] = torch.from_numpy(all_edge_label)


        try:
            protein_mapping = pd.read_csv(os.path.join(self.root, "mapping", "nodeidx2proteinid.csv"), header=0)
            protein_idx2uniprot = dict(zip(protein_mapping.node_idx, protein_mapping.uniprot))
            protein_idx2sequence = dict(zip(protein_mapping.node_idx, protein_mapping.protein_sequence))

            drug_mapping = pd.read_csv(os.path.join(self.root, "mapping", "nodeidx2drugid.csv"), header=0)
            drug_idx2smiles = dict(zip(drug_mapping.node_idx, drug_mapping.SMILES))

        except FileNotFoundError:
            raise RuntimeError('No necessary file')
        
        try:
        #     with open(os.path.join(self.raw_dir, 'protein_embedding', 'bert_embedding_Nongram.pkl'), 'rb') as f:
        #         self.bert_embed_dict = pickle.load(f)

            self.mol2vec_model = word2vec.Word2Vec.load(os.path.join(self.raw_dir, "mol2vec/model_300dim.pkl"))

        except FileNotFoundError:
            raise RuntimeError('No necessary file')

        print('Processing molecule graphs...')
        molecule_graph_list = []
        for idx in tqdm(range(len(drug_mapping))):
            smiles = drug_idx2smiles[idx]
            compound_node_features, compound_adj_matrix, _ = get_mol_features(smiles, self.inter_dataset_config.get("atom_dim", 34))
            compound_word_embedding = get_mol2vec_features(self.mol2vec_model, smiles)
            g = Data()
            g.smiles = smiles
            g.compound_node_features = compound_node_features
            g.compound_adj_matrix = compound_adj_matrix
            g.compound_word_embedding = compound_word_embedding
            molecule_graph_list.append(g)

        print('Processing protein graphs...')
        protein_graph_list = self.process_protein_graph(list(protein_idx2uniprot.values()), [protein_idx2sequence[i] for i in protein_idx2uniprot.keys()])
        # for idx, num_node in enumerate(tqdm(protein_num_node_list, total=len(protein_num_node_list))):
        #     g = Data()
        #     node_idx = compount_node_num + idx
        #     g.num_nodes = num_node
        #     ### handling contact_map
        #     g.protein_map = torch.from_numpy(self.get_contact_map(protein_idx2uniprot[node_idx])).to(dtype=torch.float)
        #     ### handling node
        #     g.protein_node_feat = torch.from_numpy(self.get_node_features(protein_idx2uniprot[node_idx])).to(dtype=torch.float)
        #     g.sequence = protein_idx2sequence[node_idx]
        #     g.protein_embedding = torch.from_numpy(self.get_pretrained_embedding(protein_idx2sequence[node_idx])).to(dtype=torch.float)
        #     g.uniprot = protein_idx2uniprot[node_idx]
        #     protein_graph_list.append(g)

        print('Saving...')
        torch.save(graph, self.processed_paths[0])
        torch.save(protein_graph_list, self.processed_paths[1])
        torch.save(molecule_graph_list, self.processed_paths[2])

    def prepare_features(self):
        data_dir = os.path.join(self.inter_dataset_root, self.dataset_name.replace('-', '_'))
        raw_dir = os.path.join(data_dir, "raw")
        self.max_len = self.inter_dataset_config.get("max_len", 2000)
        self.pseq_path = os.path.join(raw_dir, self.inter_dataset_config.protein_seq_path)
        self.vec_path = os.path.join(raw_dir, self.inter_dataset_config.vec_path)

        protein_nodes = pd.read_csv(os.path.join(self.inter_dataset_root, self.inter_dataset_config.storage[0]))
        self.protein_name2idx = dict(zip(protein_nodes.Protein, protein_nodes.node_idx))
        self.protein_idx2name = dict(zip(protein_nodes.node_idx, protein_nodes.Protein))

        # aac: amino acid sequences
        pseq_dict = {}
        protein_len = []
        for line in open(self.pseq_path):
            line = line.strip().split('\t')
            if line[0] not in pseq_dict.keys():
                pseq_dict[line[0]] = line[1]
                protein_len.append(len(line[1]))
        self.num_node_list = protein_len

        print("protein num: {}".format(len(pseq_dict)))
        print("protein average length: {}".format(np.average(protein_len)))
        print("protein max & min length: {}, {}".format(np.max(protein_len), np.min(protein_len)))

        self.acid2vec = {}
        self.dim = None
        for line in open(self.vec_path):
            line = line.strip().split('\t')
            temp = np.array([float(x) for x in line[1].split()])
            self.acid2vec[line[0]] = temp
            if self.dim is None:
                self.dim = len(temp)
        print("acid vector dimension: {}".format(self.dim))

        self.pvec_dict = {}
        for p_name in tqdm(pseq_dict.keys()):
            temp_seq = pseq_dict[p_name]
            temp_vec = []
            for acid in temp_seq:
                temp_vec.append(self.acid2vec[acid])
            temp_vec = np.array(temp_vec)
            # temp_vec = self.embed_normal(temp_vec, self.dim)
            self.pvec_dict[p_name] = temp_vec
        
        self.protein_seq_emb_dict = {}  # node-idx: protein_seq_feature
        for name in self.protein_name2idx.keys():
            self.protein_seq_emb_dict[self.protein_name2idx[name]] = self.pvec_dict[name]


def files_exist(files):
    return len(files) != 0 and all(os.path.exists(f) for f in files)


class LinkGraphDataset(InMemoryDataset):
    def __init__(self, name, config, root=None, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.config = config
        self.new_split = False

        super(LinkGraphDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if not files_exist(self.processed_paths) or self.new_split:
            ppi_split_dataset(split_mode=self.config.split)
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'link_data_processed.pt'

    def get_edge_split(self, split_type=None):
        if split_type is None:
            split_type = self.config.split

        path = os.path.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train = replace_numpy_with_torchtensor(torch.load(os.path.join(path, 'train.pt')))
        test = replace_numpy_with_torchtensor(torch.load(os.path.join(path, 'test.pt')))

        if os.path.exists(os.path.join(path, 'valid.pt')):
            valid = replace_numpy_with_torchtensor(torch.load(os.path.join(path, 'valid.pt')))
            return {'train': train, 'valid': valid, 'test': test}
        else:
            return {'train': train, 'test': test}

    @property
    def raw_file_names(self):
        file_names = ['edge']
        return [file_name + '.csv.gz' for file_name in file_names]

    def process(self):
        add_inverse_edge = self.config.get('add_inverse_edge', True)

        # loading necessary files
        try:
            edge = pd.read_csv(os.path.join(self.raw_dir, 'edge.csv.gz'), compression='gzip', header = None).values.T.astype(np.int64) # (2, num_edge) numpy array
            num_node_list = pd.read_csv(os.path.join(self.raw_dir, 'num-node-list.csv.gz'), compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_graph, ) python list
            num_edge_list = pd.read_csv(os.path.join(self.raw_dir, 'num-edge-list.csv.gz'), compression='gzip', header = None).astype(np.int64)[0].tolist() # (num_edge, ) python list

        except FileNotFoundError:
            raise RuntimeError('No necessary file')

        try:
            edge_label = pd.read_csv(os.path.join(self.raw_dir, 'edge-label.csv.gz'), compression='gzip', header = None).values
            if 'int' in str(edge_label.dtype):
                edge_label = edge_label.astype(np.int64)
            else:
                #float
                edge_label = edge_label.astype(np.float32)

        except FileNotFoundError:
            edge_label = None

        try:
            node_feat = pd.read_csv(os.path.join(self.raw_dir, 'node-feat.csv.gz'), compression='gzip', header = None).values
            if 'int' in str(node_feat.dtype):
                node_feat = node_feat.astype(np.int64)
            else:
                # float
                node_feat = node_feat.astype(np.float32)
        except FileNotFoundError:
            node_feat = None

        print('Processing graphs...')
        graph_list = []
        num_node_accum = 0
        num_edge_accum = 0

        graph = dict()
        num_node, num_edge = num_node_list[0], num_edge_list[0]
        if add_inverse_edge:
            ### duplicate edge
            duplicated_edge = np.repeat(edge[:, num_edge_accum:num_edge_accum+num_edge], 2, axis = 1)
            duplicated_edge[0, 1::2] = duplicated_edge[1,0::2]
            duplicated_edge[1, 1::2] = duplicated_edge[0,0::2]
            graph['edge_index'] = duplicated_edge

            if edge_label is not None:
                graph['edge_label'] = np.repeat(edge_label[num_edge_accum:num_edge_accum+num_edge], 2, axis=0)
            else:
                graph['edge_label'] = None

        else:
            graph['edge_index'] = edge[:, num_edge_accum:num_edge_accum+num_edge]

            if edge_label is not None:
                graph['edge_label'] = edge_label[num_edge_accum:num_edge_accum+num_edge]
            else:
                graph['edge_label'] = None

        ### handling node
        if node_feat is not None:
            graph['node_feat'] = node_feat[num_node_accum:num_node_accum+num_node]
        else:
            graph['node_feat'] = None

        graph['num_nodes'] = num_node
        num_node_accum += num_node
        num_edge_accum += num_edge
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
        # return self.collate([data])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


if __name__ == '__main__':
    pass
    # from config import parse_args, Config
    # args = parse_args()
    # config = Config(args)
    
    # train_dataset = LinkGraphDataset(args, config)
    # valid_dataset = LinkGraphDataset(args, config, split='valid')
    # test_dataset = LinkGraphDataset(args, config, split='test')

    # inter_dataset_config = config.inter_dataset_config
    # dataset_name = list(inter_dataset_config.keys())[0]
    # inter_dataset_root = args.inter_data_dir
    # inter_dataset_config = inter_dataset_config[dataset_name]

    # link_dataset = LinkGraphDataset(dataset_name, inter_dataset_config, root=os.path.join(inter_dataset_root, dataset_name.replace('-', '_')))