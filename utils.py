#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author:  Jiahua Rao
@license: BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact: jiahua.rao@gmail.com
@time:    05/2023
'''


import os
import copy
import random

from rdkit import Chem
from gensim.models import Word2Vec, word2vec

import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from itertools import product
from torch_geometric.data import Batch
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from mol2vec.features import (DfVec, MolSentence, mol2alt_sentence,
                              mol2sentence, sentences2vec)
from descriptastorus.descriptors import rdNormalizedDescriptors


def filter_invalid_smiles(smiles_list):
    valid_smiles_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumHeavyAtoms() > 0:
            valid_smiles_list.append(smiles)
    return valid_smiles_list


def replace_numpy_with_torchtensor(obj):
    # assume obj comprises either list or dictionary
    # replace all the numpy instance with torch tensor.

    if isinstance(obj, dict):
        for key in obj.keys():
            if isinstance(obj[key], np.ndarray):
                obj[key] = torch.from_numpy(obj[key])
            else:
                replace_numpy_with_torchtensor(obj[key])
    elif isinstance(obj, list):
        for i in range(len(obj)):
            if isinstance(obj[i], np.ndarray):
                obj[i] = torch.from_numpy(obj[i])
            else:
                replace_numpy_with_torchtensor(obj[i])

    # if the original input obj is numpy array
    elif isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    return obj


def write_log(logFile, text, isPrint=True):
    if isPrint:
        print(text)
    
    if logFile != print:
        logFile.write(text)
        logFile.write('\n')



def compute_loss(logits, labels, loss_func, is_gold=None, pl_weight=0.5, is_augmented=False):
    """
    Combine two types of losses: (1-alpha)*MLE (CE loss on gold) + alpha*Pl_loss (CE loss on pseudo labels)
    """
    import torch as th

    if is_augmented:# and ((n_pseudo := sum(~is_gold)) > 0):
        deal_nan = lambda x: 0 if th.isnan(x) else x
        mle_loss = deal_nan(loss_func(logits[is_gold], labels[is_gold]))
        pl_loss = deal_nan(loss_func(logits[~is_gold], labels[~is_gold]))
        loss = pl_weight * pl_loss + (1 - pl_weight) * mle_loss
    else:
        loss = loss_func(logits, labels)
    return loss


def adjust_lr(optimizer, decay_ratio, lr):
    lr_ = lr * (1 - decay_ratio)
    lr_min = lr * 0.0001
    if lr_ < lr_min:
        lr_ = lr_min
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_


def collate(data_list):
    r"""Collates a python list of data objects to the internal storage
    format of :class:`torch_geometric.data.InMemoryDataset`."""
    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if torch.is_tensor(item[key]):
            s = slices[key][-1] + item[key].size(
                item.__cat_dim__(key, item[key]))
        else:
            s = slices[key][-1] + 1
        slices[key].append(s)

    if hasattr(data_list[0], '__num_nodes__'):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)

    for key in keys:
        item = data_list[0][key]
        if torch.is_tensor(item) and len(data_list) > 1:
            data[key] = torch.cat(data[key],
                                    dim=data.__cat_dim__(key, item))
        elif torch.is_tensor(item):  # Don't duplicate attributes...
            data[key] = data[key][0]
        elif isinstance(item, int) or isinstance(item, float):
            data[key] = torch.tensor(data[key])

        slices[key] = torch.tensor(slices[key], dtype=torch.long)

    batch = Batch.from_data_list(data)
    return batch


class Metrictor_PPI:
    def __init__(self, pre_y, truth_y, true_prob, is_binary=False, logging=False):
        self.logging = logging
        self.is_binary = is_binary

        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.pre = np.array(pre_y).squeeze()
        self.tru = np.array(truth_y).squeeze()
        self.true_prob = np.array(true_prob).squeeze()
        if is_binary:
            length = pre_y.shape[0]
            for i in range(length):
                if pre_y[i] == truth_y[i]:
                    if truth_y[i] == 1:
                        self.TP += 1
                    else:
                        self.TN += 1
                elif truth_y[i] == 1:
                    self.FN += 1
                elif pre_y[i] == 1:
                    self.FP += 1
            self.num = length

        else:
            N, C = pre_y.shape
            for i in range(N):
                for j in range(C):
                    if pre_y[i][j] == truth_y[i][j]:
                        if truth_y[i][j] == 1:
                            self.TP += 1
                        else:
                            self.TN += 1
                    elif truth_y[i][j] == 1:
                        self.FN += 1
                    elif truth_y[i][j] == 0:
                        self.FP += 1
            self.num = N * C

    def show_result(self, is_print=True):
        self.Accuracy = (self.TP + self.TN) / (self.num + 1e-10)
        self.Precision = self.TP / (self.TP + self.FP + 1e-10)
        self.Recall = self.TP / (self.TP + self.FN + 1e-10)
        self.F1 = 2 * self.Precision * self.Recall / (self.Precision + self.Recall + 1e-10)
        # fpr, tpr, thresholds = metrics.roc_curve(self.tru, self.pre, pos_label=1)
        # self.Auc = metrics.auc(fpr, tpr)

        if self.is_binary:
            fpr, tpr, thresholds = roc_curve(self.tru, self.pre)
            self.Auc = auc(fpr, tpr)

            aupr_entry_1 = self.tru
            aupr_entry_2 = self.true_prob
            precision, recall, _ = precision_recall_curve(aupr_entry_1, aupr_entry_2)
            aupr = auc(recall,precision)
            self.Aupr = aupr
        else:
            aupr_entry_1 = self.tru
            aupr_entry_2 = self.true_prob

            aupr = np.zeros(7)
            for i in range(7):
                precision, recall, _ = precision_recall_curve(aupr_entry_1[:,i], aupr_entry_2[:,i])
                aupr[i] = auc(recall,precision)
            self.Aupr = np.mean(aupr)

            roc_auc = np.zeros(7)
            for i in range(7):
                fpr, tpr, _ = roc_curve(aupr_entry_1[:,i], aupr_entry_2[:,i])
                roc_auc[i] = auc(fpr, tpr)
            self.Auc = np.mean(roc_auc)

        # write_log(self.logging, "Accuracy: {}".format(self.Accuracy), isPrint=False)
        # write_log(self.logging, "Precision: {}".format(self.Precision), isPrint=False)
        # write_log(self.logging, "Recall: {}".format(self.Recall), isPrint=False)
        # write_log(self.logging, "F1-Score: {}".format(self.F1), isPrint=False)
        # write_log(self.logging, "AUC: {}".format(self.Auc), isPrint=False)
        # write_log(self.logging, "Aupr: {}".format(self.Aupr), isPrint=False)

        print("Accuracy: {}".format(self.Accuracy))
        print("Precision: {}".format(self.Precision))
        print("Recall: {}".format(self.Recall))
        print("F1-Score: {}".format(self.F1))
        print("AUC: {}".format(self.Auc))
        print("Aupr: {}".format(self.Aupr))


def Metrictor_DPI(correct_labels, scores, thres=0.5):
    auc = roc_auc_score(correct_labels, scores)
    aupr = average_precision_score(correct_labels, scores)
    predict_labels = [1. if i >= thres else 0. for i in scores]
    acc = accuracy_score(correct_labels, predict_labels)
    precision = precision_score(correct_labels, predict_labels)
    recall = recall_score(correct_labels, predict_labels)
    f1 = f1_score(correct_labels, predict_labels)
    return {"auc": auc, "acc": acc, "aupr": aupr,
            "precision": precision, "recall": recall,
            "f1": f1}


def Metrictor_DDI():
    return

def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    candiate_node.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        selected_node.append(cur_node)
        for edge_index in node_to_edge_index[cur_node]:

            if edge_index not in selected_edge_index:
                selected_edge_index.append(edge_index)

                end_node = -1
                if ppi_list[edge_index][0] == cur_node:
                    end_node = ppi_list[edge_index][1]
                else:
                    end_node = ppi_list[edge_index][0]

                if end_node not in selected_node and end_node not in candiate_node:
                    candiate_node.append(end_node)
            else:
                continue
        # print(len(selected_edge_index), len(candiate_node))
    node_list = candiate_node + selected_node
    print(len(node_list), len(selected_edge_index))
    return selected_edge_index


def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    stack.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        # print(len(selected_edge_index), len(stack), len(selected_node))
        cur_node = stack[-1]
        if cur_node in selected_node:
            flag = True
            for edge_index in node_to_edge_index[cur_node]:
                if flag:
                    end_node = -1
                    if ppi_list[edge_index][0] == cur_node:
                        end_node = ppi_list[edge_index][1]
                    else:
                        end_node = ppi_list[edge_index][0]

                    if end_node in selected_node:
                        continue
                    else:
                        stack.append(end_node)
                        flag = False
                else:
                    break
            if flag:
                stack.pop()
            continue
        else:
            selected_node.append(cur_node)
            for edge_index in node_to_edge_index[cur_node]:
                if edge_index not in selected_edge_index:
                    selected_edge_index.append(edge_index)

    return selected_edge_index


def ppi_split_dataset(split_mode='random', test_size=0.2, seed=2, consistent=False):
    prefix = "./data/high_ppi/"

    # seed = random.randint(0, 10000)
    random.seed(seed)
        
    print(f"!!!!!!!! New split. seed:{seed}")
    fp = open(prefix+f"split/{split_mode}/seed.txt", "w")
    fp.write(str(seed))
    fp.close()

    protein_path = os.path.join(prefix, "raw", "protein.SHS27k.sequences.dictionary.pro3.tsv")
    ppi_path = os.path.join(prefix, 'raw', "protein.actions.SHS27k.STRING.pro2.txt")
    
    proteins = pd.read_csv(protein_path, sep='\t', header=None, names=['Protein', 'Sequence'])

    protein_idx = 0
    protein_nodes = {}

    for i in tqdm(range(len(proteins))):
        node = proteins.loc[i, "Protein"]
        if node not in protein_nodes.keys():
            protein_nodes[node] = protein_idx
            protein_idx += 1

    protein_node_df = proteins[['Protein']]
    protein_node_df["node_idx"] = proteins.index.tolist()
    protein_node_df.to_csv(os.path.join(prefix, "mapping", "nodeidx2proteinid.csv"), index=False)

    raw_ppi_df = pd.read_csv(ppi_path, sep='\t', header=0)[['item_id_a', 'item_id_b', 'mode', 'is_directional', 'a_is_acting', 'score']]

    class_map = {'reaction': 0, 'binding': 1, 'ptmod': 2, 'activation': 3, 'inhibition': 4, 'catalysis': 5,
                'expression': 6}

    ppi_idx = 0
    ppi_label_list = []
    ppi_dict = {}
    ppi_list = []
    for i in tqdm(range(len(raw_ppi_df))):
        from_node = raw_ppi_df.loc[i, "item_id_a"]
        to_node = raw_ppi_df.loc[i, "item_id_b"]
        relation_mode = raw_ppi_df.loc[i, "mode"]
        if from_node < to_node:
            temp_data = from_node + "__" + to_node
        else:
            temp_data = to_node + "__" + from_node
        if temp_data not in ppi_dict.keys():
            ppi_dict[temp_data] = ppi_idx
            temp_label = [0, 0, 0, 0, 0, 0, 0]
            temp_label[class_map[relation_mode]] = 1
            ppi_label_list.append(temp_label)
            ppi_idx += 1
        else:
            index = ppi_dict[temp_data]
            temp_label = ppi_label_list[index]
            temp_label[class_map[relation_mode]] = 1
            ppi_label_list[index] = temp_label

    i = 0
    for ppi in tqdm(ppi_dict.keys()):
        name = ppi_dict[ppi]
        assert name == i
        i += 1
        temp = ppi.strip().split('__')
        ppi_list.append(temp)

    ppi_num = len(ppi_list)
    origin_ppi_list = copy.deepcopy(ppi_list)
    assert len(ppi_list) == len(ppi_label_list)

    for i in tqdm(range(ppi_num)):
        seq1_name = ppi_list[i][0]
        seq2_name = ppi_list[i][1]
        # print(len(protein_name))
        ppi_list[i][0] = protein_nodes[seq1_name]
        ppi_list[i][1] = protein_nodes[seq2_name]

    ppi_num = len(ppi_list)

    if split_mode == 'random':
        
        import json
        splits = json.load(open("/data/user/raojh/worksapce/MUSE/data/high_ppi/split/random/train_val_split_1.json", "r"))

        # random_list = [i for i in range(ppi_num)]
        # random.shuffle(random_list)

        train_index = splits['train_index']
        valid_index = splits['valid_index']
        # train_index = random_list[: int(ppi_num * (1 - test_size))]
        # valid_index = random_list[int(ppi_num * (1 - test_size)):]

    elif split_mode == 'bfs' or split_mode == 'dfs':
        node_to_edge_index = {}
        edge_num = ppi_num # int(edge_num // 2)
        print("edge_num: ", edge_num)
        print("ppi_list: ", len(ppi_list))
        for i in range(edge_num):
            edge = ppi_list[i]
            if edge[0] not in node_to_edge_index.keys():
                node_to_edge_index[edge[0]] = []
            node_to_edge_index[edge[0]].append(i)

            if edge[1] not in node_to_edge_index.keys():
                node_to_edge_index[edge[1]] = []
            node_to_edge_index[edge[1]].append(i)

        node_num = len(node_to_edge_index)

        sub_graph_size = int(edge_num * test_size)
        if split_mode == 'bfs':
            selected_edge_index = get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size)
        elif split_mode == 'dfs':
            selected_edge_index = get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size)

        all_edge_index = [i for i in range(edge_num)]
        unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

        train_index = unselected_edge_index
        valid_index = selected_edge_index

    elif split_mode == 'default':
        test_split_df = pd.read_csv(os.path.join(prefix, "raw", "high_ppi_split.tsv"), sep='\t', header=0, index_col=0)
        test_split_df['row'] = test_split_df.Protein_1.map(protein_nodes)
        test_split_df['col'] = test_split_df.Protein_2.map(protein_nodes)
        test_edges = test_split_df[['row', 'col']].values.tolist()
        test_edges_tuple = [tuple(i) for i in test_edges]
        edges_tuple = [tuple(i) for i in ppi_list]
        unselected_edges = list(set(edges_tuple).difference(set(test_edges_tuple)))
        train_edge = unselected_edges

        train_index = [edges_tuple.index(e) for e in train_edge] 
        valid_index = [edges_tuple.index(e) for e in test_edges_tuple]

    os.makedirs(f'{prefix}/split/{split_mode}', exist_ok=True)
    pd.DataFrame(sorted(train_index)).to_csv(f'{prefix}/split/{split_mode}/train.csv.gz', compression='gzip', header=False, index=False)
    pd.DataFrame(sorted(valid_index)).to_csv(f'{prefix}/split/{split_mode}/test.csv.gz', compression='gzip', header=False, index=False)

    train_edge = np.array(ppi_list).T[:, train_index].T
    valid_edge = np.array(ppi_list).T[:, valid_index].T

    train_label = np.array(ppi_label_list)[train_index]
    valid_label = np.array(ppi_label_list)[valid_index]

    torch.save(
        {"edge": train_edge, "label": train_label},
        os.path.join(prefix, "split", split_mode, "train.pt")
    )

    torch.save(
        {"edge": valid_edge, "label": valid_label},
        os.path.join(prefix, "split", split_mode, "test.pt")
    )


    pd.DataFrame(train_edge).to_csv(os.path.join(prefix, "raw/edge.csv.gz"), compression='gzip', header=False, index=False)
    pd.DataFrame(train_label).to_csv(os.path.join(prefix, "raw/edge-label.csv.gz"), compression='gzip', header=False, index=False)

    node_num = len(protein_nodes)
    edge_num = len(ppi_list)
    pd.DataFrame([node_num]).to_csv(os.path.join(prefix, "raw", "num-node-list.csv.gz"), compression='gzip', header=False, index=False)
    pd.DataFrame([edge_num]).to_csv(os.path.join(prefix, "raw", "num-edge-list.csv.gz"), compression='gzip', header=False, index=False)


    return {'train': torch.LongTensor(train_index), 'test': torch.LongTensor(valid_index)}



# NUM_ATOM_FEAT = 34
NUM_ATOM_FEAT = 144
# NUM_PROTEIN_FEAT = 100
NUM_PROTEIN_FEAT = 1956
NUM_PLAIN_FEAT = 200

PT = Chem.GetPeriodicTable()
RNG = rdNormalizedDescriptors.RDKit2DNormalized()

ELEMENT_LIST = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
    'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
    'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce',
    'Gd', 'Ga', 'Cs', 'unknown'
]

ATOM_CLASS_TABLE = {}
NOBLE_GAS_ATOMIC_NUM = {2, 10, 18, 36, 54, 86}
OTHER_NON_METAL_ATOMIC_NUM = {1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53}
METALLOID_ATOMIC_NUM = {5, 14, 32, 33, 51, 52, 85}
POST_TRANSITION_METAL_ATOMIC_NUM = {13, 31, 49, 50, 81, 82, 83, 84, 114}
TRANSITION_METAL_ATOMIC_NUM = set(range(21, 30 + 1)) | set(range(39, 48 + 1)) | set(range(72, 80 + 1)) | set(
    range(104, 108 + 1)) | {112}
ALKALI_METAL_ATOMIC_NUM = {3, 11, 19, 37, 55, 87}
ALKALI_EARCH_METAL_ATOMIC_NUM = {4, 12, 20, 38, 56, 88}
LANTHANOID_ATOMIC_NUM = set(range(57, 71 + 1))
ACTINOID_ATOMIC_NUM = set(range(89, 103 + 1))
ATOM_CLASSES = [
    NOBLE_GAS_ATOMIC_NUM, OTHER_NON_METAL_ATOMIC_NUM, METALLOID_ATOMIC_NUM, POST_TRANSITION_METAL_ATOMIC_NUM,
    TRANSITION_METAL_ATOMIC_NUM, ALKALI_EARCH_METAL_ATOMIC_NUM, ALKALI_METAL_ATOMIC_NUM, LANTHANOID_ATOMIC_NUM,
    ACTINOID_ATOMIC_NUM
]
for class_index, atom_class in enumerate(ATOM_CLASSES):
    for num in atom_class:
        ATOM_CLASS_TABLE[num] = class_index + 1

ALLEN_NEGATIVITY_TABLE = {
    1: 2.3,
    2: 4.16,
    3: 0.912,
    4: 1.576,
    5: 2.051,
    6: 2.544,
    7: 3.066,
    8: 3.61,
    9: 4.193,
    10: 4.787,
    11: 0.869,
    12: 1.293,
    13: 1.613,
    14: 1.916,
    15: 2.253,
    16: 2.589,
    17: 2.869,
    18: 3.242,
    19: 0.734,
    20: 1.034,
    21: 1.19,
    22: 1.38,
    23: 1.53,
    24: 1.65,
    25: 1.75,
    26: 1.8,
    27: 1.84,
    28: 1.88,
    29: 1.85,
    30: 1.59,
    31: 1.756,
    32: 1.994,
    33: 2.211,
    34: 2.424,
    35: 2.685,
    36: 2.966,
    37: 0.706,
    38: 0.963,
    39: 1.12,
    40: 1.32,
    41: 1.41,
    42: 1.47,
    43: 1.51,
    44: 1.54,
    45: 1.56,
    46: 1.58,
    47: 1.87,
    48: 1.52,
    49: 1.656,
    50: 1.824,
    51: 1.984,
    52: 2.158,
    53: 2.359,
    54: 2.582,
    55: 0.659,
    56: 0.881,
    71: 1.09,
    72: 1.16,
    73: 1.34,
    74: 1.47,
    75: 1.6,
    76: 1.65,
    77: 1.68,
    78: 1.72,
    79: 1.92,
    80: 1.76,
    81: 1.789,
    82: 1.854,
    83: 2.01,
    84: 2.19,
    85: 2.39,
    86: 2.6,
    87: 0.67,
    88: 0.89
}

ELECTRON_AFFINITY_TABLE = (
    (1, 0.75),
    (1, 0.75),
    (2, -0.52),
    (3, 0.62),
    (4, -0.52),
    (5, 0.28),
    (6, 1.26),
    (6, 1.26),
    (7, 0.00),
    (7, 0.01),
    (7, 0.01),
    (8, 1.46),
    (8, 1.46),
    (8, 1.46),
    (8, -7.71),
    (9, 3.40),
    (10, -1.20),
    (11, 0.55),
    (12, -0.41),
    (13, 0.43),
    (14, 1.39),
    (15, 0.75),
    (15, -4.85),
    (15, -9.18),
    (16, 2.08),
    (16, 2.08),
    (16, -4.72),
    (17, 3.61),
    (18, -1.00),
    (19, 0.50),
    (20, 0.02),
    (21, 0.19),
    (22, 0.08),
    (23, 0.53),
    (24, 0.68),
    (25, -0.52),
    (26, 0.15),
    (27, 0.66),
    (28, 1.16),
    (29, 1.24),
    (30, -0.62),
    (31, 0.43),
    (32, 1.23),
    (33, 0.80),
    (34, 2.02),
    (35, 3.36),
    (36, -0.62),
    (37, 0.49),
    (38, 0.05),
    (39, 0.31),
    (40, 0.43),
    (41, 0.92),
    (42, 0.75),
    (43, 0.55),
    (44, 1.05),
    (45, 1.14),
    (46, 0.56),
    (47, 1.30),
    (48, -0.72),
    (49, 0.30),
    (50, 1.11),
    (51, 1.05),
    (52, 1.97),
    (53, 3.06),
    (54, -0.83),
    (55, 0.47),
    (56, 0.14),
    (57, 0.47),
    (58, 0.65),
    (59, 0.96),
    (60, 1.92),
    (61, 0.13),
    (62, 0.16),
    (63, 0.86),
    (64, 0.14),
    (65, 1.17),
    (66, 0.35),
    (67, 0.34),
    (68, 0.31),
    (69, 1.03),
    (70, -0.02),
    (71, 0.35),
    (72, 0.02),
    (73, 0.32),
    (74, 0.82),
    (75, 0.06),
    (76, 1.10),
    (77, 1.56),
    (78, 2.13),
    (79, 2.31),
    (80, -0.52),
    (81, 0.38),
    (82, 0.36),
    (83, 0.94),
    (84, 1.90),
    (85, 2.30),
    (86, -0.72),
    (87, 0.49),
    (88, 0.10),
    (89, 0.35),
    (90, 1.17),
    (91, 0.55),
    (92, 0.53),
    (93, 0.48),
    (94, -0.50),
    (95, 0.10),
    (96, 0.28),
    (97, -1.72),
    (98, -1.01),
    (99, -0.30),
    (100, 0.35),
    (101, 0.98),
    (102, -2.33),
    (103, -0.31),
    (111, 1.57),
    (113, 0.69),
    (115, 0.37),
    (116, 0.78),
    (117, 1.72),
    (118, 0.06),
    (119, 0.66),
    (120, 0.02),
    (121, 0.57),
)
ELECTRON_AFFINITY_TABLE = {k: v for (k, v) in ELECTRON_AFFINITY_TABLE}


def one_of_k_encoding(x, allowable_set):
    # if x not in allowable_set:
    #     raise Exception("input {0} not in allowable set{1}:".format(
    #         x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H=False, use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, 'other'
    ]  # 6-dim
    results = (one_of_k_encoding_unk(atom.GetSymbol(), symbol) + one_of_k_encoding(atom.GetDegree(), degree) +
               [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
               one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]
               )  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'),
                                                      ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except Exception:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results


def atomic_features(atomic_num):
    # Symbol
    # symbol = PT.GetElementSymbol(atomic_num)
    # symbol_k = one_of_k_encoding_unk(symbol, ELEMENT_LIST)

    # Period
    outer_electrons = PT.GetNOuterElecs(atomic_num)
    outer_electrons_k = one_of_k_encoding(outer_electrons, list(range(0, 8 + 1)))

    # Default Valence
    default_electrons = PT.GetDefaultValence(atomic_num)  # -1 for transition metals
    default_electrons_k = one_of_k_encoding(default_electrons, list(range(-1, 8 + 1)))

    # Orbitals / Group / ~Row
    orbitals = next(j + 1 for j, val in enumerate([2, 10, 18, 36, 54, 86, 120]) if val >= atomic_num)
    orbitals_k = one_of_k_encoding(orbitals, list(range(0, 7 + 1)))

    # IUPAC Series
    atom_series = ATOM_CLASS_TABLE[atomic_num]
    atom_series_k = one_of_k_encoding(atom_series, list(range(0, 9 + 1)))

    # Centered Electrons
    centered_oec = abs(outer_electrons - 4)

    # Electronegativity & Electron Affinity
    try:
        allen_electronegativity = ALLEN_NEGATIVITY_TABLE[atomic_num]
    except KeyError:
        allen_electronegativity = 0
    try:
        electron_affinity = ELECTRON_AFFINITY_TABLE[atomic_num]
    except KeyError:
        electron_affinity = 0

    # Mass & Radius (van der waals / covalent / bohr 0)
    floats = [
        centered_oec, allen_electronegativity, electron_affinity,
        PT.GetAtomicWeight(atomic_num),
        PT.GetRb0(atomic_num),
        PT.GetRvdw(atomic_num),
        PT.GetRcovalent(atomic_num), outer_electrons, default_electrons, orbitals
    ]
    # print(symbol_k + outer_electrons_k + default_electrons_k + orbitals_k + atom_series_k + floats)

    # Compose feature array
    # feature_array = np.array(symbol_k + outer_electrons_k + default_electrons_k + orbitals_k + atom_series_k + floats,
    #                          dtype=np.float32)
    feature_array = np.array(outer_electrons_k + default_electrons_k + orbitals_k + atom_series_k + floats,
                             dtype=np.float32)
    # Cache in dict for future use
    return feature_array


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency) + np.eye(adjacency.shape[0])


def get_mol_features(smiles, atom_dim):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        raise RuntimeError("SMILES cannot been parsed!")
    # mol = Chem.AddHs(mol)
    # atom_feat = np.zeros((mol.GetNumAtoms(), NUM_ATOM_FEAT))
    atom_feat = np.zeros((mol.GetNumAtoms(), atom_dim))
    map_dict = dict()
    for atom in mol.GetAtoms():
        # atomic_features(atom.GetAtomicNum())
        # atom_feat[atom.GetIdx(), :] = np.append(atom_features(atom), atomic_features(atom.GetAtomicNum()))
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
        map_dict[atom.GetIdx()] = atom.GetSmarts()
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix, map_dict


def get_mol2vec_features(model, smiles):
    mol = Chem.MolFromSmiles(smiles)
    sen = (MolSentence(mol2alt_sentence(mol, 0)))
    mol_vec = (sentences2vec(sen, model, unseen='UNK'))
    return mol_vec


def molecule_collate_fn(batch):
    batch_size = len(batch)
    if not isinstance(batch[0], dict):
        new_batch = []
        for i in range(batch_size):
            new_batch.append(
                {"compound_node_features": batch[i].compound_node_features,
                 "compound_adj_matrix": batch[i].compound_adj_matrix,
                 "compound_word_embedding": batch[i].compound_word_embedding}
            )
        batch = new_batch

    compound_node_nums = [item['compound_node_features'].shape[0] for item in batch]
    max_compound_len = max(compound_node_nums)

    compound_node_features = torch.zeros((batch_size, max_compound_len, batch[0]['compound_node_features'].shape[1]))
    compound_adj_matrix = torch.zeros((batch_size, max_compound_len, max_compound_len))
    compound_word_embedding = torch.zeros(
        (batch_size, max_compound_len, batch[0]['compound_word_embedding'].shape[1]))

    for i, item in enumerate(batch):
        v = item['compound_node_features']
        compound_node_features[i, :v.shape[0], :] = torch.FloatTensor(v)
        v = item['compound_adj_matrix']
        compound_adj_matrix[i, :v.shape[0], :v.shape[0]] = torch.FloatTensor(v)
        v = item['compound_word_embedding']
        compound_word_embedding[i, :v.shape[0], :] = torch.FloatTensor(v)
    compound_node_nums = torch.LongTensor(compound_node_nums)
    return {
        'compound_node_feat': compound_node_features,
        'compound_adj': compound_adj_matrix,
        'compound_node_num': compound_node_nums,
        'compound_word_embedding': compound_word_embedding,
    }


# def protein_collate_fn(batch):
#     batch_size = len(batch)

#     protein_node_nums = [item.protein_node_feat.shape[0] for item in batch]
#     max_protein_len = max(protein_node_nums)

#     protein_node_features = torch.zeros((batch_size, max_protein_len, batch[0].protein_node_feat.shape[1]))
#     protein_contact_map = torch.zeros((batch_size, max_protein_len, max_protein_len))
#     protein_seq_embedding = torch.zeros((batch_size, max_protein_len, batch[0].protein_embedding.shape[1]))

#     for i, item in enumerate(batch):
#         v = item.protein_node_feat
#         protein_node_features[i, :v.shape[0], :] = torch.FloatTensor(v)
#         v = item.protein_map
#         protein_contact_map[i, :v.shape[0], :v.shape[0]] = torch.FloatTensor(v)

#         v = item.protein_embedding
#         # if args.pretrained == 1 and args.objective == 'classification':
#         v[:, 358] = (v[:, 358] - 7.8) / 6.5
#         protein_seq_embedding[i, :v.shape[0], :] = torch.FloatTensor(v)[:max_protein_len, :]

#     protein_node_nums = torch.LongTensor(protein_node_nums)

#     return {
#         'protein_node_feat': protein_node_features,
#         'protein_map': protein_contact_map,
#         'protein_embedding': protein_seq_embedding,
#         'protein_node_num': protein_node_nums,
#     }

def protein_collate_fn(batch):
    protein_node_nums = [item.x.shape[0] for item in batch]
    return {
        "protein_node_num": torch.LongTensor(protein_node_nums),
        "protein_data": Batch.from_data_list(batch),
    }

def check_gpus():
    '''
    GPU available check
    http://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-cuda/
    '''
    if not torch.cuda.is_available():
        print('This script could only be used to manage NVIDIA GPUs,but no GPU found in your device')
        return False
    elif not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
        print("'nvidia-smi' tool not found.")
        return False
    return True



def check_gpus():
    '''
    GPU available check
    http://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-cuda/
    '''
    if not torch.cuda.is_available():
        print('This script could only be used to manage NVIDIA GPUs,but no GPU found in your device')
        return False
    elif not 'NVIDIA System Management' in os.popen('nvidia-smi -h').read():
        print("'nvidia-smi' tool not found.")
        return False
    return True


if check_gpus():
    def parse(line, qargs):
        """
        line:
            a line of text
        qargs:
            query arguments
        return:
            a dict of gpu infos
        Pasing a line of csv format text returned by nvidia-smi
        解析一行nvidia-smi返回的csv格式文本
        """
        numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']  # 可计数的参数
        power_manage_enable = lambda v: (not 'Not Support' in v)  # lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
        to_numberic = lambda v: float(v.upper().strip().replace('MIB', '').replace('W', ''))  # 带单位字符串去掉单位
        process = lambda k, v: (
            (int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
        return {k: process(k, v) for k, v in zip(qargs, line.strip().split(','))}


    def query_gpu(qargs=[]):
        """
        qargs:
            query arguments
        return:
            a list of dict
        Querying GPUs infos
        查询GPU信息
        """
        qargs = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit'] + qargs
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = os.popen(cmd).readlines()
        return [parse(line.replace(", [N/A]", ""), qargs) for line in results]


    def by_power(d):
        """
        helper function fo sorting gpus by power
        """
        power_infos = (d['power.draw'], d['power.limit'])
        if any(v == 1 for v in power_infos):
            print('Power management unable for GPU {}'.format(d['index']))
            return 1
        return float(d['power.draw']) / d['power.limit']


    class GPUManager():
        """
        qargs:
            query arguments
        A manager which can list all available GPU devices
        and sort them and choice the most free one.Unspecified
        ones pref.
        GPU设备管理器，考虑列举出所有可用GPU设备，并加以排序，自动选出
        最空闲的设备。在一个GPUManager对象内会记录每个GPU是否已被指定，
        优先选择未指定的GPU。
        """

        def __init__(self, qargs=[]):
            '''
            '''
            self.qargs = qargs
            self.gpus = query_gpu(qargs)
            for gpu in self.gpus:
                gpu['specified'] = False
            self.gpu_num = len(self.gpus)

        def _sort_by_memory(self, gpus, by_size=False):
            if by_size:
                print('Sorted by free memory size')
                return sorted(gpus, key=lambda d: d['memory.free'], reverse=True)
            else:
                print('Sorted by free memory rate')
                return sorted(gpus, key=lambda d: float(d['memory.free']) / d['memory.total'], reverse=True)

        def _sort_by_power(self, gpus):
            return sorted(gpus, key=by_power)

        def _sort_by_custom(self, gpus, key, reverse=False, qargs=[]):
            if isinstance(key, str) and (key in qargs):
                return sorted(gpus, key=lambda d: d[key], reverse=reverse)
            if isinstance(key, type(lambda a: a)):
                return sorted(gpus, key=key, reverse=reverse)
            raise ValueError(
                "The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")

        def auto_choice(self, num=1, mode=0):
            '''
            mode:
                0:(default)sorted by free memory size
            return:
                a TF device object
            Auto choice the freest GPU device,not specified
            ones 
            自动选择最空闲GPU,返回索引
            '''
            for old_infos, new_infos in zip(self.gpus, query_gpu(self.qargs)):
                old_infos.update(new_infos)
            unspecified_gpus = [gpu for gpu in self.gpus if not gpu['specified']] or self.gpus

            if mode == 0:
                print('Choosing the GPU device has largest free memory...')
                chosen_gpus = self._sort_by_memory(unspecified_gpus, True)[:num]
            elif mode == 1:
                print('Choosing the GPU device has highest free memory rate...')
                chosen_gpus = self._sort_by_power(unspecified_gpus)[:num]
            elif mode == 2:
                print('Choosing the GPU device by power...')
                chosen_gpus = self._sort_by_power(unspecified_gpus)[:num]
            else:
                print('Given an unaviliable mode,will be chosen by memory')
                chosen_gpus = self._sort_by_memory(unspecified_gpus)[:num]
            
            if num == 1:
                chosen_gpu = chosen_gpus[0]
                chosen_gpu['specified'] = True
                index = chosen_gpu['index']
                print('Using GPU {i}:\n{info}'.format(i=index, info='\n'.join(
                    [str(k) + ':' + str(v) for k, v in chosen_gpu.items()])))
                return int(index)
            else:
                index_list = []
                for chosen_gpu in chosen_gpus:
                    chosen_gpu['specified'] = True
                    index = chosen_gpu['index']
                    print('Using GPU {i}:\n{info}'.format(i=index, info='\n'.join(
                    [str(k) + ':' + str(v) for k, v in chosen_gpu.items()])))
                    index_list.append(int(index))
                return tuple(index_list)

                
else:
    raise ImportError('GPU available check failed')