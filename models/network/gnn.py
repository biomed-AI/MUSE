#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Copyright (c) 2023, Sun Yat-sen Univeristy.
All rights reserved.

@author:   Jiahua Rao
@license:  BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact:  jiahua.rao@gmail.com
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from models.network.utils import *
from torch_sparse.matmul import spmm_add


# Edge dropout
class DropEdge(nn.Module):

    def __init__(self, dp: float = 0.0) -> None:
        super().__init__()
        self.dp = dp

    def forward(self, edge_index: Tensor):
        if self.dp == 0 or not self.training:
            return edge_index
        mask = torch.rand_like(edge_index[0], dtype=torch.float) > self.dp
        return edge_index[:, mask]
    

class GIN(torch.nn.Module):
    def __init__(self, in_dim=128, hidden=512, train_eps=True, class_num=7):
        super(GIN, self).__init__()
        self.train_eps = train_eps
        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                # nn.Linear(hidden, hidden),
                # nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv3 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        # self.drop_edge = DropEdge(0.3)
        self.lin1 = nn.Linear(hidden, hidden)
        self.fc1 = nn.Linear(2 * hidden, class_num) #clasifier for concat
        self.fc2 = nn.Linear(hidden, class_num)   #classifier for inner product

        hidden_channels = hidden
        dropout = 0.2
        ln = True
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()
        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                  nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_channels, hidden_channels),
                                  lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True))

        self.xcnlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels))

        self.xijlin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels))
        
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, class_num))

        self.beta = nn.Parameter(1.0*torch.ones((1)))

    def reset_parameters(self):
        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()
        self.gin_conv3.reset_parameters()
        self.lin1.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        for _, layer in self.xlin.named_children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for _, layer in self.xcnlin.named_children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for _, layer in self.xijlin.named_children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for _, layer in self.lin.named_children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


    def multidomainforward(self, x, adj, edges):
        # adj = self.dropadj(adj)
        xi = x[edges[0]]
        xj = x[edges[1]]

        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, edges, False, cnsampledeg=False)
        xcns = [spmm_add(cn, x)]
        xij = self.xijlin(xi * xj)
        
        xs = torch.stack(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs.mean(dim=-1)
    
    def default_forward(self, x, edges):
        x1 = x[edges[0]]
        x2 = x[edges[1]]
        x = torch.mul(x1, x2)
        x = self.fc2(x)
        return x

    def featurization(self, x, edge_index, adj, pos_edge, neg_edge=None):
        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)
        # x = self.gin_conv3(x, edge_index)
        x = self.lin1(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # pos_outs = self.multidomainforward(x, adj, pos_edge)
        xi = x[pos_edge[0]]
        xj = x[pos_edge[1]]
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, pos_edge, False, cnsampledeg=False)
        xcns = [spmm_add(cn, x)]
        xij = self.xijlin(xi * xj)
        
        xs = torch.stack(
            [self.xcnlin(xcn) * self.beta + xij for xcn in xcns],
            dim=-1)
        return xs

    def forward(self, x, edge_index, adj, pos_edge, neg_edge=None):
        # x, edge_index = data.x, data.edge_index

        # edge_index = self.drop_edge(edge_index)
        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)
        # x = self.gin_conv3(x, edge_index)
        x = self.lin1(x)
        x = F.dropout(x, p=0.1, training=self.training)

        pos_outs = self.multidomainforward(x, adj, pos_edge)
        # pos_outs = self.default_forward(x, pos_edge)
        
        if neg_edge is None:
            return pos_outs
        else:
            # neg_outs = self.default_forward(x, neg_edge)
            neg_outs = self.multidomainforward(x, adj, neg_edge)
            return pos_outs, neg_outs
