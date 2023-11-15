#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author:   Jiahua Rao
@license:  BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact:  jiahua.rao@gmail.com
@time:     2023-06-11
'''


import os
import copy
import json

from time import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling, add_self_loops

from config import parse_args, Config
from utils import write_log, Metrictor_DPI, GPUManager
from utils import molecule_collate_fn, protein_collate_fn
from dataset import DrugProteinDataset, LinkGraphDataset
from models import GNN_DPI, NetGNN
from optim import RAdam, Lookahead
import warnings
warnings.filterwarnings('ignore')


class EMTrainer:
    """
    Expectation-maximization Trainer.
    """

    def __init__(self, args, cfg):
        self.args = args
        self.config = cfg
        self.device = GPUManager().auto_choice()
        self.logging = open(self.args.logging_file, 'a', buffering=1)
        self.em_range = range(self.config.run_config.em_total_iters)

        inter_dataset_config = self.config.inter_dataset_config
        dataset_name = list(inter_dataset_config.keys())[0]
        self.inter_dataset_root = args.inter_data_dir
        self.inter_dataset_config = inter_dataset_config[dataset_name]

        with open(self.args.logging_file, "a") as f:
            f.write(json.dumps(vars(self.args)) + "\n")

        self.gnn_model = self.bulid_intra_model()
        self.link_dataset = LinkGraphDataset(dataset_name, self.inter_dataset_config, root=os.path.join(self.inter_dataset_root, dataset_name.replace('-', '_')))
        self.link_model = self.build_inter_model()

    def bulid_intra_model(self):
        intra_model_cfg = self.config.model_config.intra_model
        model = GNN_DPI(args=intra_model_cfg)
        return model

    def build_inter_model(self):
        model = NetGNN(in_dim=self.config.model_config.intra_model.hid_dim,
                       hidden=self.config.model_config.inter_model.hidden,
                       class_num=1)
        return model

    def multi_scale_em_train(self):
        best_aupr, self.best_em_iter = 0, -1
        molecule_embeddings, protein_embeddings, link_graph = None, None, None
        for self.em_iter in self.em_range:
            self.gnn_model, _ = self._maximization(link_model=self.link_model,
                                                   link_graph=link_graph,
                                                   molecule_embeddings=molecule_embeddings,
                                                   protein_embeddings=protein_embeddings)

            molecule_embeddings, protein_embeddings = self.gnn_trainer.get_graph_embedding(self.gnn_model)
            self.link_model, link_outputs = self._expectation(gnn_model=self.gnn_model, molecule_embeddings=molecule_embeddings, protein_embeddings=protein_embeddings)

            link_graph = link_outputs['graph']
            if link_outputs['best_valid_aupr'] > best_aupr:
                best_aupr = link_outputs['best_valid_aupr']
                self.best_em_iter = self.em_iter
                torch.save({'em_iter': self.em_iter,
                            'best_aupr': best_aupr,
                            'link_graph': link_graph,
                            'protein_embeddings': protein_embeddings,
                            'molecule_embeddings': molecule_embeddings,
                            'gnn_state_dict': self.gnn_model.state_dict(),
                            'link_state_dict': self.link_model.state_dict()},
                            os.path.join(self.args.res_dir, f'best_model.ckpt'))

            if self.em_iter == (self.em_range.stop - 1):
                self.final_prediction()

    def _maximization(self, link_model=None, link_graph=None, molecule_embeddings=None, protein_embeddings=None):
        # Protein GNN training
        write_log(self.logging, f'\n <<<<<<<<<< Protein GNN training >>>>>>>>>>')
        self.gnn_model.reset_parameters()
        self.gnn_trainer = DrugProteinTrainer(args=self.args,
                                              config=self.config,
                                              model=self.gnn_model,
                                              dataset=self.link_dataset,
                                              device=self.device,
                                              logging=self.logging)
        
        if protein_embeddings is None:
            molecule_embeddings, protein_embeddings = self.gnn_trainer.get_graph_embedding(self.gnn_model)

        self.gnn_model, gnn_outputs = self.gnn_trainer.train(link_model=link_model,
                                                             link_graph=link_graph,
                                                             molecule_embeddings=molecule_embeddings,
                                                             protein_embeddings=protein_embeddings,
                                                             em_iter=self.em_iter)

        msg = {
                'em_iter': self.em_iter,
                **{f'gnn_{k}': v for k, v in gnn_outputs.items()},
            }
        with open(self.args.em_log_file, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return self.gnn_model, gnn_outputs

    def _expectation(self, gnn_model=None, molecule_embeddings=None, protein_embeddings=None):
        # Link GNN training
        write_log(self.logging, f'\n <<<<<<<<<< Link GNN training >>>>>>>>>>')
        self.link_model.reset_parameters()
        protein_dataset = self.gnn_trainer.train_loader.dataset.protein_dataset
        molecule_dataset = self.gnn_trainer.train_loader.dataset.molecule_dataset
        self.link_trainer = DPILinkTrainer(args=self.args,
                                           config=self.config,
                                           model=self.link_model,
                                           dataset=self.link_dataset,
                                           device=self.device,
                                           logging=self.logging)
        self.link_model, link_outputs = self.link_trainer.train(gnn_model=gnn_model,
                                                                molecule_dataset=molecule_dataset,
                                                                protein_dataset=protein_dataset,
                                                                molecule_embeddings=molecule_embeddings,
                                                                protein_embeddings=protein_embeddings,
                                                                em_iter=self.em_iter)

        msg = {
                'em_iter': self.em_iter,
                **{f'link_{k}': v for k, v in link_outputs.items() if k != 'graph'},
            }
        with open(self.args.em_log_file, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return self.link_model, link_outputs

    def final_prediction(self, best_em_iter=None):
        best_em_iter = self.best_em_iter if best_em_iter is None else best_em_iter
        final_link_model = copy.deepcopy(self.link_model)
        final_link_model.load_state_dict(torch.load(os.path.join(self.args.res_dir, f'link_{best_em_iter}_best_model.ckpt'))['state_dict'])
        pseudo_graph = torch.load(os.path.join(self.args.res_dir, f'link_{best_em_iter}_best_model.ckpt'))['graph']

        link_trainer = DPILinkTrainer(args=self.args,
                                           config=self.config,
                                           model=final_link_model,
                                           dataset=self.link_dataset,
                                           device=self.device,
                                           logging=self.logging)
    
        if pseudo_graph is not None:
            pseudo_graph = link_trainer.graph

        _, test_metrics = link_trainer.evaluate(graph=pseudo_graph, adj=pseudo_graph.full_adj_t, save_probs=True, em_iter=best_em_iter)

        write_log(self.logging,
            "em_iter: {} Test AUC: {}, AUPR: {}, Recall: {}, Precision: {}, F1: {}"
                .format(best_em_iter, test_metrics['auc'], test_metrics['aupr'], test_metrics['recall'], test_metrics['precision'], test_metrics['f1']))
        return


class DrugProteinTrainer:
    def __init__(self, args, config, model, dataset, device, logging=None) -> None:
        self.args = args
        self.config = config
        self.link_dataset = dataset
        self.run_config = self.config.run_config
        self.patience = self.run_config.patience
        self.logging = logging
        self.device = torch.device(device)

        self.train_loader, self.valid_loader, self.test_loader = self.create_dataloaders()
        self.graph = self.link_dataset[0]
        self.test_edges = self.link_dataset.get_edge_split()['test']['edge']

        self.model = model.to(self.device)
        optimizer_inner = RAdam(self.model.parameters(), lr=self.run_config.learning_rate, weight_decay=1e-4)
        self.optimizer = optimizer_inner# Lookahead(optimizer_inner, k=5, alpha=0.5)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10,
                                                                    verbose=True)
        self.evaluator = Metrictor_DPI
        self.load_best_model = True

    def create_dataloaders(self):
        train_dataset = DrugProteinDataset(self.args, self.config, split='train')
        valid_dataset = DrugProteinDataset(self.args, self.config, split='valid')
        test_dataset = DrugProteinDataset(self.args, self.config, split='test')

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.run_config.batch_size,
                                  shuffle=True,
                                  sampler=None,
                                  collate_fn=train_dataset.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.args.num_workers,
                                  prefetch_factor=self.args.prefetch)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.run_config.batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  sampler=None,
                                  collate_fn=valid_dataset.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.args.num_workers,
                                  prefetch_factor=self.args.prefetch)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.run_config.batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 sampler=None,
                                 collate_fn=test_dataset.collate_fn,
                                 pin_memory=True,
                                 num_workers=self.args.num_workers,
                                 prefetch_factor=self.args.prefetch)
        return train_loader, valid_loader, test_loader

    def _train_epoch(self, loader, link_model=None, molecule_embeddings=None, protein_embeddings=None, pseudo_labels=None):
        self.model.train()

        steps = 0
        loss_sum = 0.0

        # num_batches = len(loader)
        # pseudo_batch_size = int(len(self.test_edges) // num_batches) + 1
        torch.cuda.empty_cache()
        unobserved_losses = self.eval_pseudo_step(self.model, self.test_edges, pseudo_labels)

        for step, batch in enumerate(self.train_loader):
            for k, v in batch.items():
                batch[k] = v.to(self.device, non_blocking=True)

            interactions = batch["label"].long()
            output = self.model(batch)
            loss = self.criterion(output, interactions.squeeze(-1))

            if self.is_augmented and self.run_config.pl_ratio > 0:
                pl_weight = self.run_config.pl_ratio
                _, observed_labels = self.link_inference(link_model, batch['edge'], self.graph, molecule_embeddings, protein_embeddings)

                pl_loss = self.criterion(output, observed_labels.squeeze(-1)) + unobserved_losses
                loss = pl_weight * pl_loss + (1 - pl_weight) * loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_sum += loss.item()
            steps += 1

            torch.cuda.empty_cache()

        return loss_sum / steps

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.model.eval()

        valid_label_list = []
        valid_prob_list = []
        valid_loss_sum = 0.0
        valid_steps = 0

        for step, batch in enumerate(test_loader):

            for k, v in batch.items():
                batch[k] = v.to(self.device, non_blocking=True)

            output = self.model(batch)
            interactions = batch["label"].long()
            loss = self.criterion(output, interactions.squeeze(-1))
            valid_loss_sum += loss.item()

            scores = F.softmax(output, dim=1)[:, 1].to("cpu").data.tolist()
            correct_labels = interactions.to("cpu").data.tolist()

            valid_label_list.extend(correct_labels)
            valid_prob_list.extend(scores)
            valid_steps += 1

        valid_loss = valid_loss_sum / valid_steps
        valid_label_list = np.array(valid_label_list)
        valid_prob_list = np.array(valid_prob_list)

        valid_metrics = self.evaluator(valid_label_list, valid_prob_list)
        return valid_loss, valid_metrics

    def train(self, link_model=None, link_graph=None, molecule_embeddings=None, protein_embeddings=None, em_iter=0):
        self.is_augmented = self.run_config.is_augmented and link_model is not None and protein_embeddings is not None
        pseudo_labels = None
        if link_model is not None and protein_embeddings is not None:
            self.graph = link_graph if link_graph is not None else self.graph
            link_model = link_model.to(self.device)
            self.graph.adj_t = SparseTensor.from_edge_index(self.graph.edge_index, sparse_sizes=(self.graph.num_nodes, self.graph.num_nodes)).to(self.device)
            self.graph.adj_t = self.graph.adj_t.to_symmetric().coalesce()
            _, pseudo_labels = self.link_inference(link_model, self.test_edges, self.graph, molecule_embeddings, protein_embeddings)

        stopper = 0
        best_valid_aupr = 0.0
        best_valid_aupr_epoch = 0

        # ! GNN Training
        for self.cur_epoch in range(self.run_config.gnn_epochs):
            start_time = time()
            loss = self._train_epoch(self.train_loader, link_model=link_model, molecule_embeddings=molecule_embeddings, protein_embeddings=protein_embeddings, pseudo_labels=pseudo_labels)
            valid_loss, valid_metrics = self.evaluate(self.valid_loader)
            test_loss, test_metrics = self.evaluate(self.test_loader)

            if self.scheduler != None:
                self.scheduler.step(loss)
                write_log(self.logging, "epoch: {}, now learning rate: {}".format(self.cur_epoch, self.scheduler.optimizer.param_groups[0]['lr']))

            if best_valid_aupr < valid_metrics["aupr"]:
                best_valid_aupr = valid_metrics["aupr"]
                best_valid_aupr_epoch = self.cur_epoch

                torch.save({'epoch': self.cur_epoch,
                            'state_dict': self.model.state_dict()},
                            os.path.join(self.args.res_dir, f'gnn_{em_iter}_best_model.ckpt'))
                stopper = 0

            write_log(self.logging,
                "epoch: {}, time {}, Training_avg: label_loss: {}, Validation_avg: loss: {}, auc: {}, aupr: {}, precision: {}, recall: {}, f1: {}, Test_avg: loss: {}, auc: {}, aupr: {}, precision: {}, recall: {}, f1: {}, Best valid_aupr: {}, in {} epoch"
                    .format(self.cur_epoch, time() - start_time, loss, valid_loss, valid_metrics["auc"], valid_metrics["aupr"], valid_metrics["precision"], valid_metrics["recall"], valid_metrics["f1"], test_loss, test_metrics["auc"], test_metrics["aupr"], test_metrics["precision"], test_metrics["recall"], test_metrics["f1"],
                            best_valid_aupr, best_valid_aupr_epoch))

            stopper += 1
            if stopper > self.patience:
                write_log(self.logging, "Early Stopping.")
                break

        # ! Finished training, load checkpoints
        if self.load_best_model:
            self.model.load_state_dict(torch.load(os.path.join(self.args.res_dir, f'gnn_{em_iter}_best_model.ckpt'))['state_dict'])
        return self.model, {"best_valid_aupr": best_valid_aupr, "best_valid_aupr_epoch": best_valid_aupr_epoch}

    def get_graph_embedding(self, model):
        protein_dataset = self.train_loader.dataset.protein_dataset

        loader = DataLoader(protein_dataset,
                            batch_size=4,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=protein_collate_fn,
                            num_workers=self.args.num_workers)

        protein_embeddings = []
        with torch.no_grad():
            for protein_data in loader:
                for k, v in protein_data.items():
                    protein_data[k] = v.to(self.device, non_blocking=True)
                protein_embeddings.append(model.get_protein_representation(protein_data).cpu())
        protein_embeddings = torch.cat(protein_embeddings, dim=0)

        torch.cuda.empty_cache()

        molecule_dataset = self.train_loader.dataset.molecule_dataset
        loader = DataLoader(molecule_dataset,
                            batch_size=4,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=molecule_collate_fn,
                            num_workers=self.args.num_workers)

        molecule_embeddings = []
        with torch.no_grad():
            for molecule_data in loader:
                for k, v in molecule_data.items():
                    molecule_data[k] = v.to(self.device, non_blocking=True)
                molecule_embeddings.append(model.get_mol_representation(molecule_data).cpu())
        molecule_embeddings = torch.cat(molecule_embeddings, dim=0)
        torch.cuda.empty_cache()
        return protein_embeddings, molecule_embeddings

    @torch.no_grad()
    def eval_pseudo_step(self, model, test_edges, pseudo_labels):
        molecule_num = len(self.train_loader.dataset.molecule_dataset)
        loss = 0
        for step, perm_idx in enumerate(DataLoader(range(test_edges.size(0)), batch_size=1, shuffle=False)):
            edges = test_edges[perm_idx]
            labels = pseudo_labels[perm_idx]
            # edges = self.test_edges[batch_index*batch_size : (batch_index+1)*batch_size]
            batch = {
                "edge": edges.to(torch.long),
                **molecule_collate_fn([self.train_loader.dataset.molecule_dataset[edge[0]] for edge in edges]),
                **protein_collate_fn([self.train_loader.dataset.protein_dataset[edge[1] - molecule_num] for edge in edges]),
            }

            for k, v in batch.items():
                batch[k] = v.to(self.device, non_blocking=True)

            unobserved_output = model(batch)
            loss += self.criterion(unobserved_output, labels.squeeze(-1)) 
        
        torch.cuda.empty_cache()
        return loss

    def link_inference(self, link_model, test_edges, graph, molecule_embeddings, protein_embeddings):
        embeddings = torch.cat([molecule_embeddings, protein_embeddings], dim=0)
        graph.x = torch.FloatTensor(embeddings.cpu())
        graph = graph.to(self.device)
        adj = graph.adj_t

        pseudo_preds = []
        m = torch.nn.Sigmoid()
        for step, perm_idx in enumerate(DataLoader(range(test_edges.size(0)), batch_size=self.run_config.batch_size, shuffle=False)):
            output = link_model(graph.x, graph.edge_index, adj, test_edges[perm_idx].t().to(self.device))
            pseudo_preds.append(output.cpu().data)
        pseudo_preds = torch.cat(pseudo_preds, dim = 0)
        pseudo_labels = (pseudo_preds.sigmoid() > 0.5).type(torch.long).to(device=self.device)
        return pseudo_preds, pseudo_labels


class DPILinkTrainer:
    def __init__(self, args, config, model, dataset, device, logging) -> None:
        self.args = args
        self.config = config
        self.logging = logging
        self.dataset = dataset
        self.device = torch.device(device)

        self.train_data, self.valid_data, self.test_data = self.create_dataloaders()
        self.train_edges, self.valid_edges, self.test_edges = self.train_data['edge'], self.valid_data['edge'], self.test_data['edge']
        # self.train_label, self.test_label = self.train_data['label'], self.test_data['label']

        self.graph = self.dataset[0].to(device)
        self.model = model.to(self.device)
        self.run_config = self.config.run_config
        self.patience = self.run_config.patience

        optimizer_inner = RAdam(self.model.parameters(), lr=self.run_config.learning_rate, weight_decay=1e-4)
        self.optimizer = optimizer_inner # Lookahead(optimizer_inner, k=5, alpha=0.5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10,
                                                                    verbose=True)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = Metrictor_DPI
        self.load_best_model = True

    def create_dataloaders(self):
        split_edges = self.dataset.get_edge_split()
        return split_edges['train'], split_edges['valid'], split_edges['test']

    def _train_epoch(self, graph, train_edges):
        self.model.train()

        steps = 0
        loss_sum = 0.0
        train_label_list = []
        train_prob_list = []


        data = graph.to(self.device)
        train_edges = train_edges.to(self.device).t()
        adjmask = torch.ones_like(train_edges[0], dtype=torch.bool)
        new_edge_index, _ = add_self_loops(data.edge_index)
        negedge = negative_sampling(new_edge_index.to(train_edges.device), data.adj_t.sizes()[0])
        for step, perm_idx in enumerate(DataLoader(range(train_edges.size(1)), self.run_config.batch_size, shuffle=True)):
            # Target Link Masking
            adjmask[perm_idx] = 0
            tei = train_edges[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                               train_edges.device, non_blocking=True)
            adjmask[perm_idx] = 1
            adj = adj.to_symmetric()

            # labels = train_label[perm_idx].type(torch.FloatTensor).to(self.device)

            # output = self.model(self.graph, train_edges[perm_idx].t().to(self.device))
            # loss = self.loss_fn(output, labels)
            pos_edge, neg_edge = train_edges[:, perm_idx], negedge[:, perm_idx]
            output, neg_output = self.model(data.x, data.edge_index, adj, pos_edge, neg_edge)

            # pos_loss = self.criterion(output, labels) 
            # neg_loss = self.criterion(neg_output, torch.zeros(neg_output.shape).to(self.device))
            pos_loss = -F.logsigmoid(output).mean()
            neg_loss = -F.logsigmoid(-neg_output).mean()
            loss = pos_loss + 0.5 * neg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # scores = F.sigmoid(output, dim=1).to("cpu").data.tolist()
            # correct_labels = labels.to("cpu").data.tolist()

            # train_label_list.extend(correct_labels)
            # train_prob_list.extend(scores)
            loss_sum += loss.item()
        
            steps += 1

        train_label_list = np.array(train_label_list)
        train_prob_list = np.array(train_prob_list)

        train_auc = 0 # roc_auc_score(train_label_list, train_prob_list)
        return loss_sum / steps, train_auc

    @torch.no_grad()
    def evaluate(self, graph=None, adj=None, test_edges=None, test_neg_edges=None, save_probs=False, em_iter=0):
        self.model.eval()

        if test_edges is None:
            test_edges = self.test_edges
            test_neg_edges = self.test_data['neg_edge']

        data = graph.to(self.device)
        # adj = data.adj_t

        valid_label_list = []
        valid_prob_list = []
        for step, perm_idx in enumerate(DataLoader(range(test_edges.size(0)), batch_size=self.run_config.batch_size, shuffle=False)):
            output = self.model(data.x, data.edge_index, adj, test_edges[perm_idx].t().to(self.device))
            # scores = F.softmax(output, dim=1)[:, 1].to("cpu").data.tolist()
            # correct_labels = labels.to("cpu").data.tolist()
            # valid_label_list.extend(correct_labels)
            valid_prob_list.extend(output.sigmoid().squeeze(-1).data.tolist())
            valid_label_list.extend(torch.ones(output.size(0),).cpu().data.tolist())

        for step, perm_idx in enumerate(DataLoader(range(test_neg_edges.size(0)), batch_size=self.run_config.batch_size, shuffle=False)):
            output = self.model(data.x, data.edge_index, adj, test_neg_edges[perm_idx].t().to(self.device))
            # scores = F.softmax(output, dim=1)[:, 1].to("cpu").data.tolist()
            # correct_labels = labels.to("cpu").data.tolist()
            # valid_label_list.extend(correct_labels)
            valid_prob_list.extend(output.sigmoid().squeeze(-1).data.tolist())
            valid_label_list.extend(torch.zeros(output.size(0),).cpu().data.tolist())

        valid_prob_list = np.array(valid_prob_list)
        valid_label_list = np.array(valid_label_list)

        valid_metrics = self.evaluator(valid_label_list, valid_prob_list)

        if save_probs:
            torch.save(
                {"predictions": valid_prob_list, "labels": valid_label_list},
                os.path.join(self.args.res_dir, f'prediction_link_{em_iter}_results.ckpt')
            )
        return valid_metrics

    def train(self, gnn_model=None, molecule_dataset=None, protein_dataset=None, molecule_embeddings=None, protein_embeddings=None, em_iter=0):
        self.is_augmented = self.run_config.is_augmented and gnn_model is not None and protein_dataset is not None
        if protein_embeddings is not None and molecule_embeddings is not None:
            embeddings = torch.cat([molecule_embeddings, protein_embeddings], dim=0)
            self.graph.x = torch.FloatTensor(embeddings.cpu())

        write_log(self.logging, f"Link Graph has {self.graph.edge_index.shape[1]} edges.")
        if gnn_model is not None:
            pseudo_edges = self.gnn_inference(gnn_model, molecule_dataset, protein_dataset)
            self.graph.edge_index = torch.cat([self.graph.edge_index, pseudo_edges.to(self.device)], dim=1)

        self.graph.adj_t = SparseTensor.from_edge_index(self.graph.edge_index, sparse_sizes=(self.graph.num_nodes, self.graph.num_nodes)).to(self.device)
        self.graph.adj_t = self.graph.adj_t.to_symmetric().coalesce()

        val_edge_index = self.valid_edges.t().to(self.device)
        full_edge_index = torch.cat([self.graph.edge_index, val_edge_index], dim=-1)
        self.graph.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(self.graph.num_nodes, self.graph.num_nodes)).coalesce()
        self.graph.full_adj_t = self.graph.full_adj_t.to_symmetric()

        stopper = 0

        best_valid_aupr = 0.0
        best_valid_aupr_epoch = 0

        # ! Link Training
        for self.cur_epoch in range(self.run_config.link_epochs):
            start_time = time()
            loss, train_auc = self._train_epoch(self.graph, self.train_edges)
            valid_metrics = self.evaluate(self.graph, self.graph.adj_t, self.valid_edges, self.valid_data['neg_edge'])
            test_metrics = self.evaluate(self.graph, self.graph.full_adj_t, self.test_edges, self.test_data['neg_edge'])

            if self.scheduler != None:
                self.scheduler.step(loss)
                write_log(self.logging, "epoch: {}, now learning rate: {}".format(self.cur_epoch, self.scheduler.optimizer.param_groups[0]['lr']))

            if best_valid_aupr < test_metrics["aupr"]:
                best_valid_aupr = test_metrics["aupr"]
                best_valid_aupr_epoch = self.cur_epoch

                torch.save({'epoch': self.cur_epoch,
                            'state_dict': self.model.state_dict(),
                            'graph': self.graph},
                            os.path.join(self.args.res_dir, f'link_{em_iter}_best_model.ckpt'))
                stopper = 0

            write_log(self.logging,
                "epoch: {}, time {}, Training_avg: label_loss: {}, Validation_avg: auc: {}, aupr: {}, precision: {}, recall: {}, f1: {}, Test_avg: auc: {}, aupr: {}, precision: {}, recall: {}, f1: {}, Best valid_aupr: {}, in {} epoch"
                    .format(self.cur_epoch, time() - start_time, loss, valid_metrics["auc"], valid_metrics["aupr"], valid_metrics["precision"], valid_metrics["recall"], valid_metrics["f1"], test_metrics["auc"], test_metrics["aupr"], test_metrics["precision"], test_metrics["recall"], test_metrics["f1"],
                            best_valid_aupr, best_valid_aupr_epoch))

            stopper += 1
            if stopper > self.patience:
                write_log(self.logging, "Early Stopping.")
                break

        # ! Finished training, load checkpoints
        if self.load_best_model:
            self.model.load_state_dict(torch.load(os.path.join(self.args.res_dir, f'link_{em_iter}_best_model.ckpt'))['state_dict'])
        return self.model, {"best_valid_aupr": best_valid_aupr,
                            "best_valid_aupr_epoch": best_valid_aupr_epoch,
                            "graph": self.graph}

    def gnn_inference(self, gnn_model, molecule_dataset, protein_dataset):
        pseudo_labels = []
        batch_size = self.run_config.batch_size
        num_batches = int(len(self.test_edges) / batch_size) + 1
        molecule_num = len(molecule_dataset)
        for batch_index in range(num_batches):
            edges = self.test_edges[batch_index*batch_size : (batch_index+1)*batch_size]
            batch = {
                "edge": edges.to(torch.long),
                **molecule_collate_fn([molecule_dataset[edge[0]] for edge in edges]),
                **protein_collate_fn([protein_dataset[edge[1] - molecule_num] for edge in edges]),
            }
            for k, v in batch.items():
                batch[k] = v.to(self.device, non_blocking=True)

            pseudo_preds = gnn_model(batch)
            pseudo_labels.append((pseudo_preds.sigmoid() > 0.7).type(torch.FloatTensor))

        pseudo_labels = torch.cat(pseudo_labels, dim=0).to(device=self.device)
        # pseudo_labels: (test_edges_num, num_class=7)
        pseudo_edges_idx = torch.nonzero(pseudo_labels.count_nonzero(dim=1))
        pseudo_edges = self.test_edges[pseudo_edges_idx].squeeze(1).t()

        pseudo_edges = np.repeat(pseudo_edges, 2, axis = 1)
        pseudo_edges[0, 1::2] = pseudo_edges[1,0::2]
        pseudo_edges[1, 1::2] = pseudo_edges[0,0::2]
        write_log(self.logging, f"{pseudo_edges.shape[1]} pseudo edges have been added.")
        return pseudo_edges 



if __name__ == '__main__':
    args = parse_args()
    config = Config(args)

    trainer = EMTrainer(args, config)
    trainer.multi_scale_em_train()
            