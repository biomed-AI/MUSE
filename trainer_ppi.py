#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@author:   Jiahua Rao
@license:  BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact:  jiahua.rao@gmail.com
@time:     2023-05-16
'''


import os
import copy
import json
import numpy as np
from time import time

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling, add_self_loops

from config import parse_args, Config
from utils import write_log, Metrictor_PPI, GPUManager
from dataset import ProteinDataset, LinkGraphDataset
from models import ProtGeoGNN, GCN_PPI, GNN_PPI, NetGNN

import warnings
warnings.filterwarnings('ignore')


class EMTrainer:
    """
    Expectation-maximization Trainer.
    """

    def __init__(self, args, cfg):
        self.args = args
        self.config = cfg
        self.device, self.device2 = GPUManager().auto_choice(num=2)
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
        protein_gnn_model = ProtGeoGNN(node_input_dim=intra_model_cfg.protein_input_dim,
                                       edge_input_dim=intra_model_cfg.protein_edge_input_dim,
                                       hidden_dim=intra_model_cfg.protein_gnn_dim,
                                       num_layers=intra_model_cfg.protein_gnn_layers,
                                       dropout=0.2,
                                       augment_eps=0,)
        model = GNN_PPI(args=intra_model_cfg, class_num=7, gnn_model=protein_gnn_model)
        return model

    def build_inter_model(self):
        model = NetGNN(in_dim=self.config.model_config.intra_model.protein_gnn_dim)
        return model

    def multi_scale_em_train(self):
        # self.intra_pre_train()
        best_f1, self.best_em_iter = 0, -1
        protein_embeddings, link_graph = None, None
        for self.em_iter in self.em_range:
            self.gnn_model, _ = self._maximization(link_model=self.link_model,
                                                   link_graph=link_graph,
                                                   protein_embeddings=protein_embeddings)

            protein_embeddings = self.gnn_trainer.get_graph_embedding(self.gnn_model)
            self.link_model, link_outputs = self._expectation(gnn_model=self.gnn_model, protein_embeddings=protein_embeddings)

            link_graph = link_outputs['graph']
            if link_outputs['best_valid_f1'] > best_f1:
                best_f1 = link_outputs['best_valid_f1']
                self.best_em_iter = self.em_iter
                torch.save({'em_iter': self.em_iter,
                            'best_metric': best_f1,
                            'link_graph': link_graph,
                            'protein_embeddings': protein_embeddings,
                            'gnn_state_dict': self.gnn_model.state_dict(),
                            'link_state_dict': self.link_model.state_dict()},
                            os.path.join(self.args.res_dir, f'best_em_iter_model.ckpt'))

            if self.em_iter == (self.em_range.stop - 1):
                self.final_prediction()

    def _maximization(self, link_model=None, link_graph=None, protein_embeddings=None):
        # Protein GNN training
        write_log(self.logging, f'\n <<<<<<<<<< Protein GNN training >>>>>>>>>>')
        self.gnn_model.reset_parameters()
        self.gnn_trainer = ProteinGNNTrainer(args=self.args,
                                             config=self.config,
                                             model=self.gnn_model,
                                             dataset=self.link_dataset,
                                             device=self.device,
                                             logging=self.logging)
        
        if protein_embeddings is None:
            protein_embeddings = self.gnn_trainer.get_graph_embedding(self.gnn_model)

        self.gnn_model, gnn_outputs = self.gnn_trainer.train(link_model=link_model,
                                                             link_graph=link_graph,
                                                             protein_embeddings=protein_embeddings,
                                                             em_iter=self.em_iter)

        msg = {
                'em_iter': self.em_iter,
                **{f'gnn_{k}': v for k, v in gnn_outputs.items()},
            }
        with open(self.args.em_log_file, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return self.gnn_model, gnn_outputs

    def _expectation(self, gnn_model=None, protein_embeddings=None):
        # Link GNN training
        write_log(self.logging, f'\n <<<<<<<<<< Link GNN training >>>>>>>>>>')
        self.link_model.reset_parameters()
        protein_dataset = self.gnn_trainer.train_loader.dataset.protein_dataset
        self.link_trainer = PPILinkTrainer(args=self.args,
                                           config=self.config,
                                           model=self.link_model,
                                           dataset=self.link_dataset,
                                           device=self.device2,
                                           logging=self.logging)
        self.link_model, link_outputs = self.link_trainer.train(gnn_model=gnn_model,
                                                                protein_dataset=protein_dataset,
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
        final_link_model.load_state_dict(torch.load(os.path.join(self.args.res_dir, f'best_em_iter_model.ckpt'))['link_state_dict'])
        pseudo_graph = torch.load(os.path.join(self.args.res_dir, f'best_em_iter_model.ckpt'))['link_graph']

        link_trainer = PPILinkTrainer(args=self.args,
                                      config=self.config,
                                      model=final_link_model,
                                      dataset=self.link_dataset,
                                      device=self.device,
                                      logging=self.logging)
    
        if pseudo_graph is not None:
            link_trainer.graph = pseudo_graph

        _, test_metrics = link_trainer.evaluate(graph=pseudo_graph, save_probs=True, em_iter=best_em_iter)

        write_log(self.logging,
            "em_iter: {} Test Accuracy: {}, Recall: {}, Precision: {}, F1: {}, AUC: {}, AUPR: {}"
                .format(best_em_iter, test_metrics.Accuracy, test_metrics.Recall, test_metrics.Precision, test_metrics.F1, test_metrics.Auc, test_metrics.Aupr))


class ProteinGNNTrainer:
    def __init__(self, args, config, model, dataset, device, logging=None) -> None:
        self.args = args
        self.config = config
        self.link_dataset = dataset
        self.run_config = self.config.run_config
        self.patience = 10
        self.logging = logging
        self.device = torch.device(device)

        self.train_loader, self.test_loader = self.create_dataloaders()
        self.graph = self.link_dataset[0]

        self.test_edges = self.link_dataset.get_edge_split()['test']['edge']

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.run_config.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10,
                                                                    verbose=True)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.evaluator = Metrictor_PPI
        self.load_best_model = True

    def create_dataloaders(self):
        train_dataset = ProteinDataset(self.args, self.config, split='train')
        test_dataset = ProteinDataset(self.args, self.config, split='test')

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.run_config.batch_size,
                                  shuffle=True,
                                  sampler=None,
                                  collate_fn=train_dataset.collate_fn,
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
        return train_loader, test_loader

    # def _train_epoch(self, loader, link_model=None, protein_embeddings=None):
    def _train_epoch(self, loader, link_model=None, protein_embeddings=None, pseudo_labels=None):
        self.model.train()

        steps = 0
        loss_sum = 0.0

        num_batches = len(loader)
        pseudo_batch_size = int(len(self.test_edges) // num_batches)
        m = torch.nn.Sigmoid()
        label_list, true_prob_list = [], []
        for step, batch in enumerate(loader):
            for k, v in batch.items():
                batch[k] = v.to(self.device, non_blocking=True)

            output = self.model(batch)
            labels = batch["label"].type(torch.FloatTensor).to(self.device)

            loss = self.loss_fn(output, labels)

            if self.is_augmented and self.run_config.pl_ratio > 0:
                pl_weight = self.run_config.pl_ratio
                new_batch = self.get_pseudo_batch(step, pseudo_batch_size)
                unobserved_output = self.model(new_batch)
                unobserved_labels = pseudo_labels[step*pseudo_batch_size : (step+1)*pseudo_batch_size]
                _, observed_labels = self.link_inference(link_model, batch['edge'], self.graph, protein_embeddings)

                pl_loss = self.loss_fn(output, observed_labels)  + self.loss_fn(unobserved_output, unobserved_labels) 
                loss = pl_weight * pl_loss + (1 - pl_weight) * loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            label_list.append(labels.cpu().data)
            true_prob_list.append(m(output).cpu().data)
            loss_sum += loss.item()
            steps += 1

        label_list = torch.cat(label_list, dim=0).cpu()
        true_prob_list = torch.cat(true_prob_list, dim = 0).cpu()

        pre_result_list = (true_prob_list > 0.5).type(torch.FloatTensor)
        metrics = Metrictor_PPI(pre_result_list, label_list, true_prob_list, logging=self.logging)
        metrics.show_result()
        torch.cuda.empty_cache()
        return loss_sum / steps, (metrics.Recall, metrics.Precision, metrics.F1)

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.model.eval()

        valid_pre_result_list = []
        valid_label_list = []
        true_prob_list = []
        valid_loss_sum = 0.0
        valid_steps = 0

        for step, batch in enumerate(test_loader):
            for k, v in batch.items():
                batch[k] = v.to(self.device, non_blocking=True)

            output = self.model(batch)
            labels = batch["label"].type(torch.FloatTensor).to(self.device)

            loss = self.loss_fn(output, labels)
            valid_loss_sum += loss.item()

            m = torch.nn.Sigmoid()
            # pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(self.device)

            # valid_pre_result_list.append(pre_result.cpu().data)
            valid_label_list.append(labels.cpu().data)
            true_prob_list.append(m(output).cpu().data)
            valid_steps += 1

        valid_loss = valid_loss_sum / valid_steps
        # valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0).cpu()
        true_prob_list = torch.cat(true_prob_list, dim = 0).cpu()

        valid_pre_result_list = (true_prob_list > 0.5).type(torch.FloatTensor)
        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list, true_prob_list, logging=self.logging)
        metrics.show_result()
        return valid_loss, metrics

    def train(self, link_model=None, link_graph=None, protein_embeddings=None, em_iter=0):
        self.is_augmented = self.run_config.is_augmented and link_model is not None and protein_embeddings is not None
        pseudo_labels = None
        if link_model is not None and protein_embeddings is not None:
            self.graph = link_graph if link_graph is not None else self.graph
            link_model = link_model.to(self.device)
            self.graph.adj_t = SparseTensor.from_edge_index(self.graph.edge_index, sparse_sizes=(self.graph.num_nodes, self.graph.num_nodes)).to(self.device)
            self.graph.adj_t = self.graph.adj_t.to_symmetric().coalesce()
            _, pseudo_labels = self.link_inference(link_model, self.test_edges, self.graph, protein_embeddings)


        stopper = 0
        best_valid_f1 = 0.0
        best_valid_f1_epoch = 0

        # ! GNN Training
        for self.cur_epoch in range(self.run_config.gnn_epochs):
            start_time = time()
            loss, (train_recall, train_precision, train_f1) = self._train_epoch(self.train_loader, link_model=link_model, protein_embeddings=protein_embeddings, pseudo_labels=pseudo_labels)
            test_loss, test_metrics = self.evaluate(self.test_loader)

            if self.scheduler != None:
                self.scheduler.step(loss)
                write_log(self.logging, "epoch: {}, now learning rate: {}".format(self.cur_epoch, self.scheduler.optimizer.param_groups[0]['lr']))

            if best_valid_f1 < test_metrics.F1:
                best_valid_f1 = test_metrics.F1
                best_valid_f1_epoch = self.cur_epoch

                torch.save({'epoch': self.cur_epoch,
                            'state_dict': self.model.state_dict()},
                            os.path.join(self.args.res_dir, f'gnn_{em_iter}_best_model.ckpt'))
                stopper = 0

            write_log(self.logging,
                "epoch: {}, time {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, AUC: {}, AUPR: {}, F1: {}, Best valid_F1: {}, in {} epoch"
                    .format(self.cur_epoch, time() - start_time, loss, train_recall, train_precision, train_f1, test_loss, test_metrics.Recall, test_metrics.Precision, test_metrics.Auc, test_metrics.Aupr, test_metrics.F1,
                            best_valid_f1, best_valid_f1_epoch))

            stopper += 1
            if stopper > self.patience:
                write_log(self.logging, "Early Stopping.")
                break

            torch.cuda.empty_cache()

        # ! Finished training, load checkpoints
        if self.load_best_model:
            self.model.load_state_dict(torch.load(os.path.join(self.args.res_dir, f'gnn_{em_iter}_best_model.ckpt'))['state_dict'])
        return self.model, {"best_valid_f1": best_valid_f1, "best_valid_f1_epoch": best_valid_f1_epoch}

    def get_graph_embedding(self, model):
        protein_dataset = self.train_loader.dataset.protein_dataset
        loader = DataLoader(protein_dataset,
                            batch_size=16,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=Batch.from_data_list,
                            num_workers=self.args.num_workers)

        protein_embeddings = []
        with torch.no_grad():
            for protein_data in loader:
                protein_data = protein_data.to(self.device)
                protein_embeddings.append(model.get_graph_representation(protein_data))
        protein_embeddings = torch.cat(protein_embeddings, dim=0)
        return protein_embeddings

    def get_pseudo_batch(self, batch_index, batch_size):
        edges = self.test_edges[batch_index*batch_size : (batch_index+1)*batch_size]
        batch = {
            "edge": edges.to(torch.long),
            "protein1_data": Batch.from_data_list([self.train_loader.dataset.protein_dataset[edge[0]] for edge in edges]),
            "protein2_data": Batch.from_data_list([self.train_loader.dataset.protein_dataset[edge[1]] for edge in edges]),
        }

        for k, v in batch.items():
            batch[k] = v.to(self.device, non_blocking=True)
        return batch

    def link_inference(self, link_model, test_edges, graph, protein_embeddings):
        graph.x = torch.FloatTensor(protein_embeddings.cpu())
        graph = graph.to(self.device)
        adj = graph.adj_t

        pseudo_preds = []
        for step, perm_idx in enumerate(DataLoader(range(test_edges.size(0)), batch_size=self.run_config.link_batch_size, shuffle=False)):
            output = link_model(graph.x, graph.edge_index, adj, test_edges[perm_idx].t().to(self.device))
            pseudo_preds.append(output.cpu().data)
        pseudo_preds = torch.cat(pseudo_preds, dim = 0)
        pseudo_labels = (pseudo_preds.sigmoid() > 0.5).type(torch.FloatTensor).to(device=self.device)
        torch.cuda.empty_cache()
        return pseudo_preds, pseudo_labels


class PPILinkTrainer:
    def __init__(self, args, config, model, dataset, device, logging) -> None:
        self.args = args
        self.config = config
        self.logging = logging
        self.dataset = dataset
        self.device = torch.device(device)

        self.train_data, self.test_data = self.create_dataloaders()
        self.train_edges, self.test_edges = self.train_data['edge'], self.test_data['edge']
        self.train_label, self.test_label = self.train_data['label'], self.test_data['label']

        self.graph = self.dataset[0].to(device)

        self.model = model.to(self.device)
        self.run_config = self.config.run_config
        self.patience = self.run_config.patience
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.run_config.link_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10,
                                                                    verbose=True)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.evaluator = Metrictor_PPI
        self.load_best_model = True

    def create_dataloaders(self):
        split_edges = self.dataset.get_edge_split()
        return split_edges['train'], split_edges['test']

    def _train_epoch(self, graph, train_edges, train_label):
        self.model.train()

        steps = 0
        recall_sum = 0.0
        precision_sum = 0.0
        f1_sum = 0.0
        auc_sum = 0.0
        loss_sum = 0.0

        data = graph.to(self.device)
        train_edges = train_edges.to(self.device).t()
        adjmask = torch.ones_like(train_edges[0], dtype=torch.bool)
        new_edge_index, _ = add_self_loops(data.edge_index)
        negedge = negative_sampling(new_edge_index.to(train_edges.device), data.adj_t.sizes()[0])
        for step, perm_idx in enumerate(DataLoader(range(train_edges.size(1)), self.run_config.link_batch_size, shuffle=True)):
            # Target Link Masking
            adjmask[perm_idx] = 0
            tei = train_edges[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                               train_edges.device, non_blocking=True)
            adjmask[perm_idx] = 1
            adj = adj.to_symmetric()
            labels = train_label[perm_idx].type(torch.FloatTensor).to(self.device)

            # output = self.model(self.graph, train_edges[perm_idx].t().to(self.device))
            # loss = self.loss_fn(output, labels)
            pos_edge, neg_edge = train_edges[:, perm_idx], negedge[:, perm_idx]
            output, neg_output = self.model(data.x, data.edge_index, adj, pos_edge, neg_edge)

            pos_loss = self.loss_fn(output, labels) 
            neg_loss = self.loss_fn(neg_output, torch.zeros(neg_output.shape).to(self.device))
            loss = pos_loss + 0.5 * neg_loss
            # loss = torch.mean([self.loss_fn(output[:, :, i], labels) for i in range(output.shape[-1])])
            # loss = loss + torch.mean([self.loss_fn(neg_output[:, :, i], torch.zeros(neg_output.shape).to(self.device)) for i in range(neg_output.shape[-1])])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            m = torch.nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(self.device)
            metrics = self.evaluator(pre_result.cpu().data, labels.cpu().data, m(output).cpu().data, logging=self.logging)
            metrics.show_result()

            recall_sum += metrics.Recall
            precision_sum += metrics.Precision
            f1_sum += metrics.F1
            auc_sum += metrics.Auc
            loss_sum += loss.item()

            if step == 0:
                write_log(self.logging, f"epoch: {self.cur_epoch}, step: {step}, Train: label_loss: {loss.item()}, pos_loss: {pos_loss}, neg_loss: {neg_loss} \
                                          precision: {metrics.Precision}, recall: {metrics.Recall}, f1: {metrics.F1}")
            steps += 1
        return loss_sum / steps, (recall_sum / steps, precision_sum / steps, f1_sum / steps)

    @torch.no_grad()
    def evaluate(self, graph=None, test_edges=None, test_label=None, save_probs=False, em_iter=0):
        self.model.eval()

        valid_pre_result_list = []
        valid_label_list = []
        true_prob_list = []
        valid_loss_sum = 0.0
        valid_steps = 0

        if test_edges is None:
            test_edges = self.test_edges
            test_label = self.test_label

        data = graph.to(self.device)
        adj = data.adj_t
        for step, perm_idx in enumerate(DataLoader(range(test_edges.size(0)), batch_size=self.run_config.link_batch_size, shuffle=False)):

            output = self.model(data.x, data.edge_index, adj, test_edges[perm_idx].t().to(self.device))
            labels = test_label[perm_idx].type(torch.FloatTensor).to(self.device)

            loss = self.loss_fn(output, labels)
            valid_loss_sum += loss.item()

            m = torch.nn.Sigmoid()
            # pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(self.device)
            # valid_pre_result_list.append(pre_result.cpu().data)
            valid_label_list.append(labels.cpu().data)
            true_prob_list.append(m(output).cpu().data)
            valid_steps += 1

        valid_loss = valid_loss_sum / valid_steps
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim = 0)
        valid_pre_result_list = (true_prob_list > 0.5).type(torch.FloatTensor)

        metrics = self.evaluator(valid_pre_result_list, valid_label_list, true_prob_list, logging=self.logging)
        metrics.show_result()

        if save_probs:
            torch.save(
                {"predictions": valid_pre_result_list, "probabilities": true_prob_list, "labels": valid_label_list},
                os.path.join(self.args.res_dir, f'prediction_link_{em_iter}_results.ckpt')
            )
        return valid_loss, metrics

    def train(self, gnn_model=None, protein_dataset=None, protein_embeddings=None, em_iter=0):
        self.is_augmented = self.run_config.is_augmented and gnn_model is not None and protein_dataset is not None
        if protein_embeddings is not None:
            self.graph.x = torch.FloatTensor(protein_embeddings.cpu())

        if gnn_model is not None:
            # gnn_model.to(self.device)
            pseudo_edges = self.gnn_inference(gnn_model, protein_dataset)
            self.graph.edge_index = torch.cat([self.graph.edge_index, pseudo_edges.to(self.device)], dim=1)

        self.graph.adj_t = SparseTensor.from_edge_index(self.graph.edge_index, sparse_sizes=(self.graph.num_nodes, self.graph.num_nodes)).to(self.device)
        self.graph.adj_t = self.graph.adj_t.to_symmetric().coalesce()

        stopper = 0
        best_valid_f1 = 0.0
        best_valid_f1_epoch = 0

        # ! Link Training
        for self.cur_epoch in range(self.run_config.link_epochs):
            start_time = time()
            loss, (train_recall, train_precision, train_f1) = self._train_epoch(self.graph, self.train_edges, self.train_label)
            valid_loss, test_metrics = self.evaluate(self.graph, self.test_edges, self.test_label)

            if self.scheduler != None:
                self.scheduler.step(loss)
                write_log(self.logging, "epoch: {}, now learning rate: {}".format(self.cur_epoch, self.scheduler.optimizer.param_groups[0]['lr']))

            if best_valid_f1 < test_metrics.F1:
                best_valid_f1 = test_metrics.F1
                best_valid_f1_epoch = self.cur_epoch

                torch.save({'epoch': self.cur_epoch,
                            'state_dict': self.model.state_dict(),
                            'graph': self.graph},
                            os.path.join(self.args.res_dir, f'link_{em_iter}_best_model.ckpt'))
                stopper = 0

            write_log(self.logging,
                "epoch: {}, time {}, Training_avg: label_loss: {}, recall: {}, precision: {}, F1: {}, Validation_avg: loss: {}, recall: {}, precision: {}, AUC: {}, AUPR: {}, F1: {}, Best valid_F1: {}, in {} epoch"
                    .format(self.cur_epoch, time() - start_time, loss, train_recall, train_precision, train_f1, valid_loss, test_metrics.Recall, test_metrics.Precision, test_metrics.Auc, test_metrics.Aupr, test_metrics.F1,
                            best_valid_f1, best_valid_f1_epoch))

            stopper += 1
            if stopper > self.patience:
                write_log(self.logging, "Early Stopping.")
                break

        # ! Finished training, load checkpoints
        if self.load_best_model:
            self.model.load_state_dict(torch.load(os.path.join(self.args.res_dir, f'link_{em_iter}_best_model.ckpt'))['state_dict'])
        return self.model, {"best_valid_f1": best_valid_f1,
                            "best_valid_f1_epoch": best_valid_f1_epoch,
                            "graph": self.graph}

    @torch.no_grad()
    def gnn_inference(self, gnn_model, protein_dataset):
        # print(self.device)
        device = gnn_model.device
        pseudo_labels = []
        batch_size = self.run_config.batch_size
        num_batches = int(len(self.test_edges) // batch_size)
        for batch_index in range(num_batches):
            edges = self.test_edges[batch_index*batch_size : (batch_index+1)*batch_size]
            batch = {
                "edge": edges.to(torch.long),
                "protein1_data": Batch.from_data_list([protein_dataset[edge[0]] for edge in edges]),
                "protein2_data": Batch.from_data_list([protein_dataset[edge[1]] for edge in edges]),
            }
            for k, v in batch.items():
                batch[k] = v.to(device, non_blocking=True)

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
    # trainer.final_prediction()