#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
Copyright (c) 2023, Sun Yat-sen Univeristy.
All rights reserved.

@author:   Jiahua Rao
@license:  BSD-3-Clause, For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
@contact:  jiahua.rao@gmail.com
'''


import os
import math
import json
import numpy as np
from time import time
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling, add_self_loops

from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from config import parse_args, Config
from ppbs_dataset import config_model, list_datasets, ProteinAtomDataset, ProteinResidueDataset, PPIDataset

from models import Model, ProtGeoGNN, NetGNN, PPIsPredictor
from utils import GPUManager, write_log, Metrictor_PPBS, Metrictor_PPI


import warnings
warnings.filterwarnings('ignore')

torch.backends.cudnn.enabled = False




class EMTrainer:
    """
    Expectation-maximization Trainer.
    """

    def __init__(self, args, cfg):
        self.args = args
        self.config = cfg
        self.device1, self.device2 = GPUManager().auto_choice(num=2)
        self.device1, self.device2, self.device3 = torch.device(self.device1), torch.device(self.device2), torch.device(self.device1)

        self.logging = open(self.args.logging_file, 'a', buffering=1)
        self.em_range = range(self.config.run_config.em_total_iters)

        with open(self.args.logging_file, "a") as f:
            f.write(json.dumps(vars(self.args)) + "\n")

        self.atom_scale_model = self.build_atom_scale_model().to(self.device1)
        self.residue_scale_model = self.build_residue_scale_model().to(self.device2)
        self.ppi_scale_model = self.build_ppi_scale_model().to(self.device1)

    def build_atom_scale_model(self):
        # create model
        model = Model(config_model)
        atom_scale_pretrained_path = self.config.model_config.atom_scale_model.pretrained
        model.load_state_dict(torch.load(atom_scale_pretrained_path, map_location="cpu"))
        return model

    def build_residue_scale_model(self):
        residue_scale_model_cfg = self.config.model_config.residue_scale_model
        model = ProtGeoGNN(**residue_scale_model_cfg)
        return model


    def build_ppi_scale_model(self):
        residue_scale_model_cfg = self.config.model_config.residue_scale_model
        model = NetGNN(in_dim=residue_scale_model_cfg.hidden_dim, class_num=1)
        return model


    def multi_scale_em_train(self):
        best_metric, self.best_em_iter = 0, -1
        atom_scale_embeddings = None
        # atom_scale_embeddings = self.atom_scale_embeddings() 
        for self.em_iter in self.em_range:
            self.residue_scale_model, residue_scale_outputs = self._maximization(ppi_scale_model=self.ppi_scale_model,
                                                                                 atom_scale_embeddings=atom_scale_embeddings)
            
            residue_scale_embeddings = self.residue_scale_trainer.get_residue_embedding(self.residue_scale_model)

            self.ppi_scale_model, ppi_scale_outputs = self._expectation(residue_scale_model=self.residue_scale_model, residue_scale_embeddings=residue_scale_embeddings)

            if float(residue_scale_outputs['test_aupr']) > best_metric:
                best_metric = float(residue_scale_outputs['test_aupr'])
                self.best_em_iter = self.em_iter

            torch.save({'em_iter': self.em_iter,
                        'test_auc': float(residue_scale_outputs['test_auc']),
                        'test_aupr': float(residue_scale_outputs['test_aupr']),
                        'embeddings': residue_scale_embeddings,
                        'ppi_scale_model': self.ppi_scale_model.state_dict(),
                        'residue_scale_state_dict': self.residue_scale_model.state_dict()},
                        os.path.join(self.args.res_dir, f'em_iter_{self.em_iter}.pt'))

            if self.em_iter == (self.em_range.stop - 1):
                self.final_prediction()


    def atom_scale_embeddings(self):   # last-iter embeddings for input
        # write_log(self.logging, f'\n <<<<<<<<<< Atom Scale Embeddings >>>>>>>>>>')
        self.atom_scale_trainer = AtomModelEmbedding(args=self.args,
                                                   config=self.config,
                                                   model=self.atom_scale_model,
                                                   device=self.device1,
                                                   device2=self.device3,
                                                   logging=self.logging)
        embeddings = self.atom_scale_trainer.get_residue_embeddings(self.atom_scale_model, device=self.device1)
        return embeddings
    
    def _maximization(self, ppi_scale_model=None, atom_scale_embeddings=None, residue_scale_embeddings=None):
        # Residue GNN training
        write_log(self.logging, f'\n <<<<<<<<<< Residue GNN training >>>>>>>>>>')
        self.residue_scale_model.reset_parameters()
        self.residue_scale_trainer = ResidueModelTrainer(args=self.args,
                                                         config=self.config,
                                                         model=self.residue_scale_model,
                                                         device=self.device2,
                                                         embeddings=atom_scale_embeddings,
                                                         logging=self.logging)

        self.residue_scale_model, residue_scale_outputs = self.residue_scale_trainer.train(ppi_scale_model=ppi_scale_model,
                                                                                           # residue_scale_embeddings=residue_scale_embeddings,
                                                                                           em_iter=self.em_iter)

        msg = {
                'em_iter': self.em_iter,
                **{f'residue_scale_{k}': v for k, v in residue_scale_outputs.items() if k != 'graph'},
            }
        with open(self.args.em_log_file, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return self.residue_scale_model, residue_scale_outputs

    def _expectation(self, residue_scale_model, protein_dataset=None, residue_scale_embeddings=None):
        write_log(self.logging, f'\n <<<<<<<<<< PPI GNN training >>>>>>>>>>')
        self.ppi_scale_model.reset_parameters()

        self.ppi_scale_trainer = PPILinkTrainer(args=self.args,
                                                config=self.config,
                                                model=self.ppi_scale_model,
                                                device=self.device1,
                                                logging=self.logging)
        
        self.ppi_scale_model, ppi_scale_outputs = self.ppi_scale_trainer.train(residue_scale_model=residue_scale_model,
                                                                               protein_dataset=protein_dataset,
                                                                               residue_scale_embeddings=residue_scale_embeddings,
                                                                               em_iter=self.em_iter)

        msg = {
                'em_iter': self.em_iter,
                **{f'ppi_scale_{k}': v for k, v in ppi_scale_outputs.items() if k != 'graph'},
            }
        with open(self.args.em_log_file, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return self.ppi_scale_model, ppi_scale_outputs

    def final_prediction():
        return



class AtomModelEmbedding:
    def __init__(self, args, config, model, device, device2=None, logging=None) -> None:
        self.args = args
        self.config = config
        self.run_config = self.config.run_config
        self.patience = 10
        self.logging = logging
        self.device = torch.device(device)
        self.device2 = torch.device(device2) if device2 is not None else torch.device("cpu")

        self.model = model.to(device)
        # setup dataloaders
        self.dataloader_train, self.valid_dataloader_list, self.test_dataloader_list = self.create_dataloaders()

    def create_dataloaders(self):
        train_dataset = ProteinAtomDataset(self.args, self.config, split='train')
        sampler = RandomSampler(train_dataset, num_samples=2000)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.run_config['batch_size'],
                                  sampler=sampler,
                                  shuffle=False, num_workers=8, 
                                  collate_fn=train_dataset.collate_fn, 
                                  pin_memory=True, prefetch_factor=2)
        
        valid_loaders = []
        test_loaders = []
        for dataset in list_datasets:
            if 'validation_70' in dataset:
                validation_dataset = ProteinAtomDataset(self.args, self.config, split=dataset)
                validation_dataloader = DataLoader(validation_dataset,
                                                   batch_size=self.run_config['batch_size'], 
                                                   shuffle=False, num_workers=8, 
                                                   collate_fn=validation_dataset.collate_fn, 
                                                   pin_memory=True, prefetch_factor=2)
                valid_loaders.append(validation_dataloader)

            if 'test_70' in dataset:
                test_dataset = ProteinAtomDataset(self.args, self.config, split=dataset)
                test_dataloader = DataLoader(test_dataset,
                                             batch_size=self.run_config['batch_size'], 
                                             shuffle=False, num_workers=8, 
                                             collate_fn=test_dataset.collate_fn, 
                                             pin_memory=True, prefetch_factor=2)
                test_loaders.append(test_dataloader)
        return train_loader, valid_loaders, test_loaders

    @torch.no_grad()
    def get_residue_embeddings(self, model, device=torch.device("cpu")):
        train_dataset = ProteinAtomDataset(self.args, self.config, split='train')
        train_loader = DataLoader(train_dataset, 
                                  batch_size=self.run_config['batch_size'],
                                  shuffle=False, num_workers=8, 
                                  collate_fn=train_dataset.collate_fn, 
                                  pin_memory=True, prefetch_factor=2)

        # evaluation mode
        model = model.to(device)
        model = model.eval()
        dataloader_all = [train_loader] + self.valid_dataloader_list + self.test_dataloader_list

        all_embeddings = {}
        all_datasets = ['train'] + [d for d in list_datasets if 'validation_70' in d] + [d for d in list_datasets if 'test_70' in d]
        # print(all_datasets)
        for dataset, dataloader in zip(all_datasets, dataloader_all):
            embeddings = []
            # evaluate model
            for batch_data in tqdm(dataloader):
                # forward propagation
                # unpack data
                _, X, ids_topk, q, M, _ = [data.to(device) if type(data) == torch.Tensor else data for data in batch_data]
                # run model
                z = model.residue_embeddings(X, ids_topk, q, M)
                embeddings.append(z.cpu())
            all_embeddings[dataset] = embeddings
        torch.cuda.empty_cache()
        return all_embeddings



class ResidueModelTrainer:
    def __init__(self, args, config, model, device, embeddings=None, logging=None) -> None:
        self.args = args
        self.config = config
        self.logging = logging
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.run_config = self.config.run_config
        self.patience = self.run_config.patience

        # setup dataloader
        train_dataloader, valid_dataloader_list, test_dataloader_list = self.create_residue_dataloaders(embeddings)

        self.train_dataloader = train_dataloader
        self.valid_dataloader_list = valid_dataloader_list
        self.test_dataloader_list = test_dataloader_list

        self.optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=self.run_config.lr, weight_decay=1e-5, eps=1e-5)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.run_config.lr, steps_per_epoch=len(train_dataloader), epochs=self.run_config.epochs)
        self.loss_tr = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.predictor = PPIsPredictor(hidden_dim=self.config.model_config.residue_scale_model.hidden_dim)


    def create_residue_dataloaders(self, embeddings):
        train_dataset = ProteinResidueDataset(self.args, self.config, split='train', embeddings=embeddings)
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size = self.run_config['residue_batch_size'],
                                      sampler=None, shuffle=False, # drop_last=True, 
                                      collate_fn=train_dataset.collate_fn,
                                      num_workers=self.args.num_workers, prefetch_factor=2)

        valid_dataloaders = []
        test_dataloaders = []
        for dataset in list_datasets:
            if 'validation_70' in dataset:
                validation_dataset = ProteinResidueDataset(self.args, self.config, split=dataset, embeddings=embeddings)
                validation_dataloader = DataLoader(validation_dataset,
                                                   batch_size=self.run_config['residue_batch_size'], 
                                                   shuffle=False, num_workers=8, 
                                                   collate_fn=validation_dataset.collate_fn, 
                                                   pin_memory=True, prefetch_factor=2)
                valid_dataloaders.append(validation_dataloader)
            
            if 'test_70' in dataset:
                test_dataset = ProteinResidueDataset(self.args, self.config, split=dataset, embeddings=embeddings)
                test_dataloader = DataLoader(test_dataset,
                                             batch_size=self.run_config['residue_batch_size'], 
                                             shuffle=False, num_workers=8, 
                                             collate_fn=test_dataset.collate_fn, 
                                             pin_memory=True, prefetch_factor=2)
                test_dataloaders.append(test_dataloader)
        return train_dataloader, valid_dataloaders, test_dataloaders


    def train(self, ppi_scale_model=None, graph=None, residue_embeddings=None, em_iter=None):
        self.is_augmented = self.run_config.is_augmented and ppi_scale_model is not None
        pseudo_labels = None
        if ppi_scale_model is not None:
            pseudo_labels = self.ppi_scale_model_inference(ppi_scale_model, graph, residue_embeddings)

        best_valid_metric = 0
        not_improve_epochs = 0

        valid_list_datasets = [dataset for dataset in list_datasets if 'validation_70' in dataset]
        test_list_datasets = [dataset for dataset in list_datasets if 'test_70' in dataset]
        train_dataloader = self.train_dataloader
        valid_dataloader_list = self.valid_dataloader_list
        test_dataloader_list = self.test_dataloader_list


        for epoch in range(self.run_config.epochs):
            train_loss = 0
            train_num = 0
            self.model.train()

            train_pred = []
            train_y = []
            bar = tqdm(train_dataloader)
            
            unobserved_losses = self.eval_pseudo_step(self.model, self.device, test_dataloader_list, pseudo_labels, self.loss_tr)
            for batch_data in bar:
                data = batch_data[0]  # use structural graph data
                
                self.optimizer.zero_grad()
                data = data.to(self.device)

                outputs = self.model(data.x, data.node_feat, data.edge_index, data.seq, data.batch)

                y = torch.nan_to_num(data.y.unsqueeze(-1), nan=0)
                loss = self.loss_tr(outputs, y) * data.y_mask.unsqueeze(-1)
                loss = torch.nan_to_num(loss, nan=0)
                loss = loss.sum() / data.y_mask.sum()

                if self.is_augmented and self.run_config.pl_ratio > 0:
                    pl_weight = self.run_config.pl_ratio
                    loss = pl_weight * unobserved_losses + (1 - pl_weight) * loss

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                outputs = outputs.sigmoid() # [num_residue, num_task]

                batch_train_pred = torch.masked_select(outputs, data.y_mask.unsqueeze(-1).bool()) 
                batch_train_y = torch.masked_select(y, data.y_mask.unsqueeze(-1).bool())

                train_pred.append(batch_train_pred.detach().cpu().numpy())
                train_y.append(batch_train_y.detach().cpu().numpy())

                train_num += len(batch_train_y)
                train_loss += len(batch_train_y) * loss.item()

                bar.set_description('loss: %.4f' % (loss.item()))

            train_loss /= train_num
            train_pred = np.concatenate(train_pred)
            train_y = np.concatenate(train_y)
            train_metric = Metrictor_PPBS(train_pred, train_y)
            # torch.cuda.empty_cache()

            # Evaluate
            self.model.eval()
            valid_pred = [[] for _ in valid_list_datasets]
            valid_y = [[] for _ in valid_list_datasets]

            for idx, (valid_dataloader) in enumerate(valid_dataloader_list):
                # val_bc_scores = []
                for batch_data in valid_dataloader:
                    data = batch_data[0]  # use structural graph data
                    data = data.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(data.x, data.node_feat, data.edge_index, data.seq, data.batch).sigmoid()

                    # val_bc_scores.append(bc_scoring(data.y.unsqueeze(-1), data.y_mask.unsqueeze(-1).bool()))
                    batch_valid_y = torch.masked_select(data.y.unsqueeze(-1), data.y_mask.unsqueeze(-1).bool())
                    batch_valid_pred = torch.masked_select(outputs, data.y_mask.unsqueeze(-1).bool())
                    valid_y[idx] += list(batch_valid_y.detach().cpu().numpy())
                    valid_pred[idx] += list(batch_valid_pred.detach().cpu().numpy())

            # m_scores = nanmean(torch.stack(val_bc_scores, dim=0)).numpy()
            valid_metrics = []
            for i in range(len(valid_list_datasets)):
                valid_metrics.append(Metrictor_PPBS(valid_pred[i], valid_y[i]))

            valid_metrics = np.array(valid_metrics)
            valid_metric = valid_metrics.mean(0) # [average_AUC, average_AUPR]

            valid_auc = ",".join(list(valid_metrics[:,0].round(6).astype('str')))
            valid_aupr = ",".join(list(valid_metrics[:,1].round(6).astype('str')))

            if (valid_metric[1]) > best_valid_metric: # use AUPR
                torch.save(self.model.state_dict(), os.path.join(self.args.res_dir, 'residue_model_%s.pt'%em_iter))
                not_improve_epochs = 0
                best_valid_metric = valid_metric[1]
                
                write_log(self.logging,'[epoch %s] lr: %.6f, train_loss: %.6f, train_auc: %.6f, train_aupr: %.6f, valid_auc: %s, valid_aupr: %s'\
                %(epoch,self.scheduler.get_last_lr()[0],train_loss,train_metric[0],train_metric[1],valid_auc,valid_aupr))
            else:
                not_improve_epochs += 1
                write_log(self.logging,'[epoch %s] lr: %.6f, train_loss: %.6f, train_auc: %.6f, train_aupr: %.6f, valid_auc: %s, valid_aupr: %s, NIE +1 ---> %s'\
                %(epoch,self.scheduler.get_last_lr()[0],train_loss,train_metric[0],train_metric[1],valid_auc,valid_aupr,not_improve_epochs))

                if not_improve_epochs >= self.patience:
                    break

        state_dict = torch.load(os.path.join(self.args.res_dir, 'residue_model_%s.pt'%em_iter), self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        test_pred = [[] for _ in test_list_datasets]
        test_y = [[] for _ in test_list_datasets]

        for idx, (test_dataloader) in enumerate(test_dataloader_list):
            for batch_data in tqdm(test_dataloader):
                data = batch_data[0]  # use structural graph data
                data = data.to(self.device)

                with torch.no_grad():
                    outputs = self.model(data.x, data.node_feat, data.edge_index, data.seq, data.batch).sigmoid()

                batch_test_y = torch.masked_select(data.y.unsqueeze(-1), data.y_mask.unsqueeze(-1).bool())
                batch_test_pred = torch.masked_select(outputs, data.y_mask.unsqueeze(-1).bool())
                test_y[idx] += list(batch_test_y.detach().cpu().numpy())
                test_pred[idx] += list(batch_test_pred.detach().cpu().numpy())

        test_metrics = []
        for i in range(len(test_list_datasets)):
            test_metrics.append(Metrictor_PPBS(test_pred[i], test_y[i]))
        test_metrics = np.array(test_metrics)
        test_metric = test_metrics.mean(0) # [average_AUC, average_AUPR]

        # test_auc = ",".join(list(test_metrics[:,0].round(6).astype('str')))
        # test_aupr = ",".join(list(test_metrics[:,1].round(6).astype('str')))
        test_auc = test_metric[0]
        test_aupr = test_metric[1]

        all_test_pred, all_test_y = [], []
        for i in range(len(test_list_datasets)):
            all_test_pred += test_pred[i]
            all_test_y += test_y[i]
        all_test_metrics = Metrictor_PPBS(all_test_pred, all_test_y)

        write_log(self.logging, f'test_auc: {test_auc}, test_aupr: {test_aupr} ' + \
                  f'test_all aupr: {all_test_metrics[1]} test_all auc: {all_test_metrics[0]}' + \
                  ", ".join(["{} aupr: {:.6f}".format(key, val[1]) for key, val in zip(test_list_datasets, test_metrics)]))

        return self.model, {"test_auc": test_auc, "test_aupr": test_aupr}


    def eval_pseudo_step(self, model, device, test_dataloaders, pseudo_labels):
        model.eval()
        pre_device = model.device
        batch_size = self.run_config['residue_batch_size']

        model.to(device)
        losses = []
        test_dataset_names = [dataset for dataset in list_datasets if 'test_70' in dataset]
        for dataset, dataloader in zip(test_dataset_names, test_dataloaders):
            dataset_pseudo_labels = pseudo_labels[dataset]
            for idx, batch_data in enumerate(dataloader):
                src_data, dst_data = batch_data[1], batch_data[2] # use structural graph data
                src_data, dst_data = src_data.to(device), dst_data.to(device)
                pseudo_y = torch.cat(dataset_pseudo_labels[idx * batch_size : (idx + 1) * batch_size]).to(device)

                with torch.no_grad():
                    src_outputs = model.get_residue_representation(src_data.x, src_data.node_feat, src_data.edge_index, src_data.seq, src_data.batch)
                    dst_outputs = model.get_residue_representation(dst_data.x, dst_data.node_feat, dst_data.edge_index, dst_data.seq, dst_data.batch)
                    outputs = self.predictor(src_outputs, src_data.seq, dst_outputs, dst_data.seq)

                loss = -F.logsigmoid(outputs, pseudo_y).mean()
                losses.append(loss)
        model.train()
        model.to(pre_device)
        return torch.FloatTensor(losses).mean().to(pre_device) * 0.01


    def ppi_scale_model_inference(self, ppi_scale_model, graph, residue_embeddings):
        graph.x = torch.FloatTensor(residue_embeddings.cpu())
        graph = graph.to(self.device)
        adj = graph.adj_t

        dataset_pseudo_preds = {}
        device = ppi_scale_model.device

        test_edges_list, test_dataloader_list = [], []
        for dataset in list_datasets:
            if 'test_70' in dataset:
                test_dataset = PPIDataset(self.args, self.config, split=dataset, embeddings=residue_embeddings)
                test_edges = test_dataset[0]['edge']
                test_dataloader = DataLoader(range(test_edges.size(0)),
                                             batch_size=self.run_config['ppi_batch_size'], 
                                             shuffle=False, num_workers=8,  
                                             pin_memory=True, prefetch_factor=2)
                test_edges_list.append(test_edges)
                test_dataloader_list.append(test_dataloader)

        test_dataset_names = [dataset for dataset in list_datasets if 'test_70' in dataset]
        for dataset, test_edges, data_loader in zip(test_dataset_names, test_edges_list, test_dataloader_list):
            test_preds = []
            device = ppi_scale_model.device
            for perm_idx in tqdm(data_loader):
                z = ppi_scale_model(graph, adj, test_edges[perm_idx].t().to(device))
                test_preds.append((z.sigmoid() > 0.7).type(torch.FloatTensor).cpu())
            dataset_pseudo_preds[dataset] = test_preds
            torch.cuda.empty_cache()
        return dataset_pseudo_preds

    @torch.no_grad()
    def get_residue_embedding(self, model):
        residue_dataset = self.train_dataloader.dataset.ppi_proteins
        loader = DataLoader(residue_dataset,
                            batch_size=1024,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=Batch.from_data_list,
                            num_workers=self.args.num_workers)

        residue_embeddings = []
        with torch.no_grad():
            for residue_data in loader:
                residue_data = residue_data.to(self.device)
                residue_embeddings.append(model.get_residue_representation(residue_data.x, residue_data.node_feat, residue_data.edge_index, residue_data.seq, residue_data.batch))
        residue_embeddings = torch.cat(residue_embeddings, dim=0)
        return residue_embeddings


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
        for step, perm_idx in enumerate(DataLoader(range(train_edges.size(1)), self.run_config.batch_size, shuffle=True)):
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
            output, neg_output = self.model(data, adj, pos_edge, neg_edge)

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
        for step, perm_idx in enumerate(DataLoader(range(test_edges.size(0)), batch_size=self.run_config.batch_size, shuffle=False)):

            output = self.model(data, adj, test_edges[perm_idx].t().to(self.device))
            labels = test_label[perm_idx].type(torch.FloatTensor).to(self.device)

            loss = self.loss_fn(output, labels)
            valid_loss_sum += loss.item()

            m = torch.nn.Sigmoid()
            pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(self.device)

            valid_pre_result_list.append(pre_result.cpu().data)
            valid_label_list.append(labels.cpu().data)
            true_prob_list.append(m(output).cpu().data)
            valid_steps += 1

        valid_loss = valid_loss_sum / valid_steps
        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)
        true_prob_list = torch.cat(true_prob_list, dim = 0)

        metrics = self.evaluator(valid_pre_result_list, valid_label_list, true_prob_list, logging=self.logging)
        metrics.show_result()

        if save_probs:
            torch.save(
                {"predictions": valid_pre_result_list, "probabilities": true_prob_list, "labels": valid_label_list},
                os.path.join(self.args.res_dir, f'prediction_link_{em_iter}_results.ckpt')
            )
        return valid_loss, metrics

    def train(self, residue_model=None, protein_dataset=None, residue_embeddings=None, em_iter=0):
        self.is_augmented = self.run_config.is_augmented and residue_model is not None and protein_dataset is not None
        if residue_embeddings is not None:
            self.graph.x = torch.FloatTensor(residue_embeddings.cpu())

        write_log(self.logging, f"Link Graph has {self.graph.edge_index.shape[1]} edges.")
        # if residue_model is not None:
        #     pseudo_edges = self.residue_inference(residue_model, protein_dataset)
        #     self.graph.edge_index = torch.cat([self.graph.edge_index, pseudo_edges.to(self.device)], dim=1)

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


    def residue_inference(self, residue_model, protein_dataset):
        pseudo_labels = []
        batch_size = self.run_config.batch_size
        num_batches = int(len(self.test_edges) / batch_size) + 1
        for batch_index in range(num_batches):
            edges = self.test_edges[batch_index*batch_size : (batch_index+1)*batch_size]
            batch = {
                "edge": edges.to(torch.long),
                "protein1_data": Batch.from_data_list([protein_dataset[edge[0]] for edge in edges]),
                "protein2_data": Batch.from_data_list([protein_dataset[edge[1]] for edge in edges]),
            }
            for k, v in batch.items():
                batch[k] = v.to(self.device, non_blocking=True)

            pseudo_preds = residue_model(batch)
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
