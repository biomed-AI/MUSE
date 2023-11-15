from models.drug.gnn import GCN as DrugGCN, GNN as DrugGNN, GNN_DDI
from models.protein.gnn import GCN as ProtGNN, GCN_PPI
from models.protein.geo_gnn import GeoGNN as ProtGeoGNN, GNN_PPI
from models.network.gnn import GIN as NetGNN


import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import unbatch

from models.network.encoder import Decoder, Encoder


class GNN_DPI(nn.Module):
    def __init__(self, args):
        super(GNN_DPI, self).__init__()

        self.protein_dim = args.protein_dim
        self.atom_dim = args.atom_dim
        self.embedding_dim = args.embedding_dim
        self.mol2vec_embedding_dim = args.mol2vec_embedding_dim
        # if args.objective == 'classification':
        decoder_hid_dim = args.hid_dim  # for GalaxyDB
        # else:
        #     decoder_hid_dim = args.hid_dim  # for Davis

        self.decoder = Decoder(
            atom_dim=decoder_hid_dim,
            hid_dim=args.hid_dim,
            n_layers=args.decoder_layers,
            n_heads=args.n_heads,
            pf_dim=args.pf_dim,
            dropout=args.dropout,
        )

        self.compound_gcn = DrugGNN(args, self.atom_dim, args.compound_gnn_dim)

        self.mol2vec_fc = nn.Linear(self.mol2vec_embedding_dim, self.mol2vec_embedding_dim)
        self.mol_concat_fc = nn.Linear(self.mol2vec_embedding_dim + args.compound_gnn_dim, decoder_hid_dim)
        self.mol_concat_ln = nn.LayerNorm(decoder_hid_dim)

        # self.tape_fc = nn.Linear(self.embedding_dim, args.hid_dim * 4)
        # self.protein_gcn = DrugGNN(args, self.protein_dim, args.protein_gnn_dim)
        # self.protein_gcn_ones = DrugGNN(args, self.protein_dim, args.protein_gnn_dim)
        self.protein_gcn = ProtGeoGNN(
            node_input_dim=args.protein_input_dim,
            edge_input_dim=args.protein_edge_input_dim,
            hidden_dim=args.protein_gnn_dim,
            num_layers=args.protein_gnn_layers,
            dropout=0.2,
            augment_eps=0,
        )

        self.encoder = Encoder(
            protein_dim=args.protein_gnn_dim,
            hid_dim=args.hid_dim,
            n_layers=args.cnn_layers,
            kernel_size=args.cnn_kernel_size,
            dropout=args.dropout,
        )

        self.concat_fc = nn.Linear(args.hid_dim * 5, args.hid_dim)
        self.concat_ln = nn.LayerNorm(args.hid_dim)

        # self.objective = args.objective
        # if self.objective == "classification":
        self.fc = nn.Linear(256, 2)
        # elif self.objective == "regression":
        #     self.fc = nn.Linear(256, 1)

    def reset_parameters(self):
        self.compound_gcn.reset_parameters()
        self.mol2vec_fc.reset_parameters()
        self.mol_concat_fc.reset_parameters()
        self.mol_concat_ln.reset_parameters()

        # self.tape_fc.reset_parameters()
        self.protein_gcn.reset_parameters()
        # self.protein_gcn_ones.reset_parameters()

        self.encoder.reset_parameters()
        self.decoder.reset_parameters()
        self.concat_fc.reset_parameters()
        self.concat_ln.reset_parameters()
        self.fc.reset_parameters()

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        batch_size = len(atom_num)
        compound_mask = torch.zeros((batch_size, compound_max_len)).type_as(atom_num)
        protein_mask = torch.zeros((batch_size, protein_max_len)).type_as(atom_num)

        for i in range(batch_size):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        return compound_mask, protein_mask

    def make_single_masks(self, num, max_len):
        batch_size = len(num)
        mask = torch.zeros((batch_size, max_len)).type_as(num)
        for i in range(batch_size):
            mask[i, :num[i]] = 1
        mask = mask.unsqueeze(1).unsqueeze(2)
        return mask
    
    def get_protein_representation(self, batch):
        # gcn_protein = self.protein_gcn(batch["protein_node_feat"], batch["protein_map"])
        # gcn_ones_protein = self.protein_gcn_ones(batch["protein_node_feat"], torch.ones_like(batch["protein_map"]))
        # gcn_protein = gcn_protein + gcn_ones_protein
        data = batch['protein_data']
        gcn_protein = self.protein_gcn(data.x, data.node_feat, data.edge_index, data.seq, data.batch)
        # enc_src = self.encoder(gcn_protein)
        return gcn_protein

        # tape_embedding = self.tape_fc(batch['protein_embedding'])
        # enc_src = torch.cat([tape_embedding, enc_src], dim=-1)
        # enc_src = self.concat_fc(enc_src)
        # enc_src = self.concat_ln(enc_src)
        # return global_mean_pool(gcn_protein, data.batch)
    
    def get_mol_representation(self, batch):
        compound_gcn = self.compound_gcn(batch["compound_node_feat"], batch["compound_adj"])
        mol2vec_embedding = self.mol2vec_fc(batch['compound_word_embedding'])
        compound = torch.cat([mol2vec_embedding, compound_gcn], dim=-1)
        compound = self.mol_concat_fc(compound)
        compound = self.mol_concat_ln(compound)
        return compound.mean(1)

    def forward(self, batch):
        batch_size = batch["compound_node_feat"].shape[0]
        compound_max_len = batch["compound_node_feat"].shape[1]
        # protein_max_len = batch["protein_node_feat"].shape[1]

        compound_gcn = self.compound_gcn(batch["compound_node_feat"], batch["compound_adj"])

        mol2vec_embedding = self.mol2vec_fc(batch['compound_word_embedding'])
        compound = torch.cat([mol2vec_embedding, compound_gcn], dim=-1)
        compound = self.mol_concat_fc(compound)
        compound = self.mol_concat_ln(compound)

        # gcn_protein = self.protein_gcn(batch["protein_node_feat"], batch["protein_map"])
        # gcn_ones_protein = self.protein_gcn_ones(batch["protein_node_feat"], torch.ones_like(batch["protein_map"]))
        # gcn_protein = gcn_protein + gcn_ones_protein
        data = batch['protein_data']
        gcn_protein = self.protein_gcn(data.x, data.node_feat, data.edge_index, data.seq, data.batch)
        # enc_src = self.encoder(gcn_protein)

        gcn_protein_list = unbatch(gcn_protein, data.batch)
        protein_max_len = max([x.shape[0] for x in gcn_protein_list])

        enc_src = torch.zeros((batch_size, protein_max_len, gcn_protein_list[0].shape[1])).to(gcn_protein.device)
        for i, v in enumerate(gcn_protein_list):
            enc_src[i, :v.shape[0], :] = v.to(torch.float)

        compound_mask, protein_mask = self.make_masks(
            batch["compound_node_num"],
            batch["protein_node_num"],
            compound_max_len,
            protein_max_len,
        )

        # tape_embedding = self.tape_fc(batch['protein_embedding'])
        # enc_src = torch.cat([tape_embedding, enc_src], dim=-1)
        # enc_src = self.concat_fc(enc_src)
        # enc_src = self.concat_ln(enc_src)
        out = self.decoder(compound, enc_src, compound_mask, protein_mask)  # for attention

        # if self.objective == "classification":
        out = self.fc(out)
        # elif self.objective == "regression":
        #     out = self.fc(out).squeeze(1)
        #     out = torch.sigmoid(out) * 14
        return out