import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn import global_mean_pool


class GCN(nn.Module):
    def __init__(self, hidden):
        super(GCN, self).__init__()
        # hidden = args.hidden
        self.conv1 = GCNConv(7, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)
  
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.sag1 = SAGPooling(hidden,0.5)
        self.sag2 = SAGPooling(hidden,0.5)
        self.sag3 = SAGPooling(hidden,0.5)
        self.sag4 = SAGPooling(hidden,0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(0.5)
        # for param in self.parameters():
        #     print(type(param), param.size())

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()

        self.sag1.reset_parameters()
        self.sag2.reset_parameters()
        self.sag3.reset_parameters()
        self.sag4.reset_parameters()

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()

    def featurize(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.bn1(x)

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x) 
        x = self.bn2(x)
        
        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x) 
        x = self.bn3(x)

        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x) 
        x = self.bn4(x)

        return x

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x) 
        x = self.bn1(x)
        y = self.sag1(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1] 

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x) 
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]  
        
        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x) 
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x) 
        x = self.bn4(x)
        y = self.sag4(x, edge_index, batch = batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        # y = self.sag4(x, edge_index, batch = batch)
        return global_mean_pool(y[0], y[3])


class GCN_PPI(nn.Module):
    def __init__(self, args, class_num=7, gnn_model=None):
        super().__init__()
        self.emb_dim = args.hidden

        self.gnn_model = gnn_model
        self.link_pred_linear = nn.Linear(2 * self.emb_dim, class_num)

    def reset_parameters(self):
        self.gnn_model.reset_parameters()
        self.link_pred_linear.reset_parameters()

    def from_pretrained(self, model_file):
        self.gnn_model.load_state_dict(torch.load(model_file))

    def get_graph_representation(self, data):
        return self.gnn_model(data.x, data.edge_index, data.batch)

    def forward(self, batch):
        protein1_data, protein2_data = batch['protein1_data'], batch['protein2_data']
        protein1_emb = self.gnn_model(protein1_data.x, protein1_data.edge_index, protein1_data.batch)
        protein2_emb = self.gnn_model(protein2_data.x, protein2_data.edge_index, protein2_data.batch)

        x = torch.cat([protein1_emb, protein2_emb], dim=1)
        x = self.link_pred_linear(x)
        return x