import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.data import Data

from torch_geometric.nn import GINConv, global_mean_pool, GCNConv
from torch_geometric.nn.pool import SAGPooling



class GCN(nn.Module):

    def __init__(self):

        super(GCN, self).__init__()
        hidden = 128
        self.conv1 = GCNConv(7, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.conv4 = GCNConv(hidden, hidden)

        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.bn4 = nn.BatchNorm1d(hidden)

        self.sag1 = SAGPooling(hidden, 0.5)
        self.sag2 = SAGPooling(hidden, 0.5)
        self.sag3 = SAGPooling(hidden, 0.5)
        self.sag4 = SAGPooling(hidden, 0.5)

        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        y = self.sag1(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv2(x, edge_index)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        y = self.sag2(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv3(x, edge_index)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn3(x)
        y = self.sag3(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        x = self.conv4(x, edge_index)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.bn4(x)
        y = self.sag4(x, edge_index, batch=batch)
        x = y[0]
        batch = y[3]
        edge_index = y[1]

        return global_mean_pool(y[0], y[3])

class GIN(nn.Module):

    def __init__(self, hidden=512, train_eps=True, class_num=7):

        super(GIN, self).__init__()

        self.train_eps = train_eps

        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(128, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        
        self.lin1 = nn.Linear(hidden, hidden)

        self.fc1 = nn.Linear(hidden, 64) # original: 7
        

    def reset_parameters(self):

        self.gin_conv1.reset_parameters()
        self.gin_conv2.reset_parameters()

        self.lin1.reset_parameters()
        self.fc1.reset_parameters()
        

    def forward(self, x, edge_index, train_edge_index, p=0.0):

        x = self.gin_conv1(x, edge_index)
        x = self.gin_conv2(x, edge_index)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.0, training=self.training)
         
        x1 = x[train_edge_index[0]]
        x2 = x[train_edge_index[1]]

        x = torch.mul(x1, x2)
        x = self.fc1(x)

        return x
   
class labelG(nn.Module):

    def __init__(self):

        super(labelG, self).__init__()

        self.conv = GCNConv(200, 64) #128 random:0.85
        #self.lin = nn.Linear(256, 64) # 0.8867
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(64)

    def forward(self, x, edge_index):

        x = self.conv(x, edge_index)

        #x = self.lin(x)

        x = self.relu(x)

        x = self.bn(x)

        return x

class lambiomhG(nn.Module):

    def __init__(self, num_classes=7, train_params=None, feat_full=None,  
                 label_full=None, cpu_eval=False, labGx=None, labe=None):

        """
        Build the multi-layer GNN architecture.

        Inputs:
              num_classes:  int, number of classes a node can belong to
              train_params: dict, training hyperparameters(e.g. learning rate)
              feat_full  :    np array of shape N x f, where N  is the total num of nodes 
                              and f is the dimension for input node feature
              label_full  :?
              cpu_eval:   bool, if True, will put the model on CPU.

        Outputs:

            None
        """
        args_global_gpu = 1
        super(lambiomhG, self).__init__()
        self.use_cuda = (args_global_gpu >=0)

        if cpu_eval:

            self.use_cuda = False

        self.BGNN = GCN()
        self.TGNN = GIN()

        
        
        self.labgraph = Data(x=labGx, edge_index=labe)
        
        self.labGNN = labelG()

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4)

        
    def forward(self, batch, p_x_all, p_edge_all, 
                adj_subgraph, train_edge_index):
        

        embs = self.BGNN(p_x_all, p_edge_all, batch)

        y = self.TGNN(embs, adj_subgraph, train_edge_index, p=0.5)

        C = self.labGNN(self.labgraph.x, self.labgraph.edge_index)

        final = y @ C.T

        return final
    
    def train_step(self, batch, p_x_all, p_edge_all, 
                   adj_subgraph, train_edge_index, labels): # minibatch_one_batch 的返回值
        """
        Forward and backward propagation
        """
        preds = self(batch, p_x_all, p_edge_all, adj_subgraph, train_edge_index)

        
        loss = self.loss_fn(preds, labels)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


        return loss, preds
    
    def evaluatewithfullG(self, Batch, p_x_ALL, p_edge_ALL, 
                          full_edge, val_edge_index):

        """
        To evaluate model in validation set on full graph  
        """
        
        return self(Batch, p_x_ALL, p_edge_ALL, full_edge, val_edge_index)
