####---------------------LOG-----------------------#########
#### 10月28日前实现在shs27k上子图采样训练 ##########
###------------------------------------------------#########


import numpy as np
import torch
from graph_samplers import edge_sampling
import time
import scipy.sparse as sp
import math
import torch.nn as nn

from lambiomHG.norm_aggr import *


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINConv, global_mean_pool, GCNConv
from torch_geometric.nn.pool import SAGPooling


def _coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.LongTensor(values)
    return torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))


class Minibatch:

    """
    Provides minibatches for the trainer or evaluator. This class is responsible for calling
    the proper graph sampler and estimating normalization coefficients.
    
    """
    def __init__(self, adj_full, role, edge_index, train_params, cpu_eval=False):

        """
        Inputs:

            adj_full       scipy CSR, adj matrix for the full graph
            adj_train      scipy CSR, adj matrix for the training graph. 
                           Since we are transductive setting, may ignore.
                            adj_full = adj_train

            role           dict, key 'train_index' -> list of training edge index[0, 6660)
                                 key 'valid_index' -> list of valid edge index [0,6660)
            train_params   dict, additional parameters related to trianing. e.g., 
                            how many subgraphs we want to get to estimate the norm coefficients
            cpu_eval        bool, whether or not we want to run full-batch evaluation on the CPU
            
        
        Outputs:
            None
        """
        adj_train = adj_full # transductive setting

        args_global_gpu = 3 # 全局变量,从外界传入
        self.use_cuda = (args_global_gpu >=0)

        if cpu_eval:
            self.use_cuda = False



        self.index_train = np.array(role['train_index'])

        self.edge_train = edge_index[:, self.index_train]
        
        self.index_val = np.array(role['valid_index'])

        self.edge_val = edge_index[:, self.index_val]
    
        self.adj_full = _coo_scipy2torch(adj_full.tocoo())

        self.adj_train = adj_train

        if self.use_cuda:

            self.adj_full = self.adj_full.cuda() # 把full graph 放在GPU上？

        self.node_train =  np.array(list(range(1553))) # 所有节点

        
        # below: book-keeping for mini-batch
        self.node_subgraph = None
        self.batch_num = -1

        self.method_sample = None
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []

        self.norm_loss_train = np.zeros(self.edge_train.size()[1]) # 训练边的数量，[0, 0,....]
        # norm_loss_test is used in full batch evaluation (without sampling).
        self.norm_aggr_train = np.zeros(self.adj_full.size()) # 对称无环的边数， 需要对称有loop 的边数？
        # [[0,0, ...],[0,0,...0]] 1553 * 1553

        self.sample_coverage = train_params['sample_coverage']
        self.deg_train = np.array(self.adj_train.sum(1)).flatten() # 计算每个节点的度， 需计loop ?


    def par_graph_sample(self, phase):

        """
        Perform graph sampling in parallel. A wrapper function for graph_samplers.py
        """
        t0 = time.time()
        _indptr, _indices, _data, _v, _edge_index = self.graph_sampler.par_sample(phase)
        t1 = time.time()
        print('sampling 200 subgraphs: time={:.3f} sec'.format(t1 - t0), end="\r")
        self.subgraphs_remaining_indptr.extend(_indptr)
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)


    def set_sampler(self, train_phases):

        """
        Pick the proper graph sampler. Run the warm-up phase to estimate loss/aggregation coefficients.

        Inputs:

                train_phases      dict, config / params for the graph sampler

                                 {'sampler': 'edge',  'size_subg_edge': 4000, 
                                 }

        Outputs:

                None
        """ 

        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.method_sample = train_phases['sampler']

        if self.method_sample == 'edge':

            self.size_subg_budget = train_phases['size_subg_edge'] * 2  # self.size_subg_budget 是节点数
            self.graph_sampler = edge_sampling(
                self.adj_train,
                self.node_train,
                self.size_subg_budget,
            )


        self.norm_loss_train = np.zeros(self.adj_train.nnz) # 13320, 从adj_train图采样

        self.norm_aggr_train = np.zeros(13320).astype(np.float32) # 所有边数

        #------------------------------------------------------------------------
        # BELOW: estimation of loss / aggregation normalization factor
        # -----------------------------------------------------------------------
        # 1. sample enough number of subgraphs
        # 2. update the counter for each node / edge in the training graph
        # 3. estimate norm factor alpha and lambda

        tot_sampled_nodes = 0
        while True:

            self.par_graph_sample('train')
            tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_data]) # 200 subgraph

            if tot_sampled_nodes > self.sample_coverage * self.node_train.size:

                break

        num_subg = len(self.subgraphs_remaining_nodes)
        for i in range(num_subg):
            self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1

            self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1

        # -----------------------------------------------------------------------------
        # 计算标准化系数alpha_{u,v} for aggrator, lambad_{u,v} for loss 
        # for v in range(self.adj_train)

    def one_batch(self, mode='train'):

        """
        Generate one minibatch for trainer. In the 'train' mode, one minibatch corresponds to one subgraph.
        In the 'val' or 'test' mode, one batch corresponds to the full graph(i.e., full-batch rather than 
        minibatch evaluations for validation / test sets).

        Inputs:

            mode    str, can be 'train', 'val' or 'valtest

        Outputs:

            node_subgraph     np array, IDs of the subgraph / full graph nodes
            adj               scipy CSR, adj matrix of the subgraph / full graph
            norm_loss        np array, loss normalization coefficients. In 'val' or 'test' modes,
                                we don't need to normalize, and so the values in this array are all 1.

        """
        if mode in ['val', 'test', 'valtest']:

            self.node_subgraph = np.arange(self.adj_full.shape[0])
            adj = self.adj_full

        else:

            assert mode == 'train'
            if len(self.subgraphs_remaining_nodes) == 0:
                self.par_graph_sample('train')

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()
            self.size_subgraph = len(self.node_subgraph)
            adj = sp.csr_matrix(
                (self.subgraphs_remaining_data.pop(),
                 self.subgraphs_remaining_indices.pop(),
                 self.subgraphs_remaining_indptr.pop()),
                 shape=(self.size_subgraph, self.size_subgraph,)
                )
            adj_edge_index = self.subgraphs_remaining_edge_index.pop()
            norm_aggr(adj.data, adj_edge_index, self.norm_aggr_train, num_proc=20)
            
            #-----------------------------------------------------
            # adj 不进行标准化
            #adj = adj_norm(adj,  deg=self.deg_train[self.node_subgraph])
            adj = _coo_scipy2torch(adj.tocoo())
            if self.use_cuda:

                adj = adj.cuda()

            self.batch_num += 1

        norm_loss = self.norm_loss_test if mode in ['val', 'test', 'valtest'] else self.norm_loss_train
        norm_loss = norm_loss[self.node_subgraph]

        return self.node_subgraph, adj, norm_loss
            
    def num_training_batches(self):

        self.edge_train

        return math.ceil(self.node_train.shape[0] / float(self.size_subg_budget))
    
    def shuffle(self):

        self.node_train = np.random.permutation(self.node_train)

        self.batch_num = -1

    def end(self):

        return (self.batch_num + 1) * self.size_subg_budget >= self.node_train.shape[0]

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

        self.fc1 = nn.Linear(hidden, 7)
        

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


##-------------------------MINIBATCH-----------------------------------------
class MiniBatch:

    def __init__(self, adj_full, role, edge_index, train_params):

        """
        Inputs:

                adj_full    scipy CSR, adj matrix for the full graph

                role           dict, key 'train_index' -> list of training edge index[0, 6660)
                                 key 'valid_index' -> list of valid edge index [0,6660)

                edge_index    torch.tensor 2 x 13320, 固定顺序，[:,:6660], [;,6660:]  两行互换相等              
                train_params   dict, additional parameters related to trianing. e.g., 
                            how many subgraphs we want to get to estimate the norm coefficients


        transductive setting, adj_train = adj_full, 从训练图上采样， 训练图就是全图
        """
        self.index_train = np.array(role['train_index'])
        self.edge_train = edge_index[:, self.index_train]
        self.index_val = np.array(role['valid_index'])
        self.edge_val = edge_index[:, self.index_val]


        self.adj_full = _coo_scipy2torch(adj_full.tocoo())

        self.adj_train = adj_full  # transductive setting, 从全图上采样

        self.node_train = np.array(list(range(adj_full.shape[0]))) # 全图节点
        # below: book-keeping for mini-batch
        self.node_subgraph = None
        self.batch_num = -1

        #self.subgraphs_remaining_indptr = []
        #self.subgraphs_remaining_indices = []
        #self.subgraphs_remaining_data = []
        #self.subgraphs_remaining_nodes = []
        #self.subgraphs_remaining_edge_index = []

        self.sample_coverage = train_params['sample_coverage']


    def par_graph_sample(self, phase):

        """
        Perform graph sampling in parallel. 
        A wrapper function for graph_samplers.py
        """
        t0 = time.time()
        _indptr, _indices, _data, _v, _edge_index = self.graph_sampler.par_sample(phase)
        t1 = time.time()
        print('sampling 200 subgraphs: time={:.3f} sec'.format(t1-t0), end="\r")
        self.subgraphs_remaining_indptr.extend(_indptr)
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)

    def set_sampler(self, train_phases):

        """
        Pick the proper graph sampler. Run the warm-up phase to 
        estimate loss/aggregation coefficients.

        Inputs:

               train_phases dict, config / params for the graph sampler
        """
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.method_sample = train_phases['sampler']

        if self.method_sample == 'edge':

            self.size_subg_budget = train_phases['size_subg_edge'] * 2  # self.size_subg_budget 是节点数
            self.graph_sampler = edge_sampling(
                self.adj_train, # 全图
                self.node_train, # 全图
                self.size_subg_budget,
            )

        self.norm_loss_train_edge = np.zeros(self.adj_train.size) # lambda, 13320

        self.norm_loss_train = np.zeros(self.adj_train.shape[0]) # 1553

        self.norm_aggr_train = np.zeros(self.adj_train.size).astype(np.float32) # alpha, 13320

        #-----------------------------------------------------------
        #BELOW: estimation of loss / aggregation normalization factors
        #-----------------------------------------------------------
        # for all samplers:
        #  1. sample enough number of subgraphs
        #  2. update the counter for each node / edge in the training graph
        #  3. estimate norm factor alpha and lambda

        tot_sampled_nodes = 0

        while True:

            self.par_graph_sample('train')
            tot_sampled_nodes = sum([len(n) for n in self.subgraphs_remaining_nodes])
            if tot_sampled_nodes > self.sample_coverage * self.node_train.size:
                break

        num_subg = len(self.subgraphs_remaining_nodes)

        for i in range(num_subg):

            self.norm_aggr_train[self.subgraphs_remaining_edge_index[i]] += 1
            self.norm_loss_train[self.subgraphs_remaining_nodes[i]] += 1

        for v in range(self.adj_train.shape[0]):

            i_s = self.adj_train.indptr[v]
            i_e = self.adj_train.indptr[v + 1]
            val = np.clip(self.norm_loss_train[v] / self.norm_aggr_train[i_s : i_e], 0, 1e4)
            val[np.isnan(val)] = 0.1
            self.norm_aggr_train[i_s : i_e] = val

        self.norm_loss_train[np.where(self.norm_loss_train==0)[0]] = 0.1
        self.norm_loss_train = num_subg / self.norm_loss_train / self.node_train.size
        self.norm_loss_train = torch.from_numpy(self.norm_loss_train.astype(np.float32))

        print()

    def one_batch(self, mode='train'):

        """
        Generate one minibatch for trainer. In the 'train' mode, one minibatch corresponds to one subgraph.
        In the 'val' or 'test' mode, one batch corresponds to the full graph(i.e., full-batch rather than 
        minibatch evaluations for validation / test sets).

        Inputs:

            mode    str, can be 'train', 'val' or 'valtest

        Outputs:

            node_subgraph     np array, IDs of the subgraph / full graph nodes
            adj               scipy CSR, adj matrix of the subgraph / full graph
            norm_loss        np array, loss normalization coefficients. In 'val' or 'test' modes,
                                we don't need to normalize, and so the values in this array are all 1.

        """



        assert mode == 'train'

        if len(self.subgraphs_remaining_nodes) == 0:
    
            self.par_graph_sample('train')

        self.node_subgraph = self.subgraphs_remaining_nodes.pop()
        self.size_subgraph = len(self.node_subgraph)
        adj = sp.csr_matrix(
                (self.subgraphs_remaining_data.pop(),
                 self.subgraphs_remaining_indices.pop(),
                 self.subgraphs_remaining_indptr.pop()),
                 shape=(self.size_subgraph, self.size_subgraph,)
                )
        adj_edge_index = self.subgraphs_remaining_edge_index.pop()
        norm_aggr(adj.data, adj_edge_index, self.norm_aggr_train, num_proc=20)
            
        #-----------------------------------------------------
        # adj 不进行标准化
        #adj = adj_norm(adj,  deg=self.deg_train[self.node_subgraph])
        #adj = _coo_scipy2torch(adj.tocoo())

        ADJ = torch.LongTensor(torch.tensor([adj.tocoo().row, adj.tocoo().col], 
                                            dtype=torch.int64))
        

        self.batch_num += 1

        norm_loss = self.norm_loss_train

        norm_loss = norm_loss[self.node_subgraph]

        return self.node_subgraph, ADJ, norm_loss
            
    def num_training_batches(self):

        self.edge_train

        return math.ceil(self.node_train.shape[0] / float(self.size_subg_budget))
    
    def shuffle(self):

        self.node_train = np.random.permutation(self.node_train)

        self.batch_num = -1

    def end(self):

        return (self.batch_num + 1) * self.size_subg_budget >= self.node_train.shape[0]



#---------------------------------------------------
#transductive setting, sample subgraph from PPI graph 
#---------------------------------------------------

#---------------------------------------------------------------------------------------
# BELOW:从minibatch.one_batch 返回 self.node_subgraph, adj, norm_loss, 转换highPPI 能计算的数据
#---------------------------------------------------------------------------------------
class multigraph2big:

    def __init__(self, p_x_ALL, p_edge_ALL) -> None:

        """
        p_x_ALL, p_edge_ALL 是 full graph 的所有节点的特征
        """

        self.p_x_ALL = p_x_ALL

        self.p_edge_ALL = p_edge_ALL
   

    def multi2big_x(self, x_ori):

        x_cat = torch.zeros(1, 7)
        x_num_index = torch.zeros(len(x_ori))
        for i in range(len(x_ori)):
            x_now = torch.tensor(x_ori[i])
            x_num_index[i] = torch.tensor(x_now.size(0))
            x_cat = torch.cat((x_cat, x_now), 0)

        return x_cat[1:, :], x_num_index
    
    def multi2big_batch(self, x_num_index):
        num_sum = x_num_index.sum()
        num_sum = num_sum.int()
        batch = torch.zeros(num_sum)
        count = 1
        for i in range(1, len(x_num_index)):
            zj1 = x_num_index[:i]
            zj11 = zj1.sum()
            zj11 = zj11.int()
            zj22 = zj11 + x_num_index[i]
            zj22 = zj22.int()
            size1 = x_num_index[i]
            size1 = size1.int()
            tc = count * torch.ones(size1)
            batch[zj11:zj22] = tc
            test = batch[zj11:zj22]
            count = count + 1
        batch = batch.int()
        return batch

    def multi2big_edge(self, edge_ori, num_index):
        edge_cat = torch.zeros(2, 1)
        edge_num_index = torch.zeros(len(edge_ori)) # 1553
        for i in range(len(edge_ori)):
            edge_index_p = edge_ori[i]
            edge_index_p = np.asarray(edge_index_p)
            edge_index_p = torch.tensor(edge_index_p.T)
            edge_num_index[i] = torch.tensor(edge_index_p.size(1))
            if i == 0:
                offset = 0
            else:
                zj = torch.tensor(num_index[:i])
                offset = zj.sum()
            edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
        return edge_cat[:, 1:], edge_num_index

    def process(self, node_subgraph):


        p_x_all = np.array(self.p_x_ALL, dtype=object)[node_subgraph]


        p_edge_all = self.p_edge_ALL[node_subgraph]

        p_x_all, x_num_index = self.multi2big_x(p_x_all)

        p_x_all = p_x_all.to(torch.float32)

        p_edge_all, edge_num_index = self.multi2big_edge(p_edge_all, x_num_index)

        p_edge_all = torch.LongTensor(p_edge_all.to(torch.int64))

        batch = self.multi2big_batch(x_num_index)

        return p_x_all.to(torch.float32), \
                    torch.LongTensor(p_edge_all.to(torch.int64)), \
                        batch.to(torch.int64)


class lambiomhG(nn.Module):

    def __init__(self, num_classes=7, train_params=None, feat_full=None,  
                 label_full=None, cpu_eval=False):

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

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4)
        
    def forward(self, batch, p_x_all, p_edge_all, 
                adj_subgraph, train_edge_index):
        

        embs = self.BGNN(p_x_all, p_edge_all, batch)

        y = self.TGNN(embs, adj_subgraph, train_edge_index, p=0.5)

        return y
    
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



def Train_Index(train_index, subg_node):


    """
    Input:


        train_index:  tensor, edge index in train set, original edge_index in full graph

        subg_node:  tensor, subg_node is original from full graph


    Output:

        select_index: tensor, train index in  subgraph with single edge
        edge_select:  tensor,  train index edge (full graph) corresponding to select_index

  
    """
    
    nodori2sg = { x : i for i, x in enumerate(subg_node)}


    select_index = []

    edge_select = []

    
    for i in range(train_index.size()[1]):

        x = train_index[:, i].tolist()

        if x[0] in subg_node and x[1] in subg_node:

            y = [nodori2sg[x[0]], nodori2sg[x[1]]]

            if sorted(y) not in select_index:

                select_index.append(sorted(y))

                edge_select.append('-'.join(map(str,sorted(x))))


    
    return torch.tensor(select_index).T,  edge_select
    



if __name__ == '__main__':


    from lambiomHG.MODEL.utils import *


    device =torch.device('cuda:0')

    ppi_prefix = './data/shs27k'

    ppi_data = LOAD_NETDATA(ppi_prefix)

    adj_full, adj_train, p_x_all, p_edge_all, \
    labelC, role, edge_index, edge_attr = ppi_data.Generate_HiGraph()

    
    
    # keys: edge_index in full graph , e.g. '1-2', '2-3', '4-5'; values: tensor([])

    edgattr  = {}

    for i, ed in enumerate(edge_index.T):

        edgattr['-'.join(map(str, ed.tolist()))] = edge_attr[i]



    ADJ_train = torch.tensor([adj_train.tocoo().row, adj_train.tocoo().col])

   ##NOTE: edge_index 为full graph 的边， 与edge_attr 一一对应

    model = lambiomhG()

    m = nn.Sigmoid()

    train_phases = {'end': 1000, 'sampler': 'edge', 'size_subg_edge': 500}

    train_params = {'lr': 0.01, 'weight_decay': 0.0, 'norm_loss': True,
                    'norm_aggr': True,  'q_threshold': 50, 'q_offset':0,
                    'dropout': 0.1, 'sample_coverage': 50, 'eval_val_every':1}
    
    m2b = multigraph2big(p_x_all, p_edge_all)

    minibatch = MiniBatch(adj_full, role, edge_index, train_params)
                    
    epoch_ph_start = 0

    minibatch.set_sampler(train_phases)

    num_batches = minibatch.num_training_batches()


    #------------------------------------------------------------#
    # BELOW: to find required data for validation on full graph


    p_X_b, p_Edge_b, Batch = m2b.process(list(range(len(p_x_all))))

    valid_id = role["valid_index"]
    val_edge_index = edge_index[:,valid_id]
    val_label = edge_attr[valid_id]

    

    for e in range(epoch_ph_start, int(train_phases['end'])):

        print('Epoch {:4d}'.format(e))

        step = 0

        time_train_ep = 0

        minibatch.shuffle()

        


        model = model.to(device)

        while not minibatch.end():

            print("\033[0;37;43m Training info:\033[0m")

            t1 = time.time()

            node_subg, adj_subg, norm_loss = minibatch.one_batch(mode='train')

            

            p_x_all, p_edge_all, batch = m2b.process(node_subg)

            train_index, edge_select = Train_Index(ADJ_train, node_subg)

            label = torch.stack([edgattr[ed] for ed in edge_select]).to(torch.float32)

            

            ls, preds = model.train_step(batch.to(device), 
                                  p_x_all.to(device), 
                                  p_edge_all.to(device), 
                                  adj_subg.to(device), 
                                  train_index.to(device), 
                                  label.to(device))
            
            time_train_ep += time.time() - t1
            
            pre_result = (m(preds) > 0.5).type(torch.FloatTensor).to(device)
            
            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data, m(preds).cpu().data)

            

            metrics.show_result(is_print=True)

            step += 1

            print('step in a epoch : {},  training loss: {}'.format(step, ls))

        
        print("\033[0;37;43m validation info:\033[0m")

        model = model.to('cpu')
        
        val_preds = model.evaluatewithfullG(Batch, p_X_b, p_Edge_b, 
                                edge_index, val_edge_index)
        

        val_result = (m(val_preds) > 0.5).type(torch.FloatTensor)
        val_metrics = Metrictor_PPI(val_result.cpu().data, val_label.cpu().data, m(val_preds).cpu().data)

        val_metrics.show_result(is_print=True)


        print('a epoch time : {:.4f} sec'.format(time_train_ep))

        