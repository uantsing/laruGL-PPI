import numpy as np
import torch
from graph_samplers import edge_sampling
import time
import scipy.sparse as sp
import math
from laruGL.norm_aggr import *

import torch
import torch.nn as nn


def _coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.LongTensor(values)
    return torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))

#---------------------------------------------------
#transductive setting, sample subgraph from PPI graph 
#---------------------------------------------------


class MiniBatch:

    def __init__(self, adj_full, role, edge_index, train_params, adj_train=None):

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

        if adj_train != None:


            self.adj_train = adj_train # inductive setting, 从训练图上采样

        else:

            self.adj_train = adj_full  # transductive setting, 从全图上采样

        self.node_train = np.array(list(range(self.adj_train.shape[0]))) # 对应的图节点
        # below: book-keeping for mini-batch
        self.node_subgraph = None
        self.batch_num = -1

        

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

            self.size_subg_budget = train_phases['size_subg_edge'] * 2  # self.size_subg_budget 无向？所以*2
            self.graph_sampler = edge_sampling(
                self.adj_train, 
                self.node_train, 
                self.size_subg_budget,
            )

        # self.norm_loss_train_edge = np.zeros(self.adj_train.size) # lambda, 13320

        # self.norm_loss_train = np.zeros(self.adj_train.shape[0]) # 1553

        # self.norm_aggr_train = np.zeros(self.adj_train.size).astype(np.float32) # alpha, 13320

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

        
        if False:

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

        

    def one_batch(self, mode='train'):

        """
        Generate one minibatch for trainer. In the 'train' mode, one minibatch corresponds to one subgraph.
        In the 'val' or 'test' mode, one batch corresponds to the full graph(i.e., full-batch rather than 
        minibatch evaluations for validation / test sets).

        Inputs:

            mode    str, can be 'train', 'val' or 'valtest

        Outputs:

            node_subgraph     np array, IDs of the subgraph 
            adj               scipy CSR, adj matrix of the subgraph
            

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
        #adj_edge_index = self.subgraphs_remaining_edge_index.pop()
        #norm_aggr(adj.data, adj_edge_index, self.norm_aggr_train, num_proc=20)
            
        #-----------------------------------------------------
        # adj 不进行标准化
        #adj = adj_norm(adj,  deg=self.deg_train[self.node_subgraph])
        #adj = _coo_scipy2torch(adj.tocoo())

        ADJ = torch.LongTensor(torch.tensor([adj.tocoo().row, adj.tocoo().col], 
                                            dtype=torch.int64))
        

        self.batch_num += 1


        return self.node_subgraph, ADJ
    
            
    def num_training_batches(self):

        self.edge_train

        return math.ceil(self.node_train.shape[0] / float(self.size_subg_budget))
    
    
    def shuffle(self):

        self.node_train = np.random.permutation(self.node_train)

        self.batch_num = -1

    def end(self):

        return (self.batch_num + 1) * self.size_subg_budget >= self.node_train.shape[0]


# class multigraph2big:

#     def __init__(self, p_x_ALL, p_edge_ALL) -> None:

#         """
#         p_x_ALL, p_edge_ALL 是 full graph 的所有节点的特征
#         """

#         self.p_x_ALL = p_x_ALL

#         self.p_edge_ALL = p_edge_ALL
   

#     def multi2big_x(self, x_ori):

#         x_cat = torch.zeros(1, 7)
#         x_num_index = torch.zeros(len(x_ori))
#         for i in range(len(x_ori)):
#             x_now = torch.tensor(x_ori[i])
#             x_num_index[i] = torch.tensor(x_now.size(0))
#             x_cat = torch.cat((x_cat, x_now), 0)

#         return x_cat[1:, :], x_num_index
    
#     def multi2big_batch(self, x_num_index):
#         num_sum = x_num_index.sum()
#         num_sum = num_sum.int()
#         batch = torch.zeros(num_sum)
#         count = 1
#         for i in range(1, len(x_num_index)):
#             zj1 = x_num_index[:i]
#             zj11 = zj1.sum()
#             zj11 = zj11.int()
#             zj22 = zj11 + x_num_index[i]
#             zj22 = zj22.int()
#             size1 = x_num_index[i]
#             size1 = size1.int()
#             tc = count * torch.ones(size1)
#             batch[zj11:zj22] = tc
#             test = batch[zj11:zj22]
#             count = count + 1
#         batch = batch.int()
#         return batch

#     def multi2big_edge(self, edge_ori, num_index):
#         edge_cat = torch.zeros(2, 1)
#         edge_num_index = torch.zeros(len(edge_ori)) # 1553
#         for i in range(len(edge_ori)):
#             edge_index_p = edge_ori[i]
#             edge_index_p = np.asarray(edge_index_p)
#             edge_index_p = torch.tensor(edge_index_p.T)
#             edge_num_index[i] = torch.tensor(edge_index_p.size(1))
#             if i == 0:
#                 offset = 0
#             else:
#                 zj = torch.tensor(num_index[:i])
#                 offset = zj.sum()
#             edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
#         return edge_cat[:, 1:], edge_num_index

#     def process(self, node_subgraph):


#         p_x_all = np.array(self.p_x_ALL, dtype=object)[node_subgraph]


#         p_edge_all = self.p_edge_ALL[node_subgraph]

#         p_x_all, x_num_index = self.multi2big_x(p_x_all)

#         p_x_all = p_x_all.to(torch.float32)

#         p_edge_all, edge_num_index = self.multi2big_edge(p_edge_all, x_num_index)

#         p_edge_all = torch.LongTensor(p_edge_all.to(torch.int64))

#         batch = self.multi2big_batch(x_num_index)

#         return p_x_all.to(torch.float32), \
#                     torch.LongTensor(p_edge_all.to(torch.int64)), \
#                         batch.to(torch.int64)



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