
import numpy as np

import scipy

import laruGL.cython_sampler as cy

class GraphSampler:

    """
    This is the sampler super-class. Any sampler in this program is supposed to perform\
    the following meta-steps:

    1. [Optional] Preprocessing: e.g., for edge sampler, we need to calculate the sampling
                    probability for each edge in the training(full) graph. This is to be
                    proformed only once per phase(or, once throughout the whole training,
                    since in most cases, training only consists of a single phase.
                    see ../train_config/README.md for definition of a phase).

    2. Parallel sampling: launch a batch of graph samplers in parallel and sample subgraphs
                        independently. For efficiency, the actual sampling operation happen
                        in cython. And the classes here is mainly just a wrapper.
                        ==> Need to set self.cy_sampler to the appropriate cython sampler
                        in `__init__` of the sampler sub-class
    3. Post-processing: upon getting the sampled subgraphs, we need to prepare the appropriate
                        information (e.g., subgraph adj with renamed indices) to enable the 
                        Pytorch trainer. Also, we need to do data conversion from C++ to Python
                        (or, mostly numpy). Post-processing is handled within the cython sampling
                        file (./cython_sampler.pyx)

    Pseudo-code for proposed edge sampling algorithms can be found in Appendix,Algo 2 of the 
    GraphSAINT paper

    Lastly, if you don't bother with writing samplers in cython, you can still code the sampler
    subclass in pure python. In this case, we have provided a function `_helper_extract_subgraph`
    for API consistency between python and cython. An example sampler in pure python is provided
    as `NodeSamplingVanillaPython` at the bottom of this file.
    """
    def __init__(self, adj_train, node_train, size_subgraph, args_preproc):

        """
        Inputs:

             adj_train    scipy spare CSR matrix of the training graph, here, is full graph
             ? node_train   1D np array storing the indices of the training nodes, full nodes?
             ?edge_train     1D np array storing the indices of the training edges, full edges?
             size_subgraph  int, the (estimated)number of nodes in the subgraph
             args_preproc   dict, addition arguments needed for pre-processing

        Outputs:

            None

        """
        self.adj_train = adj_train
        self.node_train = np.unique(node_train).astype(np.int32) # （节点分类？）训练的节点索引 
        #self.edge_train = np.unique(edge_train).astype(np.int32) # 训练边索引
        # size in terms of number of vertices in subgraph
        self.size_subgraph = size_subgraph
        self.name_sampler = 'None'
        self.preproc(**args_preproc)

    def preproc(self, **kwargs):

        pass
    
    def par_sample(self, stage, **kwargs):

        return self.cy_sampler.par_sample()
    
    def _helper_extract_subgraph(self, node_ids):

        """
        ONLY used for serial Python sampler (NOT for the parallel cython smapler).
        Return adj of node-induced subgraph and other corresponding data struct.

        Inputs:

             node_ids    1D np array, each element is the ID in the original training graph.

        Outputs:

            indptr        np array, indptr of the subg adj CSR
            indices       np  array, indices of the subg adj CSR
            data          np array, data of the subg adj CSR. Since we have aggregator 
                          normalization, we can simply set all data values to be 1
            subg_nodes    np array, i-th element stores the node ID of the original graph
                          for the i-th node in the subgraph. Used to index the full feats 
                          and label matrices.
            subg_edge_index  np array, i-th element stores the edge ID of the original graph
                            for the i-th edge in the subgraph. Used to index the full array
                            of aggregation normalization, label matrices.
            
        """
        node_ids = np.unique(node_ids)
        node_ids.sort()
        orig2subg = {n: i for i, n in enumerate(node_ids)}
        n = node_ids.size
        indptr = np.zeros(node_ids.size + 1)
        indices = []
        subg_edge_index = []
        subg_nodes = node_ids

        for nid in node_ids:

            idx_s, idx_e = self.adj_train.indptr[nid], self.adj_train.indptr[nid + 1]
            neighs = self.adj_train.indices[idx_s : idx_e]
            for i_n, n in enumerate(neighs):

                if n in orig2subg:
                    indices.append(orig2subg[n])
                    indptr[orig2subg[nid]+1] += 1
                    subg_edge_index.append(idx_s + i_n)

        indptr = indptr.cumsum().astype(np.int64)
        indices = np.array(indices)
        subg_edge_index = np.array(subg_edge_index)
        data = np.ones(indices.size)
        assert indptr[-1] == indices.size == subg_edge_index.size
        return indptr, indices, data, subg_nodes, subg_edge_index

#-----------------------------------------------------------------
# BELOW  python wrapper for parallel sampler implemented with Cython
#------------------------------------------------------------------

class edge_sampling(GraphSampler):


    def __init__(self, adj_train, node_train, num_edges_subgraph):

        
        """

        The sampler picks edges from the training graph independently, following 
        a pre-computed edge probability distribution. i.e.,

        p_{u,v} 正比于 1/ deg_u + 1/ deg_v  对于D^(-1)A , 对A行标准化
        p_{u,v} 正比于 2/ sqrt{deg_v} * 1/ sqrt{deg_u}, 对A 对称标准化 
        Such prob. dist. is derived to minimize the variance of the minibatch estimator

        Inputs:

                adj_train : adj in trainning graph(full graph)
                node_train:  1D np array storing the indices of the training nodes, full nodes?
                num_edges_subgraph:  
        """
        #----------------------------------------------------------------#
        NUM_PAR_SAMPLER = 20 

        SAMPLES_PER_PROC = -(-200 // NUM_PAR_SAMPLER)

        self.num_edges_subgraph = num_edges_subgraph
        # num subgraph nodes may not be num_edges_subgraph *2 in many cases,
        # but it is not too import to have an accurate estimation of subgraph szie.
        # So it's probably just fine to use this number.
        self.size_subgraph = num_edges_subgraph * 2
        self.deg_train = np.array(adj_train.sum(1)).flatten()

        # 1. 对于 对称 标准化

        D = scipy.sparse.dia_matrix((1 / np.sqrt(self.deg_train), 0),
                                    shape=adj_train.shape)
        
        self.adj_train_norm = D.dot(adj_train).dot(D) # adj_train：  对称无环； self.adj_train_norm 标准化的adj_train

        super().__init__(adj_train, node_train, self.size_subgraph, {})

        self.cy_sampler = cy.Edge2(
            self.adj_train.indptr,
            self.adj_train.indices,
            self.node_train,
            NUM_PAR_SAMPLER,
            SAMPLES_PER_PROC,
            self.edge_prob_tri.row,
            self.edge_prob_tri.col,
            self.edge_prob_tri.data.cumsum(),
            self.num_edges_subgraph,
        )

    def preproc(self, **kwargs):

        """
        Compute the edge probability distribution p_{u,v}
        """                           
        self.edge_prob = scipy.sparse.csr_matrix(
        (
            np.zeros(self.adj_train.size),
            self.adj_train.indices,
            self.adj_train.indptr
        ),
        shape = self.adj_train.shape,
        )
        self.edge_prob.data[:] = self.adj_train_norm.data[:]
        _adj_trans = scipy.sparse.csr_matrix.tocsc(self.adj_train_norm)
        self.edge_prob.data += _adj_trans.data   # P_e 正比于 A_{u,v} + A_{v,u} A是标准化后的A
        self.edge_prob.data *= 2 * self.num_edges_subgraph / self.edge_prob.data.sum()

        # now edge_prob is a symmetric matrix, we only keep the 
        # upper triangle part, since adj is assumed to be undirected.
        self.edge_prob_tri = scipy.sparse.triu(self.edge_prob).astype(np.float32) # NOTE: in coo format



if __name__ == '__main__':

    from laruGL.MODEL.utils import LOAD_NETDATA

    ppi_prefix = './data/shs27k'

    ppi_data = LOAD_NETDATA(ppi_prefix)

    adj_full, adj_train, p_x_all, p_edge_all, pbatch,\
    labelC, ppi_split_dict, edge_index, edge_attr = ppi_data.Generate_HiGraph(split_mode='random')

    node_train = np.array(list(range(1553))) # 所有节点, transductive setting

    num_edges_subgraph = 3000

    NUM_PAR_SAMPLER = 20 

    SAMPLES_PER_PROC = -(-200 // NUM_PAR_SAMPLER)
    
    edge_sample = edge_sampling(adj_full, node_train, num_edges_subgraph)

    print("Well Done")
