
import copy
from tqdm import tqdm
import numpy as np
import torch
import scipy.sparse as sp
import random
import json
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from sklearn.metrics import precision_recall_curve, auc



def print_file(str_, save_file_path=None):
    print(str_)
    if save_file_path != None:
        f = open(save_file_path, 'a')
        print(str_, file=f)

# ---------------------------------#
# PPI 任务的评价指标
# ---------------------------------#
class Metrictor_PPI:
    def __init__(self, pre_y, truth_y, true_prob, is_binary=False):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.pre = np.array(pre_y).squeeze()
        self.tru = np.array(truth_y).squeeze()
        self.true_prob = np.array(true_prob).squeeze()
        if is_binary:
            length = pre_y.shape[0]
            for i in range(length):
                if pre_y[i] == truth_y[i]:
                    if truth_y[i] == 1:
                        self.TP += 1
                    else:
                        self.TN += 1
                elif truth_y[i] == 1:
                    self.FN += 1
                elif pre_y[i] == 1:
                    self.FP += 1
            self.num = length

        else:
            N, C = pre_y.shape
            for i in range(N):
                for j in range(C):
                    if pre_y[i][j] == truth_y[i][j]:
                        if truth_y[i][j] == 1:
                            self.TP += 1
                        else:
                            self.TN += 1
                    elif truth_y[i][j] == 1:
                        self.FN += 1
                    elif truth_y[i][j] == 0:
                        self.FP += 1
            self.num = N * C

    def show_result(self, is_print=False, file=None):
        self.Accuracy = (self.TP + self.TN) / (self.num + 1e-10)
        self.Precision = self.TP / (self.TP + self.FP + 1e-10)
        self.Recall = self.TP / (self.TP + self.FN + 1e-10)
        self.F1 = 2 * self.Precision * self.Recall / (self.Precision + self.Recall + 1e-10)

        aupr_entry_1 = self.tru
        aupr_entry_2 = self.true_prob
        aupr = np.zeros(7)
        for i in range(7):
            precision, recall, _ = precision_recall_curve(aupr_entry_1[:,i], aupr_entry_2[:,i])
            aupr[i] = auc(recall,precision)
        self.Aupr = aupr

        if is_print:
            print_file("Accuracy: {}".format(self.Accuracy), file)
            print_file("Precision: {}".format(self.Precision), file)
            print_file("Recall: {}".format(self.Recall), file)
            print_file("F1-Score: {}".format(self.F1), file)


#---------------------------------#
# BELOW   dataset split mode 
# --------------------------------#

class UnionFindSet(object):

    def __init__(self, m):
        
        self.roots = [i for i in range(m)]
        self.rank = [0 for i in range(m)]
        self.count = m

        for i in range(m):
            self.roots[i] = i

    def find(self, member):
        tmp = []
        while member != self.roots[member]:
            tmp.append(member)
            member = self.roots[member]
        for root in tmp:
            self.roots[root] = member
        return member

    def union(self, p, q):
        parentP = self.find(p)
        parentQ = self.find(q)
        if parentP != parentQ:
            if self.rank[parentP] > self.rank[parentQ]:
                self.roots[parentQ] = parentP
            elif self.rank[parentP] < self.rank[parentQ]:
                self.roots[parentP] = parentQ
            else:
                self.roots[parentQ] = parentP
                self.rank[parentP] -= 1
            self.count -= 1


def get_bfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    candiate_node = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    candiate_node.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        cur_node = candiate_node.pop(0)
        selected_node.append(cur_node)
        for edge_index in node_to_edge_index[cur_node]:

            if edge_index not in selected_edge_index:
                selected_edge_index.append(edge_index)

                end_node = -1
                if ppi_list[edge_index][0] == cur_node:
                    end_node = ppi_list[edge_index][1]
                else:
                    end_node = ppi_list[edge_index][0]

                if end_node not in selected_node and end_node not in candiate_node:
                    candiate_node.append(end_node)
            else:
                continue
        # print(len(selected_edge_index), len(candiate_node))
    node_list = candiate_node + selected_node
    # print(len(node_list), len(selected_edge_index))
    return selected_edge_index


def get_dfs_sub_graph(ppi_list, node_num, node_to_edge_index, sub_graph_size):
    stack = []
    selected_edge_index = []
    selected_node = []

    random_node = random.randint(0, node_num - 1)
    while len(node_to_edge_index[random_node]) > 5:
        random_node = random.randint(0, node_num - 1)
    stack.append(random_node)

    while len(selected_edge_index) < sub_graph_size:
        
        cur_node = stack[-1]
        if cur_node in selected_node:
            flag = True
            for edge_index in node_to_edge_index[cur_node]:
                if flag:
                    end_node = -1
                    if ppi_list[edge_index][0] == cur_node:
                        end_node = ppi_list[edge_index][1]
                    else:
                        end_node = ppi_list[edge_index][0]

                    if end_node in selected_node:
                        continue
                    else:
                        stack.append(end_node)
                        flag = False
                else:
                    break
            if flag:
                stack.pop()
            continue
        else:
            selected_node.append(cur_node)
            for edge_index in node_to_edge_index[cur_node]:
                if edge_index not in selected_edge_index:
                    selected_edge_index.append(edge_index)

    return selected_edge_index



#------------------------------------------#
# BELOW 将 多个protein graph 转化为 big data
# -----------------------------------------# 



# def multi2big_x(x_ori):

#     x_cat = torch.zeros(1, 7)
#     x_num_index = torch.zeros(1553)
#     for i in range(1553):
#         x_now = torch.tensor(x_ori[i])
#         x_num_index[i] = torch.tensor(x_now.size(0))
#         x_cat = torch.cat((x_cat, x_now), 0)
#     return x_cat[1:, :], x_num_index


# def multi2big_batch(x_num_index):

#     num_sum = x_num_index.sum()
#     num_sum = num_sum.int()
#     batch = torch.zeros(num_sum)
#     count = 1
#     for i in range(1,1553):
#         zj1 = x_num_index[:i]
#         zj11 = zj1.sum()
#         zj11 = zj11.int()
#         zj22 = zj11 + x_num_index[i]
#         zj22 = zj22.int()
#         size1 = x_num_index[i]
#         size1 = size1.int()
#         tc = count * torch.ones(size1)
#         batch[zj11:zj22] = tc
#         count = count + 1
#     batch = batch.int()
#     return batch


# def multi2big_edge(edge_ori, num_index):
    
#     edge_cat = torch.zeros(2, 1)
#     edge_num_index = torch.zeros(1553)
#     for i in range(1553):
#         edge_index_p = edge_ori[i]
#         edge_index_p = np.asarray(edge_index_p)
#         edge_index_p = torch.tensor(edge_index_p.T)
#         edge_num_index[i] = torch.tensor(edge_index_p.size(1))
#         if i == 0:
#             offset = 0
#         else:
#             zj = torch.tensor(num_index[:i])
#             offset = zj.sum()
#         edge_cat = torch.cat((edge_cat, edge_index_p + offset), 1)
        
#     return edge_cat[:, 1:], edge_num_index


class Multigraph2Big:

    def __init__(self, p_x_all, p_edge_all) -> None:

        """
        p_x_all, p_edge_all 是 full graph 的所有节点的特征
        """

        assert len(p_x_all) == len(p_edge_all)

        self.graph_list = []

        for i in range(len(p_x_all)):

            self.graph_list.append(Data(x=torch.Tensor(p_x_all[i]), 
                                  edge_index=torch.LongTensor(p_edge_all[i]).transpose(1, 0)))
            

    def process(self, node_subgraph):


        selected_data = [self.graph_list[node] for node in node_subgraph]
    
        loader = Batch.from_data_list(selected_data)

        return loader      



#--------------------------------------------------------------#
# bulid a list of protein graph #
# def prographlist(p_x_all, p_edge_all):

#     assert len(p_x_all) == len(p_edge_all)

#     graph_list = []

#     for i in range(len(p_x_all)):

#         graph_list.append(Data(x=torch.Tensor(p_x_all[i]), 
#                                   edge_index=torch.LongTensor(p_edge_all[i]).transpose(1, 0)))
        
#     return graph_list








# class CPIDataSet(InMemoryDataset):

#     def __init__(self, dataset:str, root='./',
#                 drugn_path=None, proteinX_path=None,
#                 proteinE_path=None, CPIdata = None,
#                 seq_full=None,
#                 transform=None, pre_transform=None):
#         # 对CPI文件分割之后，再进入数据集类，
#         # CPI_path 表示训练集或测试集
#         super(CPIDataSet, self).__init__(root, transform, pre_transform)    

#         self.dataset = dataset
#         self.drugn_path = drugn_path
#         self.proteinX_path = proteinX_path
#         self.proteinE_path = proteinE_path
        
#         self.cpi = CPIdata  # 数据框

#         if os.path.isfile(self.processed_paths[0]):

#             print('Pre-processed data found:{}, loading...'.format(self.processed_paths[0]))

#             self.data, self.slices = torch.load(self.processed_paths[0])

#         else:

#             print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))

#             self.load_data()

#             self.process()

#             self.data, self.slices = torch.load(self.processed_paths[0])

    
#     @property
#     def raw_file_names(self):

#         pass
#     @property
#     def processed_file_names(self):

#         return  [self.dataset + '.pt']
    
#     def download(self):
        
#         pass
#     def _download(self):
        
#         pass
    
#     def _process(self):

#         if not os.path.exists(self.processed_dir):

#             os.makedirs(self.processed_dir)

#     def load_data(self):

#         with open(self.drugn_path) as file:

#             self.drugn = json.load(file)

#         self.p_x_all = torch.load(self.proteinX_path)

#         self.p_edge_all = np.load(self.proteinE_path, allow_pickle=True)


#     def process(self):

#         xd_n = list(self.cpi["SMILES"])  # use cpi["SMILES"] instead of cpi.SMILES(顺序会变)

#         xp_n = list(self.cpi["uniprotID"])

#         y = list(self.cpi['kiba'])

#         dnGData_list = []

#         data_len = len(xd_n)

#         for i in range(data_len):

#             d_n = str(xd_n[i])

#             target_x = self.p_x_all[xp_n[i]]

#             target_e = self.p_edge_all[xp_n[i]]

#             label = y[i]

#             dnGData = DATA.Data(
#                 x=torch.Tensor(target_x),
#                 edge_index=torch.LongTensor(target_e).transpose(1,0),
#                 y = torch.FloatTensor([label]),
#                 d_n=d_n)
            
#             dnGData_list.append(dnGData)

#         data, slices = self.collate(dnGData_list)

#         torch.save((data, slices), self.processed_paths[0])


#---------------------------------------------------------------------------#
# 加载网络数据并构建Hiearachical Graph with edge label(attribute)
#----------------------------------------------------------------------------#


class LOAD_NETDATA:

    """
    加载网络数据

        网络数据存储于文件夹prefix中
    """

    def __init__(self, prefix):

        self.ppi_path = '{}/protein.actions.SHS27K.STRING.txt'.format(prefix)
        self.pseq_path = '{}/AF_SHS27K_uniprot_seq.txt'.format(prefix)
        
        self.p_feat_matrix = '{}/AF_x_list_7.pt'.format(prefix)
        self.p_adj_matrix = '{}/AF_edge_12_sgnl.npy'.format(prefix)

        

    def load_ppi_data(self, skip_head=True):

        self.ppi_list = []
        self.ppi_dict = {}
        self.ppi_label_list = []
        self.protein_dict = {}
        self.protein_name = {}

        name = 0
        ppi_name = 0
        self.node_num = 0
        self.edge_num = 0

        class_map = {'reaction':0,
                     'binding': 1,
                     'ptmod': 2,
                     'activation': 3,
                     'inhibition': 4,
                     'catalysis': 5,
                     'expression':6}
        
        for line in tqdm(open(self.ppi_path)):

            if skip_head:

                skip_head = False
                continue
            line = line.strip().split('\t')

            # get node and node name
            if line[0] not in self.protein_name.keys():
                self.protein_name[line[0]] = name
                name += 1

            if line[1] not in self.protein_name.keys():
                self.protein_name[line[1]] = name
                name += 1

            # get edge and its label
            temp_data = ""

            if line[0] < line[1]:

                temp_data = line[0] + "__" + line[1]

            else:
                temp_data = line[1] + "__" + line[0]

            if temp_data not in self.ppi_dict.keys():

                self.ppi_dict[temp_data] = ppi_name
                temp_label = [0] * 7
                temp_label[class_map[line[2]]] = 1
                self.ppi_label_list.append(temp_label)
                ppi_name += 1

            else:
                index = self.ppi_dict[temp_data]
                temp_label = self.ppi_label_list[index]
                temp_label[class_map[line[2]]] = 1
                self.ppi_label_list[index] = temp_label

        
        i = 0
        for ppi in tqdm(self.ppi_dict.keys()):

            name = self.ppi_dict[ppi]
            assert name == i
            i += 1
            temp = ppi.strip().split('__')
            self.ppi_list.append(temp)
        
        ppi_num = len(self.ppi_list)
        self.origin_ppi_list = copy.deepcopy(self.ppi_list)
        assert len(self.ppi_list) == len(self.ppi_label_list)
        for i in tqdm(range(ppi_num)):
            seq1_name = self.ppi_list[i][0]
            seq2_name = self.ppi_list[i][1]

            self.ppi_list[i][0] = self.protein_name[seq1_name]
            self.ppi_list[i][1] = self.protein_name[seq2_name]

        for i in tqdm(range(ppi_num)):

            temp_ppi = self.ppi_list[i][::-1]
            temp_ppi_label = self.ppi_label_list[i]

            self.ppi_list.append(temp_ppi)
            self.ppi_label_list.append(temp_ppi_label)

        self.node_num = len(self.protein_name)
        self.edge_num = len(self.ppi_list)

    def get_connected_num(self):

        self.ufs = UnionFindSet(self.node_num)
        ppi_ndary = np.array(self.ppi_list)
        for edge in ppi_ndary:
            start, end = edge[0], edge[1]
            self.ufs.union(start, end)

    def generate_PPI_full(self):

        """
        generate full PPI adj, PPI label (edge attribute)
        """

        self.get_connected_num()

        print("Connected domain num:{}".format(self.ufs.count))

        ppi_list = np.array(self.ppi_list)
        ppi_label_list = np.array(self.ppi_label_list)

        self.edge_index = torch.tensor(ppi_list, dtype=torch.long).T # 2  x 13320
        
        self.edge_attr = torch.tensor(ppi_label_list, dtype=torch.long) # 13320 x 7

        row = self.edge_index[0, :]
        col = self.edge_index[1, :]
        self.adj_full = sp.csr_matrix(([1]*len(row), (row, col)), shape=[self.node_num, self.node_num])

        print()

    def split_dataset(self, test_size=0.2, 
                      random_new=True, mode='random', savedir=None):

        if random_new:

            if mode == 'random':

                ppi_num = int(self.edge_num // 2)
                random_list = [i for i in range(ppi_num)]
                random.shuffle(random_list)

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = random_list[: int(ppi_num * (1 - test_size))]
                self.ppi_split_dict['valid_index'] = random_list[int(ppi_num * (1 - test_size)):]

                jsobj = json.dumps(self.ppi_split_dict)

                with open('{}/{}_train_valid_index.json'.format(savedir, mode) , 'w') as f:

                    f.write(jsobj)
                    f.close()

            elif mode == 'bfs' or mode == 'dfs':

                print("use {} method split train and valid dataset".format(mode))
                node_to_edge_index = {}
                edge_num = int(self.edge_num // 2)
                for i in range(edge_num):

                    edge = self.ppi_list[i]
                    if edge[0] not in node_to_edge_index.keys():
                        node_to_edge_index[edge[0]] = []

                    node_to_edge_index[edge[0]].append(i)

                    if edge[1] not in node_to_edge_index.keys():

                        node_to_edge_index[edge[1]] = []

                    node_to_edge_index[edge[1]].append(i)

                node_num = len(node_to_edge_index)

                sub_graph_size = int(edge_num * test_size)

                if mode == 'bfs':

                    selected_edge_index = get_bfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)

                elif mode == 'dfs':
                    
                    selected_edge_index = get_dfs_sub_graph(self.ppi_list, node_num, node_to_edge_index, sub_graph_size)

                all_edge_index = [i for i in range(edge_num)]

                unselected_edge_index = list(set(all_edge_index).difference(set(selected_edge_index)))

                self.ppi_split_dict = {}
                self.ppi_split_dict['train_index'] = unselected_edge_index
                self.ppi_split_dict['valid_index'] = selected_edge_index

                assert len(unselected_edge_index) + len(selected_edge_index) == edge_num

                jsobj = json.dumps(self.ppi_split_dict)

                with open('{}/{}_train_valid_index.json'.format(savedir, mode), 'w') as f:

                    f.write(jsobj)
                    f.close()

            else:
                print("your mode is {}, you should use bfs, dfs or random".format(mode))

            
        else:
            with open('{}/{}_train_valid_index.json'.format(savedir, mode), encoding='utf-8-sig', errors='ignore') as f:
                str = f.read()
                self.ppi_split_dict = json.loads(str, strict=False)
                f.close()

    def generate_adj_train(self):

        train_edge_id = self.ppi_split_dict['train_index']

        train_edge = self.edge_index[:, train_edge_id]

        r = torch.cat((train_edge[0, :], train_edge[1,:]))
        c = torch.cat((train_edge[1, :], train_edge[0,:]))
        self.adj_train = sp.csr_matrix(([1]*len(r), (r, c)), shape=[self.node_num, self.node_num])

    def load_prograph(self):

        self.p_x_all = torch.load(self.p_feat_matrix)
        self.p_edge_all = np.load(self.p_adj_matrix, allow_pickle=True)



    def Generate_HiGraph(self, split_mode, savedir):

        self.load_ppi_data()
        self.get_connected_num()
        self.generate_PPI_full() # 对称无环
        self.split_dataset(mode=split_mode, savedir=savedir)
        self.generate_adj_train() # 对称无环
        self.load_prograph() # _sgnl 单边 无loop
        

        return (self.adj_full, self.adj_train, 
              self.p_x_all, self.p_edge_all, 
              self.ppi_split_dict, self.edge_index,
              self.edge_attr)





if __name__ == '__main__':


    ppi_prefix = './data/shs27k'

    ppi_data = LOAD_NETDATA(ppi_prefix)

    

    adj_full, adj_train, p_x_all, p_edge_all, pbatch,\
    ppi_split_dict, edge_index, edge_attr = ppi_data.Generate_HiGraph() 
    # edge_index 是边[[0, 1,2,3..],[2,5, 0, 4..]], (2, 13320); edge_attr 边标签：（13320， 7）



    print('Done')


