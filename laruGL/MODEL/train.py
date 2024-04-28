from laruGL.MODEL.utils import *
from laruGL.MODEL.gnn import lambiomhG
from laruGL.MODEL.minibatch_ import *

import torch

import torch.nn as nn

import argparse
import os

parser = argparse.ArgumentParser(description='model_training')
parser.add_argument('--ppi_prefix', default='./data/shs27k', type=str,
                    help='ppi dataset prefix')

parser.add_argument('--setting', default='transductive', type=str,
                    help='to select setting, transductive or inductive')

parser.add_argument('--split_mode', default='random', type=str,
                    help='split method, random, bfs, or dfs')

parser.add_argument('--size_subg_edge', default=None, type=int, 
                    help='subgraph size (signal edge), for 27k, max=6660')


parser.add_argument('--device', default='cuda:0', type=str,
                    help='used device, cpu, cuda:0, cuda:1, cuda:2....')




args = parser.parse_args()

device =torch.device(args.device)

ppi_prefix = args.ppi_prefix

savedir = './experiment/{}/{}'.format(args.setting, args.split_mode)
if not os.path.exists(savedir):
    os.mkdir(savedir)

with open("{}/config.txt".format(savedir), 'a+') as f:
        
    f.write(f"setting:{args.setting}\nsplit_mode:{args.split_mode}\n")
    f.write(f"size_subg_edge:{str(args.size_subg_edge)}\n")
    f.flush()


ppi_data = LOAD_NETDATA(ppi_prefix)

adj_full, adj_train, p_x_all, p_edge_all, \
    role, edge_index, edge_attr = ppi_data.Generate_HiGraph(split_mode=args.split_mode, 
                                                            savedir=savedir)

edgattr  = {}

for i, ed in enumerate(edge_index.T):
        
    edgattr['-'.join(map(str, ed.tolist()))] = edge_attr[i]


ADJ_train = torch.tensor([adj_train.tocoo().row, adj_train.tocoo().col])

labGx = np.load("data/shs27k/features_200.npy")
labGx = torch.tensor(labGx).to(device)
labe = torch.LongTensor(torch.tensor([[0, 0,  1,  2],
                                    [1, 5,  5,  4]])).to(device)

model = lambiomhG(labGx=labGx, labe=labe)

m = nn.Sigmoid()

train_phases = {'end': 500, 'sampler': 'edge', 'size_subg_edge': 2000} # 500, args.size_subg_edge

train_params = {'lr': 0.01, 'weight_decay': 0.0, 'norm_loss': True,
                    'norm_aggr': True,  'q_threshold': 50, 'q_offset':0,
                    'dropout': 0.1, 'sample_coverage': 50, 'eval_val_every':1}


M2B = Multigraph2Big(p_x_all, p_edge_all)

assert args.setting in ["transductive", "inductive"], "unknown setting! "

if args.setting == "transductive":
     
    minibatch = MiniBatch(adj_full, role, edge_index, train_params)

else:
     
    minibatch = MiniBatch(adj_full, role, edge_index, train_params, adj_train)


epoch_ph_start = 0

minibatch.set_sampler(train_phases)

num_batches = minibatch.num_training_batches()

#------------------------------------------------------------#
# BELOW: to find required data for validation on full graph


loader_full= M2B.process(list(range(len(p_x_all))))

p_X_b, p_Edge_b, batcH = loader_full.x, loader_full.edge_index, loader_full.batch

valid_id = role["valid_index"]

val_edge_index = edge_index[:,valid_id]

val_label = edge_attr[valid_id]

global_best_valid_f1 = 0.0

for e in range(epoch_ph_start, int(train_phases['end'])):

        print('Epoch {:4d}'.format(e))

        step = 0

        time_train_ep = 0

        minibatch.shuffle()

        model = model.to(device)

        while not minibatch.end():
              

            t1 = time.time()

            node_subg, adj_subg = minibatch.one_batch(mode='train') 

            loader = M2B.process(node_subg)

            p_x_all_, p_edge_all_, batch_ = loader.x, loader.edge_index, loader.batch


            train_index, edge_select = Train_Index(ADJ_train, node_subg)

            label = torch.stack([edgattr[ed] for ed in edge_select]).to(torch.float32)

            ls, preds = model.train_step(batch_.to(device), 
                                  p_x_all_.to(device), 
                                  p_edge_all_.to(device), 
                                  adj_subg.to(device), 
                                  train_index.to(device), 
                                  label.to(device))
            
            time_train_ep += time.time() - t1
            
            pre_result = (m(preds) > 0.5).type(torch.FloatTensor).to(device)
            
            metrics = Metrictor_PPI(pre_result.cpu().data, label.cpu().data, m(preds).cpu().data)

            metrics.show_result(is_print=False)

            step += 1

            print('step {} @ epoch {},  training loss: {}'.format(step, e, ls))

        
        
        print_file("epoch: {}, Train: label_loss: {}, precision: {}, recall: {}, f1: {}"
                       .format(e, ls, metrics.Precision, metrics.Recall, metrics.F1),
                       "{}/{}_training.txt".format(savedir, args.split_mode))
        print('a epoch time : {:.4f} sec'.format(time_train_ep))
        

        print("\033[0;37;43m validation info:\033[0m")

        #model = model.to('cpu')
        
        val_preds = model.evaluatewithfullG(batcH.to(device), p_X_b.to(device), p_Edge_b.to(device), 
                                edge_index.to(device), val_edge_index.to(device))
        

        val_result = (m(val_preds) > 0.5).type(torch.FloatTensor)
        val_metrics = Metrictor_PPI(val_result.cpu().data, val_label.cpu().data, m(val_preds).cpu().data)

        val_metrics.show_result(is_print=False)


        print_file("epoch:{}, Validation: precision:{}, recall:{}, f1:{}, Aupr: {}"
                   .format(e, val_metrics.Precision, val_metrics.Recall,
                            val_metrics.F1, val_metrics.Aupr),
                    "{}/{}_training.txt".format(savedir, args.split_mode))
        
        if global_best_valid_f1 < val_metrics.F1:


            global_best_valid_f1 = val_metrics.F1

            val_p = val_metrics.Precision

            val_r = val_metrics.Recall

            val_aupr = val_metrics.Aupr

            best_epoch = e

            
            torch.save({'epoch': e,
                        'state_dict': model.state_dict()},
                        "{}/{}_gnn_model_valid_best.ckpt".format(savedir, args.split_mode)
                        )
            
print_file("size subgraph edge : {}\n Best epoch: {}, Validation: precision:{}, recall:{}, f1:{}, Aupr:{}"
           .format(args.size_subg_edge, best_epoch, val_p, val_r, global_best_valid_f1, val_aupr),
           "{}/{}_test.txt".format(savedir, args.split_mode))

        
        
        
              





