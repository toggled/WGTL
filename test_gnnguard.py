import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCN_GNNGuard 
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import scipy.sparse as sp
import argparse
from utils import load_npz
# import gudhi as gd
import json,os
import pandas as pd
# from torch_geometric.data import InMemoryDataset, Data

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora','citeseer','polblogs','pubmed'], help='dataset')
parser.add_argument('--lr', type=float, default=0.001,  help='learning rate')
parser.add_argument('--drop_rate', type=float, default=0.5,  help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,  help='weight decay rate')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--epoch', type=float, default=200,  help='epochs')
parser.add_argument('--device', type=str,default = 'cuda:0',help='cuda:0/cuda:1/...')

args = parser.parse_args()
print(args)
args.cuda = torch.cuda.is_available()
device = args.device
# load original data and attacked data
# prefix = 'data/nettack/' # 'data/'sta
# attack='nettack' #'meta'
prefix = 'data/'
attack = 'meta'
adj, features, labels = load_npz(prefix + args.dataset + '/' + args.dataset + '.npz')
f = open(prefix + args.dataset + '/' + args.dataset + '_prognn_splits.json')
idx = json.load(f)
idx_train, idx_val, idx_test = np.array(idx['idx_train']), np.array(idx['idx_val']), np.array(idx['idx_test'])
if args.ptb_rate == 0:
    print('loading unpurturbed adj')
    perturbed_adj,_,_ = load_npz(prefix + args.dataset + '/' + args.dataset + '.npz')
else:
    print('loading perturbed adj. ')
    perturbed_adj = sp.load_npz(prefix + args.dataset + '/' + args.dataset + '_'+attack+'_adj_'+str(args.ptb_rate)+'.npz')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Setup GAT Model
# model = SAGE(nfeat=features.shape[1], nhid=32, \
#              nlayers = 2, nclass=int(labels.max())+1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device)
model = GCN_GNNGuard(nfeat=features.shape[1], nhid=64, \
                    nclass=int(labels.max())+1, \
                    dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device,
                    n_edge=1, # Specific to JKNet
                    with_relu=True,with_bias=True
                    )
model = model.to(device)
model.fit(features=features, adj = perturbed_adj, \
          labels = labels, idx_train = idx_train, \
        idx_val=idx_val, train_iters=args.epoch, verbose=True)
# # using validation to pick model
model.eval()
# You can use the inner function of model to test
acc = model.test(idx_test)
output = {'seed':args.seed,'acc':acc}
os.system('mkdir -p GNNGuard')
csv_name = 'GNNGuard/GNNGuard_acc_'+args.dataset + "_" + str(args.ptb_rate) + '.'+attack+'.csv'
if os.path.exists(csv_name):
    result_df = pd.read_csv(csv_name)
else:
    result_df = pd.DataFrame()
result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
result.to_csv(csv_name, header=True, index=False)
#print(result.head(10))
print(csv_name)
print('Mean=> ',result['acc'].mean(),' std => ',result['acc'].std())