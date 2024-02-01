import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import ChebNet 
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import scipy.sparse as sp
import argparse
from utils import load_npz
# import gudhi as gd
import json,os
import pandas as pd
from torch_geometric.data import Data

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora','citeseer','polblogs','pubmed'], help='dataset')
parser.add_argument('--lr', type=float, default=0.01,  help='learning rate')
parser.add_argument('--drop_rate', type=float, default=0.5,  help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,  help='weight decay rate')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--epoch', type=float, default=200,  help='epochs')
parser.add_argument('--device', type=str,default = 'cuda:0',help='cuda:0/cuda:1/...')
parser.add_argument('--type', type=str,default = 'meta',help='meta/net/metapgd/pgd')
parser.add_argument('--colab',type=bool,default = True)

args = parser.parse_args()
print(args)
args.cuda = torch.cuda.is_available()
device = args.device
# load original data and attacked data
if args.type == 'meta':
    prefix = ["data/",'/content/drive/MyDrive/WitnesscomplexGNNs/data/'][args.colab]
    attack = 'meta'
if args.type=='net':
    prefix = ["data/nettack/",'/content/drive/MyDrive/WitnesscomplexGNNs/data/nettack/'][args.colab]
    attack='nettack' 
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


# Setup ChebNet Model
model = ChebNet(nfeat=features.shape[1], nhid=32, num_hops=3, \
            nclass=int(labels.max())+1, \
            dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, \
            with_bias=True, device=device)
model = model.to(device)
features = torch.FloatTensor(features.todense()).float()
labels = torch.LongTensor(labels)
idx_train0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
idx_train0[idx_train] = 1
idx_val0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
idx_val0[idx_val] = 1
idx_test0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
idx_test0[idx_test] = 1
data = Data(x=features,edge_index = torch.LongTensor(np.array(perturbed_adj.nonzero())),\
                    y=labels,train_mask=idx_train0, val_mask = idx_val0, test_mask = idx_test0)

model.fit(data, train_iters=args.epoch, patience = 100, verbose=True)
# model.train_with_early_stopping(data, args.epoch, 100, True)
# # using validation to pick model
# model.eval()
# You can use the inner function of model to test
acc = model.test()
output = {'seed':args.seed,'acc':acc}
prefix2 = ["Chebnet",'/content/drive/MyDrive/WitnesscomplexGNNs/Chebnet'][args.colab]
os.system('mkdir -p '+prefix2)
csv_name = prefix2+'/Chebnet_acc_'+args.dataset + "_" + str(args.ptb_rate) + '.'+attack+'.csv'
if os.path.exists(csv_name):
    result_df = pd.read_csv(csv_name)
else:
    result_df = pd.DataFrame()
result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
result.to_csv(csv_name, header=True, index=False)
#print(result.head(10))
print(csv_name)
print('Mean=> ',result['acc'].mean(),' std => ',result['acc'].std())