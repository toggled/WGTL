import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCN, GWitCompNN, LWitCompNN_V1, LWitCompNN_V2, LWitCompNN_V3, CWitCompNN_V1, CWitCompNN_V2, CWitCompNN_V3
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import scipy.sparse as sp
import argparse
from utils import load_npz
#import gudhi as gd
import json,os 
import pandas as pd 

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora','citeseer','polblogs','pubmed'], help='dataset')
parser.add_argument('--lr', type=float, default=0.001,  help='learning rate')
parser.add_argument('--drop_rate', type=float, default=0.5,  help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,  help='weight decay rate')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--epoch', type=int, default=200,  help='epochs')
parser.add_argument('--alpha', type=float, default=0.8,  help='alpha')
parser.add_argument('--beta', type=float, default=0.1,  help='beta')
parser.add_argument('--gamma', type=float, default=0.1,  help='gamma')
parser.add_argument('--lambda_coeff', type=float, default=0.01,  help='lambda_coeff')
parser.add_argument('--nhid', type=int, default=128,  help='nhid')
# parser.add_argument('--topo', type=str,default = 'witptb_local',help='witorig/witptb/witptb_local/vrorig/vrptb')
parser.add_argument('--topo', type=str,default = 'local',help='local/global/both')
parser.add_argument('--method',type=str,default='transformer',help='transformer/resnet/cnn')
parser.add_argument('--device', type=str,default = 'cuda:0',help='cuda:0/cuda:1/...')
parser.add_argument('--lm_perc',type=float,default=0.05,help='%nodes as landmarks')
parser.add_argument('--vr',type=bool,default = False)
parser.add_argument('--spvr',type=bool,default = False)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
# load original data and attacked data
#prefix = 'data/nettack/' # 'data/'
#attack='nettack' #'meta'
prefix =  'data/'
attack='meta'
#prefix = 'data/pgd/' # 'data/'
#attack='pgd' #'meta'
#prefix = 'data/metapgd/' # 'data/'
#attack='metapgd' #'meta'
vr=['','.vr'][args.vr]
# vr=['','.spvr'][args.spvr]
globalpi = prefix + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_'+str(args.lm_perc)+'_PI' + vr+'.npz'
localpi = prefix + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_'+str(args.lm_perc)+'_localPI' + vr+'.npz'

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

# Setup WitCompNN Model        
# load witness complex topological features
if args.topo == 'witorig':
    witness_complex_feat = torch.FloatTensor(np.load(prefix + args.dataset + '/' + args.dataset + '_PI' + '.npz', allow_pickle=True)['arr_0'])

# if args.topo == 'witptb':
elif args.topo == 'global': # load Global PI computed on the perturbed Adj matrix. GWTL
    # witness_complex_feat = torch.FloatTensor(np.random.rand(1,50,50)) #torch.FloatTensor(np.load('data/' + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_PI' + '.npz', allow_pickle=True)['arr_0'])
    witness_complex_feat = torch.FloatTensor(np.load(globalpi, allow_pickle=True)['arr_0'])
    print('shape of global PI representation: ',witness_complex_feat.shape)

elif args.topo == 'local': # Load Local PI computed on the perturbed Adj matrix. LWTL
    # local_witness_complex_feat => Shape (#nodes x 50 x 50)
    tmp = np.load(localpi, allow_pickle=True)['arr_0']
    # print('tmp.shape: ',tmp.shape)
    # local_witness_complex_feat_ = np.random.rand(2485, 50, 50) # torch.FloatTensor(np.load('data/' + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_localPI' + '.npz', allow_pickle=True)['arr_0'])
    local_witness_complex_feat_ = np.expand_dims(tmp, axis=1)  # (#nodes, 1, pi_dim, pi_dim)
    local_witness_complex_feat = torch.FloatTensor(local_witness_complex_feat_)
    print('shape of local PI representation: ',local_witness_complex_feat.shape) # (#nodes, 1, pi_dim, pi_dim)
else: # GWTL + LWTL + Topoloss
    # for both
    global_witness_complex_feat = torch.FloatTensor(np.load(globalpi, allow_pickle=True)['arr_0']) # torch.FloatTensor(np.random.rand(1, 50, 50))
    local_witness_complex_feat_ = np.load(localpi, allow_pickle=True)['arr_0']
    local_witness_complex_feat_ = np.expand_dims(local_witness_complex_feat_, axis=1)  # (#nodes, 1, pi_dim, pi_dim)
    local_witness_complex_feat = torch.FloatTensor(local_witness_complex_feat_)

    witness_complex_feats = [global_witness_complex_feat, local_witness_complex_feat]
    print('shapes of PIs representation: ', global_witness_complex_feat.shape, local_witness_complex_feat.shape)


topo_type = args.topo # 'local', 'global'
method = args.method
aggregation_method = 'attention' # einsum, weighted_sum, attention

if topo_type == 'global':
    model = GWitCompNN(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate,  lr=args.lr, weight_decay=args.weight_decay,  aggregation_method = aggregation_method, device=device)
    model = model.to(device)
    model.fit(features, perturbed_adj, witness_complex_feat, labels, idx_train, train_iters=args.epoch, verbose=True)
elif topo_type == 'local':
    if method == 'resnet':
        print("You are using ResNet now!")
        model = LWitCompNN_V1(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, aggregation_method = aggregation_method)
        model = model.to(device)
        model.fit(features, perturbed_adj, local_witness_complex_feat, labels, idx_train, train_iters=args.epoch,  verbose=True)
    elif method == 'cnn':
        print("You are using CNN now!")
        model = LWitCompNN_V2(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, aggregation_method = aggregation_method)
        model = model.to(device)
        model.fit(features, perturbed_adj, local_witness_complex_feat, labels, idx_train, train_iters=args.epoch, verbose=True)
    elif method == 'transformer':
        print("You are using Transformer now!")
        model = LWitCompNN_V3(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, aggregation_method = aggregation_method)
        model = model.to(device)
        model.fit(features, perturbed_adj, local_witness_complex_feat, labels, idx_train, train_iters=args.epoch, verbose=True)
else:
    # for both
    if method == 'resnet':
        print("You are using ResNet on global&local features now!")
        model = CWitCompNN_V1(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, gamma = args.gamma, lambda_coeff = args.lambda_coeff, aggregation_method = aggregation_method)
        model = model.to(device)
        model.fit(features, perturbed_adj, witness_complex_feats[0], witness_complex_feats[1], labels, idx_train, idx_val, train_iters=args.epoch,  verbose=True)
    elif method == 'cnn':
        print("You are using CNN on global&local features now!")
        model = CWitCompNN_V2(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, gamma = args.gamma, lambda_coeff = args.lambda_coeff, aggregation_method = aggregation_method)
        model = model.to(device)
        model.fit(features, perturbed_adj, witness_complex_feats[0], witness_complex_feats[1], labels, idx_train, idx_val, train_iters=args.epoch, verbose=True)
    elif method == 'transformer':
        print("You are using Transformer on global&local features now!")
        model = CWitCompNN_V3(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, gamma = args.gamma, lambda_coeff = args.lambda_coeff, aggregation_method = aggregation_method)
        model = model.to(device)
        model.fit(features, perturbed_adj, witness_complex_feats[0], witness_complex_feats[1], labels, idx_train, idx_val, train_iters=args.epoch, verbose=True)


# # using validation to pick model
# model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
model.eval()
# You can use the inner function of model to test
acc = model.test(idx_test)

output = {'seed':args.seed,'acc':acc}
csv_name = aggregation_method+"_"+args.topo+'_'+args.method+'_'+args.dataset + "_" + str(args.ptb_rate)+'_'+str(args.lm_perc) + '.'+attack+vr+'.csv'
if os.path.exists(csv_name):
    result_df = pd.read_csv(csv_name)
else:
    result_df = pd.DataFrame()
result = pd.concat([result_df, pd.DataFrame(output,index = [0])])
result.to_csv(csv_name, header=True, index=False)
# print(result.head(10))
print(csv_name)
print('Mean=> ',result['acc'].mean(),' std => ',result['acc'].std())


# local_witness_complex_feat_ = np.expand_dims(local_witness_complex_feat_, axis=1)  # (#nodes, 1, pi_dim, pi_dim)
# local_witness_complex_feat = np.tile(local_witness_complex_feat_, (1, 3, 1, 1))  # (#nodes, 3, pi_dim, pi_dim) which can be fed into resnet
# local_witness_complex_feat = torch.FloatTensor(local_witness_complex_feat)

# python landmark_main.py --lm_perc 0.1 --seed 1 --method transformer --topo both --device cuda:0
# python landmark_main.py --lm_perc 0.15 --seed 1 --method transformer --topo both --device cuda:1
# python landmark_main.py --lm_perc 0.2 --seed 1 --method transformer --topo both --device cuda:2