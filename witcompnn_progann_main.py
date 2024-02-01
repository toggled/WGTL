import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCN, ProGNN, GWitCompNN, LWitCompNN_V1, LWitCompNN_V2, LWitCompNN_V3, CWitCompNN_V1, CWitCompNN_V2, CWitCompNN_V3
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import scipy.sparse as sp
import argparse
from utils import load_npz
#import gudhi as gd
import json

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora'], help='dataset')
parser.add_argument('--lr', type=float, default=0.001,  help='learning rate')
parser.add_argument('--drop_rate', type=float, default=0.5,  help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=5e-4,  help='weight decay rate')
parser.add_argument('--ptb_rate', type=float, default=0.2,  help='pertubation rate')
parser.add_argument('--epoch', type=float, default=200,  help='epochs')
parser.add_argument('--alpha', type=float, default=0.8,  help='alpha')
parser.add_argument('--beta', type=float, default=0.1,  help='beta')
parser.add_argument('--gamma', type=float, default=0.1,  help='gamma')
parser.add_argument('--lambda_coeff', type=float, default=0.01,  help='lambda_coeff')
parser.add_argument('--nhid', type=int, default=128,  help='nhid')
parser.add_argument('--topo', type=str,default = 'witptb_both',help='witorig/witptb/witptb_local/vrorig/vrptb')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load original data and attacked data
adj, features, labels = load_npz('data/' + args.dataset + '/' + args.dataset + '.npz')
f = open('data/' + args.dataset + '/' + args.dataset + '_prognn_splits.json')
idx = json.load(f)
idx_train, idx_val, idx_test = np.array(idx['idx_train']), np.array(idx['idx_val']), np.array(idx['idx_test'])
perturbed_adj = sp.load_npz('data/' + args.dataset + '/' + args.dataset + '_meta_adj_'+str(args.ptb_rate)+'.npz')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Setup WitCompNN Model        
# load witness complex topological features
if args.topo == 'witorig':
    ori_witness_complex_feat = torch.FloatTensor(np.load('data/' + args.dataset + '/' + args.dataset + '_PI' + '.npz', allow_pickle=True)['arr_0'])

if args.topo == 'witptb':
    global_witness_complex_feat = torch.FloatTensor(np.load('data/' + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_PI' + '.npz', allow_pickle=True)['arr_0'])
    print('shape of global PI representation: ',global_witness_complex_feat.shape)

if args.topo == 'witptb_local':
    # local_witness_complex_feat => Shape (#nodes x 50 x 50)
    local_witness_complex_feat_ = np.load('data/' + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_localPI' + '.npz', allow_pickle=True)['arr_0']
    local_witness_complex_feat_ = np.expand_dims(local_witness_complex_feat_, axis=1)  # (#nodes, 1, pi_dim, pi_dim)
    local_witness_complex_feat = torch.FloatTensor(local_witness_complex_feat_)
    print('shape of local PI representation: ',local_witness_complex_feat.shape) # (#nodes, 1, pi_dim, pi_dim)

if args.topo == 'witptb_both':
    global_witness_complex_feat = torch.FloatTensor(np.load('data/' + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_PI' + '.npz', allow_pickle=True)['arr_0'])
    local_witness_complex_feat_ = np.load('data/' + args.dataset + '/' + args.dataset + '_' + str(args.ptb_rate) + '_localPI' + '.npz', allow_pickle=True)['arr_0']
    local_witness_complex_feat_ = np.expand_dims(local_witness_complex_feat_, axis=1)  # (#nodes, 1, pi_dim, pi_dim)
    local_witness_complex_feat = torch.FloatTensor(local_witness_complex_feat_)

    witness_complex_feats = [global_witness_complex_feat, local_witness_complex_feat]
    print('shapes of PIs representation: ', global_witness_complex_feat.shape, local_witness_complex_feat.shape)



topo_type = 'both' # 'local', 'global', 'both'
method = 'transformer' # resnet, cnn, transformer
aggregation_method = 'attention' # einsum, weighted_sum, attention


if topo_type == 'global':
    # for global only
    model = GWitCompNN(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate,  lr=args.lr, weight_decay=args.weight_decay, device=device)
    model = model.to(device)
    model.fit(features, perturbed_adj, witness_complex_feat, labels, idx_train, train_iters=args.epoch, verbose=True)
elif topo_type == 'local':
    # for local only
    if method == 'resnet':
        print("You are using ResNet on local features now!")
        model = LWitCompNN_V1(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, aggregation_method = aggregation_method)
        model = model.to(device)
        model.fit(features, perturbed_adj, local_witness_complex_feat, labels, idx_train, idx_val, train_iters=args.epoch,  verbose=True)
    elif method == 'cnn':
        print("You are using CNN on local features now!")
        model = LWitCompNN_V2(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, aggregation_method = aggregation_method)
        model = model.to(device)
        model.fit(features, perturbed_adj, local_witness_complex_feat, labels, idx_train, idx_val, train_iters=args.epoch, verbose=True)
    elif method == 'transformer':
        print("You are using Transformer on local features now!")
        model = LWitCompNN_V3(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, aggregation_method = aggregation_method)
        model = model.to(device)
        model.fit(features, perturbed_adj, local_witness_complex_feat, labels, idx_train, idx_val, train_iters=args.epoch, verbose=True)
elif topo_type == 'both':
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

model = GCN(nfeat=features.shape[1],
            nhid=args.nhid,
            nclass=labels.max().item() + 1,
            dropout=0.5, device=device)
prognn = ProGNN(model, args, device)
prognn.fit(features, perturbed_adj, labels, idx_train, idx_val)

# # using validation to pick model
# model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
model.eval()
# You can use the inner function of model to test
model.test(idx_test)

