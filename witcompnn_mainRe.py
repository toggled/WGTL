import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph.defense import GCN, GWitCompNN, LWitCompNN_V1, LWitCompNN_V2, LWitCompNN_V3, CWitCompNN_V1, CWitCompNN_V2, CWitCompNN_V3, SimPGCNWGTL, ChebNetWGTL, SGCWGTL, SAGEWGTL, GNNGuardWGTL, GATWGTL
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
parser.add_argument('--dataset', type=str, default='cora', choices=['cora','citeseer','polblogs','pubmed',\
                                'snap-patents','fb100','actor','roman_empire'], help='dataset')
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
parser.add_argument('--topo', type=str,default = 'both',help='local/global/both')
parser.add_argument('--backbone', type=str,default = 'GCN',help='GCN/H2GCN/SIMPGCN/Chebnet/APPNP/SAGE/GNNGuard/GAT')
parser.add_argument('--method',type=str,default='transformer',help='transformer/resnet/cnn')
parser.add_argument('--device', type=str,default = 'cuda:0',help='cuda:0/cuda:1/...')
parser.add_argument('--type', type=str,default = 'meta',help='meta/net/metapgd/pgd')
parser.add_argument('--colab',type=bool,default = False)
parser.add_argument('--topo_unpurturbed',type=bool,default = False)


args = parser.parse_args()
args.cuda = torch.cuda.is_available()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
# load original data and attacked data
if args.type == 'meta':
    prefix = ["data/",'/content/drive/MyDrive/WitnesscomplexGNNs/data/'][args.colab]
    attack = 'meta'
if args.type=='net':
    prefix = ["data/nettack/",'/content/drive/MyDrive/WitnesscomplexGNNs/data/nettack/'][args.colab]
    attack='nettack' 
if args.dataset in ['fb100','snap-patents','actor','roman_empire']: # Heterophily dataset:
    from deeprobust.graph.data import CustomDataset 
    prefix2 = ['data/','/content/drive/MyDrive/WitnesscomplexGNNs/data/'][args.colab]
    data = CustomDataset(root=prefix2+args.dataset, name=args.dataset, setting='nettack')
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    features, labels = data.features, data.labels
else:
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
    witness_complex_feat = torch.FloatTensor(np.load(prefix + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_PI' + '.npz', allow_pickle=True)['arr_0'])
    print('shape of global PI representation: ',witness_complex_feat.shape)

elif args.topo == 'local': # Load Local PI computed on the perturbed Adj matrix. LWTL
    # local_witness_complex_feat => Shape (#nodes x 50 x 50)
    tmp = np.load(prefix + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_localPI' + '.npz', allow_pickle=True)['arr_0']
    # print('tmp.shape: ',tmp.shape)
    # local_witness_complex_feat_ = np.random.rand(2485, 50, 50) # torch.FloatTensor(np.load('data/' + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_localPI' + '.npz', allow_pickle=True)['arr_0'])
    local_witness_complex_feat_ = np.expand_dims(tmp, axis=1)  # (#nodes, 1, pi_dim, pi_dim)
    local_witness_complex_feat = torch.FloatTensor(local_witness_complex_feat_)
    print('shape of local PI representation: ',local_witness_complex_feat.shape) # (#nodes, 1, pi_dim, pi_dim)
else: # GWTL + LWTL + Topoloss
    # for both
    # Loading topological features computed on the unpurturbed graph===>
    if args.topo_unpurturbed:
        print('Loading topo. features computed on the Unpurturbed graph===>')
        global_witness_complex_feat = torch.FloatTensor(np.load(prefix + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_PI' + '.npz', allow_pickle=True)['arr_0']) # torch.FloatTensor(np.random.rand(1, 50, 50))
        local_witness_complex_feat_ = np.load(prefix + args.dataset + '/' + args.dataset + '_' + str(args.ptb_rate) + '_localPI' + '.npz', allow_pickle=True)['arr_0']
    else:
        # Loading topological features computed on the purturbed graph===>
        print('Loading topo. features computed on the purturbed graph===>')
        if args.ptb_rate == 0:
            global_witness_complex_feat = torch.FloatTensor(np.load(prefix + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_PI' + '.npz', allow_pickle=True)['arr_0']) # torch.FloatTensor(np.random.rand(1, 50, 50))
            local_witness_complex_feat_ = np.load(prefix + args.dataset + '/' + args.dataset + '_' + str(args.ptb_rate) + '_localPI' + '.npz', allow_pickle=True)['arr_0']
        else:

            global_witness_complex_feat = torch.FloatTensor(np.load(prefix + args.dataset + '/' + args.dataset + '_'+str(args.ptb_rate)+'_PInew' + '.npz', allow_pickle=True)['arr_0']) # torch.FloatTensor(np.random.rand(1, 50, 50))
            local_witness_complex_feat_ = np.load(prefix + args.dataset + '/' + args.dataset + '_' + str(args.ptb_rate) + '_localPInew' + '.npz', allow_pickle=True)['arr_0']
    local_witness_complex_feat_ = np.expand_dims(local_witness_complex_feat_, axis=1)  # (#nodes, 1, pi_dim, pi_dim)
    local_witness_complex_feat = torch.FloatTensor(local_witness_complex_feat_)

    witness_complex_feats = [global_witness_complex_feat, local_witness_complex_feat]
    print('shapes of PIs representation: ', global_witness_complex_feat.shape, local_witness_complex_feat.shape)


topo_type = args.topo # 'local', 'global'
method = args.method
aggregation_method = 'attention' # einsum, weighted_sum, attention
if args.backbone == 'GCN':
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
            model.fit(features, perturbed_adj, witness_complex_feats[0], witness_complex_feats[1], labels, idx_train, idx_val, train_iters=args.epoch, verbose=True, patience = 100)
    # # using validation to pick model
    # model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
    model.eval()
    # You can use the inner function of model to test
    acc = model.test(idx_test)
elif args.backbone == 'SAGE':
    print("You are using Transformer on global&local features now!")
    model = SAGEWGTL(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, gamma = args.gamma, lambda_coeff = args.lambda_coeff, aggregation_method = aggregation_method)
    model = model.to(device)
    model.fit(features, perturbed_adj, witness_complex_feats[0], witness_complex_feats[1], labels, idx_train, idx_val, train_iters=args.epoch, verbose=True)
    model.eval()
    acc = model.test(idx_test)
elif args.backbone == "GAT":
    print('GAT')
    model = GATWGTL(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, device=device, alpha = args.alpha, beta = args.beta, gamma = args.gamma, lambda_coeff = args.lambda_coeff, aggregation_method = aggregation_method)
    model = model.to(device)
    model.fit(features, perturbed_adj, witness_complex_feats[0], witness_complex_feats[1], labels, idx_train, idx_val, train_iters=args.epoch, verbose=True, patience=100)
    model.eval()
    acc = model.test(idx_test)

elif args.backbone == 'H2GCN': #Only implemented for topo_type=both and method=transformer
    from torch_geometric.data import Data
    assert method=='transformer'
    assert topo_type=='both'
    from deeprobust.graph.defense_pyg import H2GCNWGTL 
    hidden_channels = 64
    num_layers = 2
    num_mlp_layers = 1
    nfeat=features.shape[1]
    nhid=args.nhid
    nclass=int(labels.max()) + 1
    
    # model = H2GCNWGTL(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, \
    #                   dropout=args.drop_rate, lr=0.01, weight_decay=args.weight_decay, \
    #                 device=device, alpha = args.alpha, beta = args.beta, gamma = args.gamma, \
    #                 lambda_coeff = args.lambda_coeff, aggregation_method = aggregation_method,\
    #             in_channels=features.shape[1], hidden_channels=hidden_channels, out_channels=int(labels.max()) + 1,\
    #             num_nodes=features.shape[0], adj=perturbed_adj,\
    #             num_layers=num_layers,num_mlp_layers=num_mlp_layers)
    model = H2GCNWGTL(nfeat=nfeat, nhid=nhid, nclass=nclass, \
                in_channels=nfeat, hidden_channels=hidden_channels, out_channels=nclass, \
                num_nodes=features.shape[0], adj=perturbed_adj,\
                dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, \
                # with_relu=True, with_bias=True, \
                alpha = args.alpha, beta = args.beta, gamma = args.gamma, lambda_coeff = args.lambda_coeff, \
                aggregation_method = aggregation_method, \
                num_layers=num_layers,  num_mlp_layers=num_mlp_layers,\
                use_bn=True, conv_dropout=True, device = args.device)
    model = model.to(device)

    if sp.issparse(features):
        features = torch.FloatTensor(np.array(features.todense()))
    else:
        features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    idx_train0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_train0[idx_train] = 1
    idx_val0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_val0[idx_val] = 1
    idx_test0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_test0[idx_test] = 1
    data = Data(x=features, y=labels,\
                train_mask=idx_train0, val_mask = idx_val0, test_mask = idx_test0)
    data = data.to(device)

    model.fit(data=data, global_witness_complex_feat=witness_complex_feats[0], \
              local_witness_complex_feat=witness_complex_feats[1],train_iters=args.epoch, verbose=True)

    # # using validation to pick model
    # model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)
    model.eval()
    # You can use the inner function of model to test
    acc = model.test()
elif args.backbone == 'SIMPGCN':
    model = SimPGCNWGTL(nnodes = features.shape[0], nfeat=features.shape[1], nhid=16, nclass=int(labels.max()) + 1, \
                        dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, \
                        seed = args.seed, device=device, \
                        alpha = args.alpha, beta = args.beta, gamma2 = args.gamma, \
                        lambda_coeff = 3, aggregation_method = aggregation_method)
    model = model.to(device)
    model.fit(features, perturbed_adj, witness_complex_feats[0], witness_complex_feats[1], labels, idx_train, idx_val, train_iters=args.epoch, verbose=True)
    model.eval()
    # You can use the inner function of model to test
    acc = model.test(idx_test)
    # os.system('rm -r saved_knn'+str(args.seed))
elif args.backbone == 'Chebnet': # nhid=args.nhid, 
    model = ChebNetWGTL(nfeat=features.shape[1], nhid=32, num_hops=3, \
                    nclass=int(labels.max()) + 1, \
                        dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, \
                        device=device, alpha = args.alpha, beta = args.beta, gamma = args.gamma, \
                        lambda_coeff = args.lambda_coeff, aggregation_method = aggregation_method)
    model = model.to(device)
    features = torch.FloatTensor(features.todense()).float()
    labels = torch.LongTensor(labels)
    idx_train0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_train0[idx_train] = 1
    idx_val0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_val0[idx_val] = 1
    idx_test0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_test0[idx_test] = 1
    from torch_geometric.data import Data
    data = Data(x=features,edge_index = torch.LongTensor(np.array(perturbed_adj.nonzero())),\
                        y=labels,train_mask=idx_train0, val_mask = idx_val0, test_mask = idx_test0)

    model.fit(data, witness_complex_feats[0], witness_complex_feats[1], train_iters=args.epoch, patience = 100, verbose=True)
    # model.eval()
    # You can use the inner function of model to test
    acc = model.test()
elif args.backbone == 'SGC': # nhid=args.nhid,    
    model = SGCWGTL(nfeat=features.shape[1], nhid=64, K=2, \
                    nclass=int(labels.max()) + 1, \
                        lr=args.lr, weight_decay=args.weight_decay, \
                        device=device, alpha = args.alpha, beta = args.beta, gamma = args.gamma, \
                        lambda_coeff = args.lambda_coeff, aggregation_method = aggregation_method)
    model = model.to(device)
    features = torch.FloatTensor(features.todense()).float()
    labels = torch.LongTensor(labels)
    idx_train0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_train0[idx_train] = 1
    idx_val0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_val0[idx_val] = 1
    idx_test0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_test0[idx_test] = 1
    from torch_geometric.data import Data
    data = Data(x=features,edge_index = torch.LongTensor(np.array(perturbed_adj.nonzero())),\
                        y=labels,train_mask=idx_train0, val_mask = idx_val0, test_mask = idx_test0)

    model.fit(data, witness_complex_feats[0], witness_complex_feats[1], train_iters=args.epoch, patience = 100, verbose=True)
    # model.eval()
    # You can use the inner function of model to test
    acc = model.test()
elif args.backbone == 'APPNP': # nhid=args.nhid,    
    from deeprobust.graph.defense_pyg import APPNP_WGTL
    a = 0.1 # 0.1
    model = APPNP_WGTL(nfeat=features.shape[1], nhid=64, nclass=int(labels.max())+1, K=10, alpha=a, with_bias=True, with_bn=False, \
                        dropout=args.drop_rate, lr=args.lr, weight_decay=args.weight_decay, \
                        device=device, alpha2 = args.alpha, beta = args.beta, gamma = args.gamma, \
                        lambda_coeff = args.lambda_coeff, aggregation_method = aggregation_method)
    model = model.to(device)
    features = torch.FloatTensor(features.todense()).float()
    labels = torch.LongTensor(labels)
    idx_train0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_train0[idx_train] = 1
    idx_val0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_val0[idx_val] = 1
    idx_test0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_test0[idx_test] = 1
    from torch_geometric.data import Data
    data = Data(x=features,edge_index = torch.LongTensor(np.array(perturbed_adj.nonzero())),\
                        y=labels,train_mask=idx_train0, val_mask = idx_val0, test_mask = idx_test0)

    model.fit(data, witness_complex_feats[0], witness_complex_feats[1], train_iters=args.epoch, patience = 100, verbose=True)
    # model.eval()
    # You can use the inner function of model to test
    acc = model.test()
elif args.backbone == 'GNNGuard':
    model = GNNGuardWGTL(nfeat=features.shape[1], nhid=args.nhid, nclass=int(labels.max()) + 1, 
                        lr=args.lr, weight_decay=args.weight_decay, dropout=args.drop_rate, 
                        device=device, alpha = args.alpha, beta = args.beta, gamma = args.gamma, \
                        lambda_coeff = args.lambda_coeff, aggregation_method = aggregation_method)
    model = model.to(device)
    model.fit(features, perturbed_adj, witness_complex_feats[0], witness_complex_feats[1], labels, idx_train, idx_val, train_iters=args.epoch, verbose=True, patience=100)
    model.eval()
    # print('gnnguard test')
    # You can use the inner function of model to test
    acc = model.test(idx_test)

output = {'seed':args.seed,'acc':acc}
csv_name = [args.backbone+"/",'/content/drive/MyDrive/WitnesscomplexGNNs/'+args.backbone+"/"][args.colab]
csv_name += args.backbone+'_'+aggregation_method+"_"+args.topo+'_'+args.method+'_'+args.dataset + "_" + str(args.ptb_rate) + '.'+attack+'.csv'
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