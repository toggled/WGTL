### Code to generate PRBCD attack
import torch, os
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset,CustomDataset
import argparse
from deeprobust.graph.data import Pyg2Dpr
from deeprobust.graph.global_attack import PRBCD

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', 'fb100','snap-patents','actor','roman_empire','ogb-arxiv'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--device',type=str,default='cuda:0',help='device')
args = parser.parse_args()

device=args.device
prefix = 'data/ogb-arxiv/'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

def get_ogbnarxiv():
    from ogb.nodeproppred import PygNodePropPredDataset
    import torch_geometric.transforms as T
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv')
    n_classes = dataset.num_classes
    # dataset.transform = T.NormalizeFeatures()
    transform = T.Compose([T.ToUndirected(), T.NormalizeFeatures()])
    data = transform(dataset[0])
    # print(data)
    features, labels = data.x, data.y
    # nclass = max(labels).item()+1
    split_idx = dataset.get_idx_split() 
    idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
    labels = torch.LongTensor(labels.squeeze(1))
    idx_train0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_train0[idx_train] = 1
    idx_val0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_val0[idx_val] = 1
    idx_test0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_test0[idx_test] = 1
    data.train_mask = idx_train0
    data.val_mask = idx_val0
    data.test_mask = idx_test0
    data.y = labels 
    return data, n_classes

if args.dataset in ['fb100','snap-patents','actor','roman_empire']:
    data = CustomDataset(root='data/'+args.dataset, name=args.dataset, setting='nettack')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    print('#Edges: ',(adj.sum()//2))
    print('#Nodes: ',adj.shape)
    perturbations = int(args.ptb_rate * (adj.sum()//2))
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    print(adj.shape,' ',features.shape,' ',labels.shape)
    print('defining surrogate')
    # Setup Surrogate Model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
            dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)
    print('uploading surrogate to device')
    surrogate = surrogate.to(device)
    print('fit surrogate')
    surrogate.fit(features, adj, labels, idx_train)

elif args.dataset in ['ogb-arxiv']:
    from deeprobust.graph.defense_pyg import GCN
    data,n_classes = get_ogbnarxiv()
    surrogate = GCN(nfeat=data.x.shape[1], nhid=256, nclass=n_classes,
            dropout=0, lr=args.lr, weight_decay=1e-3,nlayers=3,
            device=device).to(device)
    surrogate.initialize()

    print('fit surrogate')
    surrogate.data = data.to(device)
    surrogate.train_with_early_stopping(train_iters=3000, patience=300, verbose=True)
    surrogate.test()
    print('done')
else:
    data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    print('#Edges: ',(adj.sum()//2))
    print('#Nodes: ',adj.shape)
    perturbations = int(args.ptb_rate * (adj.sum()//2))
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    print(adj.shape,' ',features.shape,' ',labels.shape)
    print('defining surrogate')
    # Setup Surrogate Model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
            dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)
    print('uploading surrogate to device')
    surrogate = surrogate.to(device)
    print('fit surrogate')
    surrogate.fit(features, adj, labels, idx_train) # train_iters = 500



if args.dataset == 'ogb-arxiv':
    model = PRBCD(data.to(device), model=surrogate, device=device, epochs=500)
else:
    from torch_geometric.data import Data
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
    model = PRBCD(data.to(device), model=surrogate, device=device, epochs=500)


def main():
    edge_index, edge_weight = model.attack(ptb_rate=args.ptb_rate)
    print('========GCN on perturbed graph======')
    data.edge_index = edge_index 
    poisoned_model = GCN(nfeat=data.x.shape[1], nhid=256, nclass=n_classes,
        dropout=0, lr=args.lr, weight_decay=1e-3,nlayers=3,
        device=device).to(device)
    poisoned_model.initialize()
    poisoned_model.data = data.to(device)
    poisoned_model.train_with_early_stopping(train_iters=3000, patience=300, verbose=True)
    poisoned_model.test()
    os.system('mkdir -p '+prefix)
    torch.save(data,prefix+args.dataset+'_prbcd_pygdata_'+str(args.ptb_rate)+'.pt')

if __name__ == '__main__':
    main()

# python prbcd_attack.py --dataset ogb-arxiv --ptb_rate 0.1 --seed 0
