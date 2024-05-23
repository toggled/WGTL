from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
def get_ogbarxiv():
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv')
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
    return data 
