import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import APPNP as APPNPConv
from torch.nn import Linear
import deeprobust.graph.utils as utils
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy
from itertools import product
from tqdm import tqdm
from vit_pytorch import ViT
import gudhi as gd
# from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import GCNConv

def topo_loss(PD, p, q):
    new_PD = PD[:-1,:]
    start, end = new_PD[:,0], new_PD[:,1]
    lengths = end - start
    means = (end+start)/2
    lengths = torch.FloatTensor(lengths)
    means = torch.FloatTensor(means)
    output = torch.sum(torch.mul(torch.pow(lengths, p), torch.pow(means, q)))
    return output

class CNN(nn.Module):
    def __init__(self, dim_hidden, dim_out):
        super(CNN, self).__init__()
        self.dim_out = dim_out
        self.features = nn.Sequential(
            nn.Conv2d(1, dim_hidden, kernel_size=2, stride=2), #channel of witness_complex_topo is 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(dim_hidden, dim_out, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, witness_complex_topo):
        feature = self.features(witness_complex_topo)
        feature = self.maxpool(feature)
        feature = feature.view(-1, self.dim_out) #B, dim_out - here B = 1
        return feature

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
    
class APPNP_WGTL(nn.Module):

    def __init__(self, nfeat, nhid, nclass, K=10, alpha=0.1, dropout=0.5, lr=0.01,\
                 alpha2 = 0.8, beta = 0.1, gamma = 0.1, lambda_coeff = 0.001, aggregation_method = 'attention',\
                with_bn=False, weight_decay=5e-4, with_bias=True, device=None):

        super(APPNP_WGTL, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device


        self.lin1 = Linear(nfeat, nhid)
        if with_bn:
            self.bn1 = nn.BatchNorm1d(nhid)
            self.bn2 = nn.BatchNorm1d(nclass)

        # self.lin2 = Linear(nhid, nclass)
        self.prop1 = APPNPConv(K, alpha)
        self.gc2 = GCNConv(nhid, nclass, bias=with_bias)

        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.with_bn = with_bn
        # GTL params
        self.agg = aggregation_method
        self.alpha2 = alpha2
        self.beta = beta
        self.gamma = gamma
        self.lambda_coeff = lambda_coeff
        self.gcnn = CNN(64, nhid)
        self.lcnn = ViT(
                image_size = 50,
                patch_size = 25,
                num_classes = nhid,
                dim = 32, #1024,
                depth = 6,
                heads = 16,
                mlp_dim = 64, #2048,
                dropout = 0.1,
                channels=1, # channel is 1
                emb_dropout = 0.1
                )
        self.attention = Attention(nhid)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        local_witness_comp_topo = self.lcnn(self.local_witness_complex_feat)
        global_witness_comp_topo = self.gcnn(self.global_witness_complex_feat)
        sim_matrix = cosine_similarity(X=local_witness_comp_topo.cpu().detach().numpy(), Y=local_witness_comp_topo.cpu().detach().numpy())
        skeleton_protein0 = gd.RipsComplex(
            distance_matrix=sim_matrix,
            max_edge_length=1.0
        )

        Rips_simplex_tree_protein0 = skeleton_protein0.create_simplex_tree(max_dimension=1)
        BarCodes_Rips0 = Rips_simplex_tree_protein0.persistence()
        local_witness_comp_topo_pd = Rips_simplex_tree_protein0.persistence_intervals_in_dimension(0)
        loss_topo = topo_loss(PD = local_witness_comp_topo_pd, p = 2, q= 1)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        if self.with_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.lin2(x)
        # if self.with_bn:
        #     x = self.bn2(x)
        x = self.prop1(x, edge_index, edge_weight=None)

        # x and witness_comp_topo have the same size
        if self.agg == 'einsum':
            global_witness_comp_topo = global_witness_comp_topo.view(global_witness_comp_topo.size(1))
            x = torch.einsum('nb, b-> nb', x, global_witness_comp_topo)
            x = torch.einsum('nb, nb-> nb', x, local_witness_comp_topo)
        elif self.agg == 'weighted_sum':
            x = self.alpha2 * x + self.beta * global_witness_comp_topo + self.gamma * local_witness_comp_topo
        elif self.agg == 'attention':
            global_witness_comp_topo = global_witness_comp_topo.view(global_witness_comp_topo.size(1))
            emb = torch.stack([x, local_witness_comp_topo], dim=1)
            emb, att = self.attention(emb)
            x = torch.einsum('nb, b-> nb', emb, global_witness_comp_topo)

        x = self.gc2(x, edge_index)
        return F.log_softmax(x, dim=1), loss_topo


    def initialize(self):
        self.lin1.reset_parameters()
        # self.lin2.reset_parameters()
        self.gc2.reset_parameters()
        self.prop1.reset_parameters()
        if self.with_bn:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()
    # ----
    def fit(self, pyg_data, global_witness_complex_feat, local_witness_complex_feat, train_iters=200, initialize=True, verbose=False, patience=500,**kwargs):
        """Train the ChebNet model, when idx_val is not None, pick the best model
        according to the validation loss.

        Parameters
        ----------
        pyg_data :
            pytorch geometric dataset object
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        if initialize:
            self.initialize()

        self.data = pyg_data.to(self.device)
        self.global_witness_complex_feat = global_witness_complex_feat.to(self.device)
        self.local_witness_complex_feat = local_witness_complex_feat.to(self.device)
        # By default, it is trained with early stopping on validation
        self.train_with_early_stopping(train_iters, patience, verbose)

    def train_with_early_stopping(self, train_iters, patience, verbose):
        """early stopping based on the validation loss
        """
        if verbose:
            print('=== training APPNP+WGTL model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output,loss_topo = self.forward(self.data)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask])+ self.lambda_coeff * loss_topo
            loss_train.backward()
            optimizer.step()

            if verbose and i % 20 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output,_ = self.forward(self.data)
            loss_val = F.nll_loss(output[val_mask], labels[val_mask])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self):
        """Evaluate ChebNet performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        test_mask = self.data.test_mask
        labels = self.data.y
        output,_ = self.forward(self.data)
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def predict(self):
        """
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of ChebNet
        """

        self.eval()
        return self.forward(self.data)


# if __name__ == "__main__":
#     from deeprobust.graph.data import Dataset, Dpr2Pyg
#     data = Dataset(root='/tmp/', name='cora', setting='gcn')
#     adj, features, labels = data.adj, data.features, data.labels
#     idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
#     model = GCN(nfeat=features.shape[1],
#           nhid=16,
#           nclass=labels.max().item() + 1,
#           dropout=0.5, device='cuda')
#     model = model.to('cuda')
#     pyg_data = Dpr2Pyg(data)[0]

#     import ipdb
#     ipdb.set_trace()

#     model.fit(pyg_data, verbose=True) # train with earlystopping
#     model.test()
#     print(model.predict())
