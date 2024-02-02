import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
# from torch_geometric.nn import SAGEConv, GATConv, APPNP, MessagePassing
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import scipy.sparse
import numpy as np
# from .base_model import BaseModel
import torch.optim as optim
from deeprobust.graph import utils

class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    # def forward(self, x, adj_t, adj_t2, global_witness_comp_topo):
    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        # print(global_witness_comp_topo.shape,' => ',x.shape,' -- ',x2.shape)
        # x3 = (global_witness_comp_topo*x).squeeze()
        # x3 = global_witness_comp_topo.squeeze()
        # print('x3.shape: ',x3.shape)
        # return torch.cat([x1, x2, x3], dim=1)
        return torch.cat([x1, x2], dim=1)
    
class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,\
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.x
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    
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
    
class H2GCNWGTL(nn.Module):
    """ our implementation """
    def __init__(self, 
                 nfeat, nhid, nclass, \
                in_channels, hidden_channels, out_channels, num_nodes, adj,\
                dropout=0.5, lr=0.01, weight_decay=5e-4,
                # with_relu=True, with_bias=True, \
                alpha = 0.8, beta = 0.1, gamma = 0.1, lambda_coeff = 0.001, \
                aggregation_method = 'attention', 
                num_layers=2,  num_mlp_layers=1,\
                use_bn=True, conv_dropout=True, device = 'cuda:0'):
        super(H2GCNWGTL, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        # self.gcnn = CNN(64, nhid)
        self.agg = aggregation_method
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_coeff = lambda_coeff
        self.hidden_channels = hidden_channels
        # Define a linear layer for the projection
        self.global_linear_layer = torch.nn.Linear(50 * 50, num_nodes * hidden_channels)
        # self.local_linear_layer = torch.nn.Linear(50 * 50, hidden_channels)
        ##
        self.feature_embed = MLP(in_channels, hidden_channels,\
                hidden_channels, num_layers=num_mlp_layers, dropout=dropout)

        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        # bnsz = hidden_channels*3*len(self.convs)
        bnsz = hidden_channels*2*len(self.convs)
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d( bnsz) )
        # self.bns.append(nn.BatchNorm1d(hidden_channels*3*len(self.convs) ) )

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(bnsz ) )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        # last_dim = hidden_channels*(2**(num_layers+1)-1)
        # last_dim = hidden_channels*(2**(num_layers+2)-(num_layers+1)-2)
        last_dim = hidden_channels*(2**(num_layers+1))
        # print('last_dim: ',last_dim,hidden_channels)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.device = device 
        self.lr = lr 
        self.weight_decay = weight_decay
        self.init_adj(adj)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.global_linear_layer.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def init_adj(self, adj_t):
    # def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        
        # if isinstance(edge_index, SparseTensor):
        #     dev = edge_index.device
        #     adj_t = edge_index
        #     adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        #     adj_t[adj_t > 0] = 1
        #     adj_t[adj_t < 0] = 0
        #     adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        # elif isinstance(edge_index, torch.Tensor):
        #     row, col = edge_index
        #     adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))
        adj_t[adj_t > 0] = 1
        adj_t[adj_t < 0] = 0
        adj_t = SparseTensor.from_scipy(adj_t).to(self.device)
        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        
        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(self.device)
        self.adj_t2 = adj_t2.to(self.device)



    def forward(self, data):
        x = data.x

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        x = self.feature_embed(data)
        # print('x.shape: ',x.shape)
        global_witness_comp_topo = self.global_linear_layer(self.global_witness_complex_feat.view(1,-1))
        # global_witness_comp_topo = F.adaptive_avg_pool2d(self.global_witness_complex_feat, (1, 1))
        # Reshape to the desired shape
        global_witness_comp_topo = global_witness_comp_topo.view(1, -1, self.hidden_channels)

        # Reshape the tensor to have a single dimension for the spatial features
        # reshaped_tensor = self.local_witness_complex_feat.view(self.num_nodes, -1)

        # local_witness_comp_topo = self.local_linear_layer(reshaped_tensor)
        # print('global_witness_comp_topo.shape: ',global_witness_comp_topo.shape)
        # print('self.local_witness_complex_feat.shape: ',self.local_witness_complex_feat.shape)        
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            # x = conv(x, adj_t, adj_t2, global_witness_comp_topo) 
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x, adj_t, adj_t2, global_witness_comp_topo)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)
        # print([xxx.shape for xxx in xs])
        x = self.jump(xs)
        # print('after jk: ',x.shape)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        # print(x.shape,' ',global_witness_comp_topo.shape)
        x = torch.cat([x,global_witness_comp_topo.squeeze()],dim=1) # ----
        x = self.final_project(x)
        x = F.log_softmax(x, dim=1)
        return x

    # def fit(self,data, train_iters = 200, verbose = False):
    def fit(self,data, global_witness_complex_feat, local_witness_complex_feat, \
            train_iters=200, initialize=True, verbose=True, normalize=True, patience=100, **kwargs):
        if initialize:
            self.reset_parameters()
        self.data = data 
        self.training = True 
        # adj = adj.to(self.device)
        # if normalize:
        #     if utils.is_sparse_tensor(adj):
        #         adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        #     else:
        #         adj_norm = utils.normalize_adj_tensor(adj)
        # else:
        #     adj_norm = adj
        # self.adj_norm = adj_norm
        self.global_witness_complex_feat = global_witness_complex_feat.to(self.device)
        # self.local_witness_complex_feat = local_witness_complex_feat.to(self.device)

        # self.train_without_early_stopping(train_iters,verbose)
        self.train_with_early_stopping(train_iters=train_iters,patience = patience, verbose=verbose)
    
    # def train_without_early_stopping(self, train_iters, verbose):
    #     """early stopping based on the validation loss
    #     """
    #     if verbose:
    #         print(f'=== training H2GCN model ===')
    #     optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    #     labels = self.data.y
    #     train_mask = self.data.train_mask
    #     self.reset_parameters()
    #     for i in range(train_iters):
    #         self.train()
    #         optimizer.zero_grad()

    #         output = self.forward(self.data)

    #         loss_train = F.nll_loss(output[train_mask], labels[train_mask])
    #         loss_train.backward()
    #         optimizer.step()

    #         if verbose and (i % 50 == 0 or i==train_iters-1):
    #             print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

    #     self.eval()
    #     output = self.forward(self.data)
    #     self.output = output
    
    def train_with_early_stopping(self, train_iters=200, patience=100, verbose=True):
        """early stopping based on the validation loss
        """
        if verbose:
            print(f'=== training H2GCN+WGTL model (with early stopping) ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100
        best_acc_val = 0
        best_epoch = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            # output,loss_topo = self.forward(self.data) ######
            output = self.forward(self.data) ######

            # loss_train = F.nll_loss(output[train_mask], labels[train_mask]) + self.lambda_coeff * loss_topo
            loss_train = F.nll_loss(output[train_mask], labels[train_mask]) 
            loss_train.backward()
            optimizer.step()

            if verbose and (i % 50 == 0 or i==train_iters-1):
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            # output,_ = self.forward(self.data)
            output = self.forward(self.data)
            # loss_val = F.nll_loss(output[val_mask], labels[val_mask])
            acc_val = utils.accuracy(output[val_mask], labels[val_mask])
            # print(acc)

            # if best_loss_val > loss_val:
            #     best_loss_val = loss_val
            #     self.output = output
            #     weights = deepcopy(self.state_dict())
            #     patience = early_stopping
            #     best_epoch = i
            # else:
            #     patience -= 1

            if best_acc_val < acc_val:
                best_acc_val = acc_val
                self.output = output
                # weights = deepcopy(self.state_dict())
                patience = early_stopping
                best_epoch = i
            else:
                patience -= 1

            if i > early_stopping and patience <= 0:
                break

        if verbose:
             # print('=== early stopping at {0}, loss_val = {1} ==='.format(best_epoch, best_loss_val) )
             print('=== early stopping at {0}, acc_val = {1} ==='.format(best_epoch, best_acc_val) )
        # self.load_state_dict(weights)

    def test(self):
        """Evaluate model performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        # self.eval()
        self.training = False 
        test_mask = self.data.test_mask
        labels = self.data.y
        # output,_ = self.forward(self.data) ##########
        output = self.forward(self.data) ##########
        # output = self.output
        loss_test = F.nll_loss(output[test_mask], labels[test_mask])
        acc_test = utils.accuracy(output[test_mask], labels[test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


if __name__=='__main__':
    from deeprobust.graph.data import Dataset
    from torch_geometric.data import Data
    data = Dataset(root='/tmp/', name='citeseer', setting='prognn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    c = int(labels.max()) + 1
    d = features.shape[1]
    n = features.shape[0]
    device = 'cuda:0'
    features = torch.FloatTensor(features.todense()).float()
    labels = torch.LongTensor(labels)
    idx_train0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_train0[idx_train] = 1
    idx_val0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_val0[idx_val] = 1
    idx_test0 = torch.zeros((features.shape[0], ), dtype=torch.bool)
    idx_test0[idx_test] = 1
    data = Data(x=features, y=labels,train_mask=idx_train0, val_mask = idx_val0, test_mask = idx_test0)
    data = data.to(device)
    hidden_channels = 64
    num_layers = 2
    num_mlp_layers = 1
    model = H2GCNWGTL(d, hidden_channels, c, n, SparseTensor.from_scipy(adj), \
                #   dataset.graph['edge_index'], dataset.graph['num_nodes'],
                        num_layers=num_layers, dropout=0.5,lr=0.01,weight_decay=5e-4, \
                        num_mlp_layers=num_mlp_layers)
    model = model.to(device)
    # pyg_data = Dpr2Pyg(data)[0]
    model.fit(data, train_iters=200, verbose=True)
    model.eval()
    model.test()
