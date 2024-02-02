import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
import torchvision.models as models
import numpy as np
from vit_pytorch import ViT
import gudhi as gd
#from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from torch.nn import Sequential, Linear, ReLU
from .gcn_conv import GCNConv

# https://github.com/lucidrains/vit-pytorch#vision-transformer---pytorch
# pip install vit-pytorch

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

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = 0.5 #dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0.5 #alpha
        self.concat = True

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        h = h.to_dense()
        adj = adj.to_dense()
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GCN_GNNGuard(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, drop=False, weight_decay=5e-4, n_edge=1,
                 with_relu=True,
                 with_bias=True, device=None):

        super(GCN_GNNGuard, self).__init__()

        #assert device is not None, "Please specify 'device'!"
        self.device = 'cpu' #device

        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr

        weight_decay = 0  # set weight_decay as 0

        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.n_edge = n_edge
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.gate = Parameter(torch.rand(1))  # creat a generator between [0,1]
        self.test_value = Parameter(torch.rand(1))
        self.drop_learn_1 = Linear(2, 1)
        self.drop_learn_2 = Linear(2, 1)
        self.drop = drop
        self.bn1 = torch.nn.BatchNorm1d(nhid)
        self.bn2 = torch.nn.BatchNorm1d(nhid)
        nclass = int(nclass)

        """GCN from geometric"""
        """network from torch-geometric, """
        self.gc1 = GCNConv(nfeat, nhid, bias=True, )
        self.gc2 = GCNConv(nhid, nclass, bias=True, )

        """GAT from torch-geometric"""
        # nclass = int(nclass)
        # self.gc1 = GATConv(nfeat, nhid, heads=8, dropout=0.6)
        # self.gc2 = GATConv(nhid*8, nclass, heads=1, concat=True, dropout=0.6)

        """GIN from torch-geometric"""
        # dim = 32
        # nn1 = Sequential(Linear(nfeat, dim), ReLU(), )
        # self.gc1 = GINConv(nn1)
        # # self.bn1 = torch.nn.BatchNorm1d(dim)
        # nn2 = Sequential(Linear(dim, dim), ReLU(), )
        # self.gc2 = GINConv(nn2)
        # self.jump = JumpingKnowledge(mode='cat')
        # # self.bn2 = torch.nn.BatchNorm1d(dim)
        # self.fc2 = Linear(dim, int(nclass))

        # """JK-Nets"""
        # num_features = nfeat
        # dim = 32
        # nn1 = Sequential(Linear(num_features, dim), ReLU(), )
        # self.gc1 = GINConv(nn1)
        # self.bn1 = torch.nn.BatchNorm1d(dim)
        #
        # nn2 = Sequential(Linear(dim, dim), ReLU(), )
        # self.gc2 = GINConv(nn2)
        # nn3 = Sequential(Linear(dim, dim), ReLU(), )
        # self.gc3 = GINConv(nn3)
        #
        # self.jump = JumpingKnowledge(mode='cat') # 'cat', 'lstm', 'max'
        # self.bn2 = torch.nn.BatchNorm1d(dim)
        # # self.fc1 = Linear(dim*3, dim)
        # self.fc2 = Linear(dim*2, int(nclass))

    def forward(self, x, adj):
        """we don't change the edge_index, just update the edge_weight;
        some edge_weight are regarded as removed if it equals to zero"""
        x = x.to_dense()
        self.attention = False

        """GCN and GAT"""
        #if self.attention:
        #    adj = self.att_coef(x, adj, i=0)
        edge_index = adj._indices().to(x.get_device())
        # print(x.get_device(), edge_index.get_device(),adj.get_device())
        x = self.gc1(x, edge_index, edge_weight=adj._values())
        x = F.relu(x)
        # x = self.bn1(x)
        if self.attention:  # if attention=True, use attention mechanism
            adj_2 = self.att_coef(x, adj, i=1)
            adj_memory = adj_2.to_dense()  # without memory
            # adj_memory = self.gate * adj.to_dense() + (1 - self.gate) * adj_2.to_dense()
            row, col = adj_memory.nonzero()[:, 0], adj_memory.nonzero()[:, 1]
            edge_index = torch.stack((row, col), dim=0)
            adj_values = adj_memory[row, col]
        else:
            edge_index = adj._indices()
            adj_values = adj._values()
        edge_index = edge_index.to(x.get_device())
        adj_values = adj_values.to(x.get_device())
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight=adj_values)

        # """GIN"""
        # if self.attention:
        #     adj = self.att_coef(x, adj, i=0)
        # x = F.relu(self.gc1(x, edge_index=edge_index, edge_weight=adj._values()))
        # if self.attention:  # if attention=True, use attention mechanism
        #     adj_2 = self.att_coef(x, adj, i=1)
        #     adj_values = self.gate * adj._values() + (1 - self.gate) * adj_2._values()
        # else:
        #     adj_values = adj._values()
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = F.relu(self.gc2(x, edge_index=edge_index, edge_weight=adj_values))
        # # x = [x] ### Add Jumping        # x = self.jump(x)
        # x = F.dropout(x, p=0.2,training=self.training)
        # x = self.fc2(x)

        # """JK-Nets"""
        # if self.attention:
        #     adj = self.att_coef(x, adj, i=0)
        # x1 = F.relu(self.gc1(x, edge_index=edge_index, edge_weight=adj._values()))
        # if self.attention:  # if attention=True, use attention mechanism
        #     adj_2 = self.att_coef(x1, adj, i=1)
        #     adj_values = self.gate * adj._values() + (1 - self.gate) * adj_2._values()
        # else:
        #     adj_values = adj._values()
        # x1 = F.dropout(x1, self.dropout, training=self.training)
        # x2 = F.relu(self.gc2(x1, edge_index=edge_index, edge_weight=adj_values))
        # x2 = F.dropout(x2, self.dropout, training=self.training)
        # x_last = self.jump([x1, x2])
        # x_last = F.dropout(x_last, self.dropout,training=self.training)
        # x = self.fc2(x_last)

        return x  #F.log_softmax(x, dim=1)

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()
        self.drop_learn_1.reset_parameters()
        self.drop_learn_2.reset_parameters()
        try:
            self.gate.reset_parameters()
            self.fc2.reset_parameters()
        except:
            pass

    def att_coef(self, fea, edge_index, is_lil=False, i=0):
        if is_lil == False:
            edge_index = edge_index._indices()
        else:
            edge_index = edge_index.tocoo()

        n_node = fea.shape[0]
        row, col = edge_index[0].cpu().data.numpy()[:], edge_index[1].cpu().data.numpy()[:]

        fea_copy = fea.cpu().data.numpy()
        sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        sim = sim_matrix[row, col]
        sim[sim < 0.1] = 0
        # print('dropped {} edges'.format(1-sim.nonzero()[0].shape[0]/len(sim)))

        # """use jaccard for binary features and cosine for numeric features"""
        # fea_start, fea_end = fea[edge_index[0]], fea[edge_index[1]]
        # isbinray = np.array_equal(fea_copy, fea_copy.astype(bool))  # check is the fea are binary
        # np.seterr(divide='ignore', invalid='ignore')
        # if isbinray:
        #     fea_start, fea_end = fea_start.T, fea_end.T
        #     sim = jaccard_score(fea_start, fea_end, average=None)  # similarity scores of each edge
        # else:
        #     fea_copy[np.isinf(fea_copy)] = 0
        #     fea_copy[np.isnan(fea_copy)] = 0
        #     sim_matrix = cosine_similarity(X=fea_copy, Y=fea_copy)  # try cosine similarity
        #     sim = sim_matrix[edge_index[0], edge_index[1]]
        #     sim[sim < 0.01] = 0

        """build a attention matrix"""
        att_dense = lil_matrix((n_node, n_node), dtype=np.float32)
        att_dense[row, col] = sim
        if att_dense[0, 0] == 1:
            att_dense = att_dense - sp.diags(att_dense.diagonal(), offsets=0, format="lil")
        # normalization, make the sum of each row is 1
        att_dense_norm = normalize(att_dense, axis=1, norm='l1')

        """add learnable dropout, make character vector"""
        if self.drop:
            character = np.vstack((att_dense_norm[row, col].A1,
                                   att_dense_norm[col, row].A1))
            character = torch.from_numpy(character.T)
            drop_score = self.drop_learn_1(character)
            drop_score = torch.sigmoid(drop_score)  # do not use softmax since we only have one element
            mm = torch.nn.Threshold(0.5, 0)
            drop_score = mm(drop_score)
            mm_2 = torch.nn.Threshold(-0.49, 1)
            drop_score = mm_2(-drop_score)
            drop_decision = drop_score.clone().requires_grad_()
            # print('rate of left edges', drop_decision.sum().data/drop_decision.shape[0])
            drop_matrix = lil_matrix((n_node, n_node), dtype=np.float32)
            drop_matrix[row, col] = drop_decision.cpu().data.numpy().squeeze(-1)
            att_dense_norm = att_dense_norm.multiply(drop_matrix.tocsr())  # update, remove the 0 edges

        if att_dense_norm[0, 0] == 0:  # add the weights of self-loop only add self-loop at the first layer
            degree = (att_dense_norm != 0).sum(1).A1
            lam = 1 / (degree + 1)  # degree +1 is to add itself
            self_weight = sp.diags(np.array(lam), offsets=0, format="lil")
            att = att_dense_norm + self_weight  # add the self loop
        else:
            att = att_dense_norm

        row, col = att.nonzero()
        att_adj = np.vstack((row, col))
        att_edge_weight = att[row, col]
        att_edge_weight = np.exp(att_edge_weight)  # exponent, kind of softmax
        att_edge_weight = torch.tensor(np.array(att_edge_weight)[0], dtype=torch.float32)  # .cuda()
        att_adj = torch.tensor(att_adj, dtype=torch.int64)  # .cuda()

        shape = (n_node, n_node)
        new_adj = torch.sparse.FloatTensor(att_adj, att_edge_weight, shape)
        return new_adj

    def add_loop_sparse(self, adj, fill_value=1):
        # make identify sparse tensor
        row = torch.range(0, int(adj.shape[0] - 1), dtype=torch.int64)
        i = torch.stack((row, row), dim=0)
        v = torch.ones(adj.shape[0], dtype=torch.float32)
        shape = adj.shape
        I_n = torch.sparse.FloatTensor(i, v, shape)
        return adj + I_n.to(self.device)

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200,
            attention=False, initialize=True, verbose=False, normalize=False, patience=500, ):
        '''
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        '''
        self.sim = None
        # self.idx_test = idx_test
        self.attention = attention
        # if self.attention:
        #     att_0 = self.att_coef_1(features, adj)
        #     adj = att_0 # update adj
        #     self.sim = att_0 # update att_0

        # self.device = self.gc1.weight.device

        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        adj = self.add_loop_sparse(adj)

        """The normalization gonna be done in the GCNConv"""
        self.adj_norm = adj
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train],
                                    weight=None)  # this weight is the weight of each training nodes
            loss_train.backward()
            optimizer.step()
            if verbose and i % 20 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            # print('epoch', i)
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            self.eval()

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            # acc_test = utils.accuracy(output[self.idx_test], labels[self.idx_test])

            if verbose and i % 50 == 0:
                print('Epoch {}, training loss: {}, val acc: {}, '.format(i, loss_train.item(), acc_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        # """my test"""
        # output_ = self.forward(self.features, self.adj_norm)
        # acc_test_ = utils.accuracy(output_[self.idx_test], labels[self.idx_test])
        # print('With best weights, test acc:', acc_test_)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

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
            print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val))
        self.load_state_dict(weights)

    def test(self, idx_test):
        # self.eval()
        output = self.predict()  # here use the self.features and self.adj_norm in training stage
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()

    def _set_parameters(self):
        # TODO
        pass

    def predict(self, features=None, adj=None):
        '''By default, inputs are unnormalized data'''
        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GNNGuardWGTL(nn.Module):
    """ 2 Layer Graph Convolutional Network + witness complex-based layer
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None, alpha = 0.8, beta = 0.1, gamma = 0.1, lambda_coeff = 0.001, aggregation_method = 'attention'):

        super(GNNGuardWGTL, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
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
        self.mlp_score = nn.Linear(nhid, 1)
        self.mlp_witness = nn.Linear(nhid, 2)
        self.attention = Attention(nhid)
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.gat1 = GCN_GNNGuard(nfeat, 64, nclass=nhid, dropout=0.5, lr=0.01, drop=False, weight_decay=5e-4, n_edge=1,with_relu=True,
                 with_bias=True, device=None)
        self.gat2 = GCN_GNNGuard(nfeat, 64, nclass=nhid, dropout=0.5, lr=0.01, drop=False, weight_decay=5e-4, n_edge=1,with_relu=True,
                 with_bias=True, device=None)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.agg = aggregation_method
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_coeff = lambda_coeff

    def forward(self, x, adj, global_witness_complex_feat, local_witness_complex_feat):
        if self.with_relu:
            x = F.relu(self.gat1(x, adj))
        else:
            x = self.gat2(x, adj)

        local_witness_comp_topo = self.lcnn(local_witness_complex_feat)
        global_witness_comp_topo = self.gcnn(global_witness_complex_feat)

        sim_matrix = cosine_similarity(X=local_witness_comp_topo.cpu().detach().numpy(), Y=local_witness_comp_topo.cpu().detach().numpy())
        skeleton_protein0 = gd.RipsComplex(
            distance_matrix=sim_matrix,
            max_edge_length=1.0
        )

        Rips_simplex_tree_protein0 = skeleton_protein0.create_simplex_tree(max_dimension=1)
        BarCodes_Rips0 = Rips_simplex_tree_protein0.persistence()
        local_witness_comp_topo_pd = Rips_simplex_tree_protein0.persistence_intervals_in_dimension(0)
        loss_topo = topo_loss(PD = local_witness_comp_topo_pd, p = 2, q= 1)

        # x and witness_comp_topo have the same size
        if self.agg == 'einsum':
            global_witness_comp_topo = global_witness_comp_topo.view(global_witness_comp_topo.size(1))
            x = torch.einsum('nb, b-> nb', x, global_witness_comp_topo)
            x = torch.einsum('nb, nb-> nb', x, local_witness_comp_topo)
        elif self.agg == 'weighted_sum':
            x = self.alpha * x + self.beta * global_witness_comp_topo + self.gamma * local_witness_comp_topo
        elif self.agg == 'attention':
            global_witness_comp_topo = global_witness_comp_topo.view(global_witness_comp_topo.size(1))
            #emb = torch.stack([x, local_witness_comp_topo], dim=1)
            #emb, att = self.attention(emb)
            # emb = x * 0.9 + 0.1*local_witness_comp_topo # Cora
            emb = x * 0.99 + 0.01*local_witness_comp_topo #Polblogs
            x = torch.einsum('nb, b-> nb', emb, global_witness_comp_topo)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1), loss_topo

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, global_witness_complex_feat, local_witness_complex_feat, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features.to(self.device)
        self.labels = labels
        #self.witness_complex_feat = witness_complex_feat.to(self.device)
        self.global_witness_complex_feat = global_witness_complex_feat.to(self.device)
        self.local_witness_complex_feat = local_witness_complex_feat.to(self.device)

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output, loss_topo = self.forward(self.features.to(self.device), self.adj_norm, self.global_witness_complex_feat.to(self.device), self.local_witness_complex_feat.to(self.device))
            loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + self.lambda_coeff * loss_topo
            loss_train.backward()
            optimizer.step()
            if verbose and i % 20 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output, _ = self.forward(self.features.to(self.device), self.adj_norm, self.global_witness_complex_feat.to(self.device), self.local_witness_complex_feat.to(self.device))
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training GNNGuard+WGTL w/ val model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0
        # print('train')
        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, loss_topo = self.forward(self.features, self.adj_norm, self.global_witness_complex_feat, self.local_witness_complex_feat)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + self.lambda_coeff * loss_topo
            loss_train.backward()
            optimizer.step()

            if verbose and i % 20 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output, _ = self.forward(self.features, self.adj_norm, self.global_witness_complex_feat, self.local_witness_complex_feat)
            # print('after forward')
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())
                # print('acc val: ',acc_val)
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output, loss_topo = self.forward(self.features, self.adj_norm, self.global_witness_complex_feat, self.local_witness_complex_feat)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + self.lambda_coeff * loss_topo
            loss_train.backward()
            optimizer.step()

            if verbose and i % 20 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output, _ = self.forward(self.features, self.adj_norm, self.global_witness_complex_feat, self.local_witness_complex_feat)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

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

    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output, _ = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm, self.global_witness_complex_feat, self.local_witness_complex_feat)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm, self.global_witness_complex_feat, self.local_witness_complex_feat)



