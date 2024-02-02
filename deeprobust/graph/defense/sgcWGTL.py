"""
Extended from https://github.com/rusty1s/pytorch_geometric/tree/master/benchmark/citation
"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
# from torch_geometric.nn import SGConv
from vit_pytorch import ViT
import gudhi as gd
from sklearn.metrics.pairwise import cosine_similarity

from typing import Optional
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
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


class SGConv(MessagePassing):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops :math:`K`. (default: :obj:`1`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X}` on
            first execution, and will use the cached version for further
            executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:**
          node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 1,
                 cached: bool = False, add_self_loops: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        # self.cached = cached
        self.add_self_loops = add_self_loops

        # self._cached_x = None

        self.lin = Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        # self._cached_x = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        # cache = self._cached_x
        # if cache is None:
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)

        for k in range(self.K):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                size=None)
                # if self.cached:
                #     self._cached_x = x
        # else:
        #     x = cache.detach()

        return self.lin(x)

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')

class SGCWGTL(torch.nn.Module):
    """ SGC based on pytorch geometric. Simplifying Graph Convolutional Networks.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nclass : int
        size of output dimension
    K: int
        number of propagation in SGC
    cached : bool
        whether to set the cache flag in SGConv
    lr : float
        learning rate for SGC
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_bias: bool
        whether to include bias term in SGC weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train SGC.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import SGC
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> sgc = SGC(nfeat=features.shape[1], K=3, lr=0.1,
              nclass=labels.max().item() + 1, device='cuda')
    >>> sgc = sgc.to('cuda')
    >>> pyg_data = Dpr2Pyg(data) # convert deeprobust dataset to pyg dataset
    >>> sgc.fit(pyg_data, train_iters=200, patience=200, verbose=True) # train with earlystopping
    """


    def __init__(self, nfeat, nclass, K=2, cached=True, lr=0.01, nhid=64,
            weight_decay=5e-4, with_bias=True, device=None,
            alpha = 0.8, beta = 0.1, gamma = 0.1, lambda_coeff = 0.001, aggregation_method = 'attention'):

        super(SGCWGTL, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device

        self.conv1 = SGConv(nfeat,
                nhid, bias=with_bias, K=K, cached=cached)
        # self.conv2 = SGConv(nhid,
        #         nclass, K=1, bias=with_bias)
        self.gc2 = GCNConv(nhid, nclass, bias=with_bias)

        self.weight_decay = weight_decay
        self.lr = lr
        self.output = None
        self.best_model = None
        self.best_output = None
        self.agg = aggregation_method
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_coeff = lambda_coeff
        # nhid=nclass
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

        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        # print('x.shape: ',x.shape)
        # x and witness_comp_topo have the same size
        if self.agg == 'einsum':
            global_witness_comp_topo = global_witness_comp_topo.view(global_witness_comp_topo.size(1))
            x = torch.einsum('nb, b-> nb', x, global_witness_comp_topo)
            x = torch.einsum('nb, nb-> nb', x, local_witness_comp_topo)
        elif self.agg == 'weighted_sum':
            # print('global_witness_comp_topo.shape: ', global_witness_comp_topo.shape)
            x = self.alpha * x + self.beta * global_witness_comp_topo + self.gamma * local_witness_comp_topo
        elif self.agg == 'attention':
            global_witness_comp_topo = global_witness_comp_topo.view(global_witness_comp_topo.size(1))
            # print('x.shape: ',x.shape, local_witness_comp_topo.shape,global_witness_comp_topo.shape)
            emb = torch.stack([x, local_witness_comp_topo], dim=1)
            emb, att = self.attention(emb)
            x = torch.einsum('nb, b-> nb', emb, global_witness_comp_topo)
        # x = self.conv2(x,edge_index)
        x = self.gc2(x,edge_index)
        return F.log_softmax(x, dim=1), loss_topo

    def initialize(self):
        """Initialize parameters of SGC.
        """
        self.conv1.reset_parameters()
        # self.conv2.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, pyg_data, global_witness_complex_feat, local_witness_complex_feat, train_iters=200, initialize=True, verbose=False, patience=500, **kwargs):
        """Train the SGC model, when idx_val is not None, pick the best model
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

        # self.device = self.conv1.weight.device
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
            print('=== training SGC model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        labels = self.data.y
        train_mask, val_mask = self.data.train_mask, self.data.val_mask

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output,loss_topo = self.forward(self.data)

            loss_train = F.nll_loss(output[train_mask], labels[train_mask]) + self.lambda_coeff * loss_topo
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
        """Evaluate SGC performance on test set.

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
            output (log probabilities) of SGC
        """

        self.eval()
        return self.forward(self.data)


if __name__ == "__main__":
    from deeprobust.graph.data import Dataset, Dpr2Pyg
    # # from deeprobust.graph.defense import SGC
    # data = Dataset(root='/tmp/', name='cora')
    # adj, features, labels = data.adj, data.features, data.labels
    # idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    # sgc = SGC(nfeat=features.shape[1],
    #       nclass=labels.max().item() + 1, device='cpu')
    # sgc = sgc.to('cpu')
    # pyg_data = Dpr2Pyg(data)
    # sgc.fit(pyg_data, verbose=True) # train with earlystopping
    # sgc.test()
    # print(sgc.predict())

