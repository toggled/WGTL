import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx 
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def largest_connected_components(adj, n_components=1):
    _, component_indices = sp.csgraph.connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep

def load_npz(file_name, is_sparse=True, require_lcc = True):
    with np.load(file_name,allow_pickle=True) as loader:
        # loader = dict(loader)
        if is_sparse:
            adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                 loader['adj_indptr']), shape=loader['adj_shape'])
            if 'attr_data' in loader:
                features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                          loader['attr_indptr']), shape=loader['attr_shape'])
            else:
                features = None
            labels = loader.get('labels')
        else:
            adj = loader['adj_data']
            if 'attr_data' in loader:
                features = loader['attr_data']
            else:
                features = None
            labels = loader.get('labels')
    if features is None:
        features = np.eye(adj.shape[0])
    features = sp.csr_matrix(features, dtype=np.float32)

    adj = adj + adj.T
    adj = adj.tolil()
    adj[adj > 1] = 1

    if require_lcc:
        lcc = largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels = labels[lcc]
        assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    # whether to set diag=0?
    adj.setdiag(0)
    adj = adj.astype("float32").tocsr()
    adj.eliminate_zeros()

    assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
    assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

    return adj, features, labels

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def all_pair_shortest_path_lengths_sparse(graph):
    # Ensure the input graph is connected (optional step)
    if not nx.is_connected(graph):
        raise ValueError("Input graph must be connected for all-pair shortest path computation.")

    # Get the number of nodes in the graph
    num_nodes = len(graph.nodes)

    # Compute the shortest path lengths using NetworkX
    shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))

    # Create a sparse distance matrix
    distance_matrix = np.inf * np.ones((num_nodes, num_nodes), dtype=np.float32)

    # Fill in the distance matrix with the computed shortest path lengths
    for source, targets in shortest_paths.items():
        for target, length in targets.items():
            distance_matrix[source][target] = length

    # Convert the distance matrix to a sparse CSR matrix
    sparse_distance_matrix = csr_matrix(distance_matrix)

    return sparse_distance_matrix, np.max(distance_matrix)

def getGreedyPerm(D):
    """
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points

    Return
    ------
    lamdas: list
        Insertion radii of all points
    """

    N = D.shape[0]
    #By default, takes the first point in the permutation to be the
    #first point in the point cloud, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return lambdas[perm]

def getApproxSparseDM(lambdas, eps, D):
    """
    Purpose: To return the sparse edge list with the warped distances, sorted by weight

    Parameters
    ----------
    lambdas: list
        insertion radii for points
    eps: float
        epsilon approximation constant
    D: ndarray
        NxN distance matrix, okay to modify because last time it's used

    Return
    ------
    DSparse: scipy.sparse
        A sparse NxN matrix with the reweighted edges
    """
    N = D.shape[0]
    E0 = (1+eps)/eps
    E1 = (1+eps)**2/eps

    # Create initial sparse list candidates (Lemma 6)
    # Search neighborhoods
    nBounds = ((eps**2+3*eps+2)/eps)*lambdas

    # Set all distances outside of search neighborhood to infinity
    D[D > nBounds[:, None]] = np.inf
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    idx = I < J
    I = I[(D < np.inf)*(idx == 1)]
    J = J[(D < np.inf)*(idx == 1)]
    D = D[(D < np.inf)*(idx == 1)]

    #Prune sparse list and update warped edge lengths (Algorithm 3 pg. 14)
    minlam = np.minimum(lambdas[I], lambdas[J])
    maxlam = np.maximum(lambdas[I], lambdas[J])

    # Rule out edges between vertices whose balls stop growing before they touch
    # or where one of them would have been deleted.  M stores which of these
    # happens first
    M = np.minimum((E0 + E1)*minlam, E0*(minlam + maxlam))

    t = np.arange(len(I))
    t = t[D <= M]
    (I, J, D) = (I[t], J[t], D[t])
    minlam = minlam[t]
    maxlam = maxlam[t]

    # Now figure out the metric of the edges that are actually added
    t = np.ones(len(I))

    # If cones haven't turned into cylinders, metric is unchanged
    t[D <= 2*minlam*E0] = 0

    # Otherwise, if they meet before the M condition above, the metric is warped
    D[t == 1] = 2.0*(D[t == 1] - minlam[t == 1]*E0) # Multiply by 2 convention
    return sp.coo_matrix((D, (I, J)), shape=(N, N)).tocsr()

