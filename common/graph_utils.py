from __future__ import absolute_import

import torch
import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges_merge(num_pts, edges, sub_edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.full(edges.shape[0], 1.0), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    sub_edges = np.array(sub_edges, dtype=np.int32)
    sub_data, n, m = np.full(sub_edges.shape[0], 0.5), sub_edges[:, 0], sub_edges[:, 1]
    sub_adj_mx = sp.coo_matrix((sub_data, (n, m)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    sub_adj_mx = sub_adj_mx + sub_adj_mx.T.multiply(sub_adj_mx.T > sub_adj_mx) - sub_adj_mx.multiply(
        sub_adj_mx.T > sub_adj_mx)

    adj_mx = adj_mx + sub_adj_mx
    adj_mx = normalize(adj_mx) #+ sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    
    adj_mx = adj_mx * (1-torch.eye(adj_mx.shape[0])) + torch.eye(adj_mx.shape[0])

    return adj_mx


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx)  # + sp.eye(adj_mx.shape[0]))
    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)

    adj_mx = adj_mx * (1 - torch.eye(adj_mx.shape[0])) + torch.eye(adj_mx.shape[0])

    return adj_mx


def adj_mx_from_skeleton(skeleton):
    num_joints = skeleton.num_joints()
    edges = list(filter(lambda x: x[1] >= 0, zip(list(range(0, num_joints)), skeleton.parents())))
    ext_edges_1 = []
    ext_edges_2 = []
    ext_edges_3 = []
    ext_edges_4 = []
    for i in range(num_joints):
        for j in range(len(skeleton._joints_graph_ext_1[i])):
            if skeleton._joints_graph_ext_1[i][j] > -1:
                ext_edges_1.append(tuple([i, skeleton._joints_graph_ext_1[i][j]]))
        for j in range(len(skeleton._joints_graph_ext_2[i])):
            if skeleton._joints_graph_ext_2[i][j] > -1:
                ext_edges_2.append(tuple([i, skeleton._joints_graph_ext_2[i][j]]))
        for j in range(len(skeleton._joints_graph_ext_3[i])):
            if skeleton._joints_graph_ext_3[i][j] > -1:
                ext_edges_3.append(tuple([i, skeleton._joints_graph_ext_3[i][j]]))
        for j in range(len(skeleton._joints_graph_ext_4[i])):
            if skeleton._joints_graph_ext_4[i][j] > -1:
                ext_edges_4.append(tuple([i, skeleton._joints_graph_ext_4[i][j]]))

    adj = adj_mx_from_edges(num_joints, edges, sparse=False)
    adj_ext_1 = adj_mx_from_edges(num_joints, ext_edges_1, sparse=False)
    adj_ext_2 = adj_mx_from_edges(num_joints, ext_edges_2, sparse=False)
    adj_ext_3 = adj_mx_from_edges(num_joints, ext_edges_3, sparse=False)
    adj_ext_4 = adj_mx_from_edges(num_joints, ext_edges_4, sparse=False)
    return adj, adj_ext_1, adj_ext_2, adj_ext_3, adj_ext_4

