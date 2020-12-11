import torch
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F
import numpy as np
import logging


logging.basicConfig(
    filename='log/train.log',
    filemode='w',
    level=logging.DEBUG,
)


def unpack_to_batch(X, n_obs_list):
    indexes = np.cumsum([0] + n_obs_list)
    return [
        X[indexes[i]:indexes[i+1], ]
        for i in range(len(indexes)-1)
    ]


def standardize(X):
    X = X / (X.std(dim=0, keepdim=True) + 1e-6)
    X = X - torch.mean(X, dim=0, keepdim=True)
    X = (1./np.sqrt(max(X.size(0) - 1, 1))) * X
    return X


def batch_corr(X, n_obs_list):
    X = standardize(X)
    batch = unpack_to_batch(X, n_obs_list)
    pad = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return pad.permute(0, 2, 1)@pad


def batch_cross_corr(X_lower, X_upper, Bdry_norm_T, n_obs_list_upper):
    X_lower = standardize(X_lower)
    X_upper = standardize(X_upper)
    X_lower_prop = Bdry_norm_T@X_lower
    batch_lower = unpack_to_batch(X_lower_prop, n_obs_list_upper)
    batch_upper = unpack_to_batch(X_upper, n_obs_list_upper)

    pad_lower = torch.nn.utils.rnn.pad_sequence(batch_lower, batch_first=True)
    pad_upper = torch.nn.utils.rnn.pad_sequence(batch_upper, batch_first=True)

    return pad_upper.permute(0, 2, 1)@pad_lower


class MLP(nn.Module):
    def __init__(
        self,
        h_sizes,
        out_size,
        h_dropouts='auto',
        h_bias=False,
        out_bias=False,
    ):
        super(MLP, self).__init__()
        self.h_sizes = h_sizes
        self.out_size = out_size
        self.h_dropouts = h_dropouts
        self.h_bias = h_bias
        self.out_bias = out_bias
        self.hidden = nn.ModuleList()

        if self.h_dropouts == 'auto':
            self.h_dropouts = [0.5]*len(self.h_sizes)
        if self.h_dropouts is not None:
            self.hidden.append(nn.Dropout(p=self.h_dropouts[0]))
        for k in range(len(self.h_sizes)-1):
            self.hidden.append(nn.Linear(
                self.h_sizes[k],
                self.h_sizes[k+1],
                bias=self.h_bias,
            ))
            if self.h_dropouts is not None:
                self.hidden.append(nn.Dropout(p=self.h_dropouts[k]))
        # Output layer
        self.out = nn.Linear(
            self.h_sizes[-1],
            self.out_size,
            bias=self.out_bias,
        )

    def forward(self, x):
        # Feedforward
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.out(x)


class SortPool(nn.Module):
    def __init__(
        self,
        k
    ):
        super(SortPool, self).__init__()
        self.k = k

    def forward(
        self,
        X
    ):
        _, top_k_indices = X.topk(k=self.k, dim=0)

        sortpooled = X.index_select(0, top_k_indices[:, -1])

        return sortpooled


class GraphConv(nn.Module):
    def __init__(
        self,
        num_node_feats,
        num_edge_feats,
        num_triangle_feats,
        output_node_dim,
        output_edge_dim,
        output_triangle_dim,
        bias,
        p,
    ):
        super(GraphConv, self).__init__()
        self.num_node_feats = num_node_feats
        self.output_node_dim = output_node_dim
        self.bias = bias
        self.p = p

        logging.info('initializing GraphConv')
        logging.info(f'num_node_feats: {self.num_node_feats}')
        logging.info(f'output_node_dim: {self.output_node_dim}')
        logging.info(f'bias: {self.bias}')
        logging.info(f'p: {self.p}')
        self.n2n_weights = nn.Linear(
            self.num_node_feats*self.p,
            self.output_node_dim,
            bias=self.bias,
        )
        logging.info(
            f'self.n2n_weights.weight.shape: '
            f'{self.n2n_weights.weight.shape}'
        )

    def forward(
        self,
        L0,
        X0,
    ):
        logging.info(f'X0.shape before powers: {X0.shape}')
        X0_list = [torch.pow(X0, n) for n in range(1, self.p+1)]
        X0 = torch.cat(X0_list, dim=1)
        logging.info(f'X0.shape after powers: {X0.shape}')

        # node to node
        n2n = self.n2n_weights(X0)  # Y00 = X0*W00
        n2n = torch.mm(L0, n2n)  # L0*Y00

        return F.relu(n2n)


class SCConv(nn.Module):
    def __init__(
        self,
        num_node_feats,
        num_edge_feats,
        num_triangle_feats,
        output_node_dim,
        output_edge_dim,
        output_triangle_dim,
        bias,
        p,
    ):
        super(SCConv, self).__init__()
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.num_triangle_feats = num_triangle_feats
        self.output_node_dim = output_node_dim
        self.output_edge_dim = output_edge_dim
        self.output_triangle_dim = output_triangle_dim
        self.bias = bias
        self.p = p

        self.n2n_weights = nn.Linear(
            self.num_node_feats*self.p,
            self.output_node_dim,
            bias=self.bias,
        )

        self.n2e_weights = nn.Linear(
            self.num_node_feats*self.p,
            self.output_edge_dim,
            bias=self.bias,
        )

        self.e2e_weights = nn.Linear(
            self.num_edge_feats*self.p,
            self.output_edge_dim,
            bias=self.bias,
        )

        self.e2n_weights = nn.Linear(
            self.num_edge_feats*self.p,
            self.output_node_dim,
            bias=self.bias,
        )

        self.e2t_weights = nn.Linear(
            self.num_edge_feats*self.p,
            self.output_triangle_dim,
            bias=self.bias,
        )

        self.t2e_weights = nn.Linear(
            self.num_triangle_feats*self.p,
            self.output_edge_dim,
            bias=self.bias,
        )

        self.t2t_weights = nn.Linear(
            self.num_triangle_feats*self.p,
            self.output_triangle_dim,
            bias=self.bias,
        )

    def forward(
        self,
        L0,
        L1,
        L2,
        D1invB1,
        D2B1TD1inv,
        B2TD2inv,
        B2D3,
        X0,
        X1,
        X2,
    ):
        X0_list = [torch.pow(X0, n) for n in range(1, self.p+1)]
        X1_list = [torch.pow(X1, n) for n in range(1, self.p+1)]
        X2_list = [torch.pow(X2, n) for n in range(1, self.p+1)]

        X0 = torch.cat(X0_list, dim=1)
        X1 = torch.cat(X1_list, dim=1)
        X2 = torch.cat(X2_list, dim=1)

        # node to node
        n2n = self.n2n_weights(X0)  # Y00 = X0*W00
        n2n = torch.sparse.mm(L0, n2n)  # L0*Y00

        # node to edge
        n2e = self.n2e_weights(X0)  # Y01 = X0*W01
        n2e = torch.sparse.mm(D2B1TD1inv, n2e)  # D2*B1.T*D1^-1*Y01

        # edge to node
        e2n = self.e2n_weights(X1)  # Y10 = X1*W10
        e2n = torch.sparse.mm(D1invB1, e2n)  # D1invB1*Y10

        # edge to edge
        e2e = self.e2e_weights(X1)  # Y11 = X1*W11
        e2e = torch.sparse.mm(L1, e2e)  # L1*Y11

        # edge to triangle
        e2t = self.e2t_weights(X1)  # Y21 = X1*W21
        e2t = torch.sparse.mm(B2TD2inv, e2t)  # B2TD2inv*Y21

        # triangle to triangle
        t2t = self.t2t_weights(X2)  # Y22 = X2*W22
        t2t = torch.sparse.mm(L2, t2t)  # L2*Y22

        # triangle to edge
        t2e = self.t2e_weights(X2)  # Y12 = X2*W12
        t2e = torch.sparse.mm(B2D3, t2e)  # B2D3*Y12

        X0 = (1/2.)*F.relu(n2n + e2n)
        X1 = (1/3.)*F.relu(e2e + n2e + t2e)
        X2 = (1/2.)*F.relu(t2t + e2t)

        return X0, X1, X2


class SCCorr(nn.Module):
    def __init__(
        self,
    ):
        super(SCCorr, self).__init__()

    def forward(
        self,
        X0,
        X1,
        X2,
        D2B1TD1inv,
        B2TD2inv,
        num_nodes,
        num_edges,
        num_triangles,
    ):
        X0corr = batch_corr(X0, n_obs_list=num_nodes)
        X1corr = batch_corr(X1, n_obs_list=num_edges)
        X2corr = batch_corr(X2, n_obs_list=num_triangles)

        X01corr = batch_cross_corr(
            X_lower=X0,
            X_upper=X1,
            Bdry_norm_T=D2B1TD1inv,
            n_obs_list_upper=num_edges,
        )
        X12corr = batch_cross_corr(
            X_lower=X1,
            X_upper=X2,
            Bdry_norm_T=B2TD2inv,
            n_obs_list_upper=num_triangles,
        )

        return X0corr, X1corr, X2corr, X01corr, X12corr


class ConvNet(nn.Module):
    def __init__(
        self,
        params,
    ):
        super(ConvNet, self).__init__()
        self.params = params

        # stride is same as kernel_size
        self.conv1d_nodes_1 = Conv1d(
            in_channels=1,
            out_channels=self.params['conv1d_nodes_1']['out_channels'],
            groups=1,
            kernel_size=self.params['conv1d_nodes_1']['kernel_size'],
            stride=self.params['conv1d_nodes_1']['stride'],
            padding=self.params['conv1d_nodes_1']['padding'],
            dilation=self.params['conv1d_nodes_1']['dilation'],
            bias=False
        )

        self.mlp_out_conv1d = MLP(
            h_sizes=[
                self.params['mlp_out_conv1d']['input_size'],
                # mlp_out_input_size,
                self.params['mlp_out_conv1d']['h_size'],
            ],
            out_size=self.params['mlp_out_conv1d']['out_size'],
            h_dropouts=self.params['mlp_out_conv1d']['h_dropouts'],
            h_bias=self.params['mlp_out_conv1d']['h_bias'],
            out_bias=self.params['mlp_out_conv1d']['out_bias'],
        )

    def forward(
        self,
        L0,
        L1,
        L2,
        D1invB1,
        D2B1TD1inv,
        B2TD2inv,
        B2D3,
        X0,
        X1,
        X2,
        num_nodes,
        num_edges,
        num_triangles,
    ):
        X0_batch = unpack_to_batch(X0, num_nodes)
        nodes = torch.nn.utils.rnn.pad_sequence(X0_batch, batch_first=True)
        nodes = nodes.flatten(start_dim=1).unsqueeze(1)
        nodes = self.conv1d_nodes_1(nodes)
        nodes = F.relu(nodes)
        nodes = nodes.flatten(start_dim=1)

        return self.mlp_out_conv1d(nodes)


class GraphNet(nn.Module):
    def __init__(
        self,
        params,
    ):
        super(GraphNet, self).__init__()
        self.params = params

        self.graphconv1 = GraphConv(
            num_node_feats=self.params['num_node_feats'],
            num_edge_feats=self.params['num_edge_feats'],
            num_triangle_feats=self.params['num_triangle_feats'],
            output_node_dim=self.params['conv1']['output_node_dim'],
            output_edge_dim=self.params['conv1']['output_edge_dim'],
            output_triangle_dim=self.params['conv1']['output_triangle_dim'],
            bias=self.params['conv1']['bias'],
            p=self.params['conv1']['p'],
        )

        # stride is same as kernel_size
        self.conv1d_nodes_1 = Conv1d(
            in_channels=1,
            out_channels=self.params['conv1d_nodes_1']['out_channels'],
            groups=1,
            kernel_size=self.params['conv1d_nodes_1']['kernel_size'],
            stride=self.params['conv1d_nodes_1']['stride'],
            padding=self.params['conv1d_nodes_1']['padding'],
            dilation=self.params['conv1d_nodes_1']['dilation'],
            bias=False
        )

        self.mlp_out_conv1d = MLP(
            h_sizes=[
                self.params['mlp_out_conv1d']['input_size'],
                self.params['mlp_out_conv1d']['h_size'],
            ],
            out_size=self.params['mlp_out_conv1d']['out_size'],
            h_dropouts=self.params['mlp_out_conv1d']['h_dropouts'],
            h_bias=self.params['mlp_out_conv1d']['h_bias'],
            out_bias=self.params['mlp_out_conv1d']['out_bias'],
        )

    def forward(
        self,
        L0,
        L1,
        L2,
        D1invB1,
        D2B1TD1inv,
        B2TD2inv,
        B2D3,
        X0,
        X1,
        X2,
        num_nodes,
        num_edges,
        num_triangles,
    ):
        X01 = self.graphconv1(
            L0=L0,
            X0=X0,
        )

        features_to_concat_0 = [
            X01,
        ]
        if self.params['orig_features'] is True:
            features_to_concat_0.insert(0, X0)

        X0_cat = torch.cat(features_to_concat_0, dim=1)

        X0_batch = unpack_to_batch(X0_cat, num_nodes)
        nodes = torch.nn.utils.rnn.pad_sequence(X0_batch, batch_first=True)
        nodes = nodes.flatten(start_dim=1).unsqueeze(1)
        nodes = self.conv1d_nodes_1(nodes)
        nodes = F.relu(nodes)
        nodes = nodes.flatten(start_dim=1)

        return self.mlp_out_conv1d(nodes)


class SCNet(nn.Module):
    def __init__(
        self,
        params,
    ):
        super(SCNet, self).__init__()
        self.params = params

        self.scconv1 = SCConv(
            num_node_feats=self.params['num_node_feats'],
            num_edge_feats=self.params['num_edge_feats'],
            num_triangle_feats=self.params['num_triangle_feats'],
            output_node_dim=self.params['conv1']['output_node_dim'],
            output_edge_dim=self.params['conv1']['output_edge_dim'],
            output_triangle_dim=self.params['conv1']['output_triangle_dim'],
            bias=self.params['conv1']['bias'],
            p=self.params['conv1']['p'],
        )

        self.sortpool_nodes = SortPool(k=self.params['k_nodes'])
        self.sortpool_edges = SortPool(k=self.params['k_edges'])
        self.sortpool_triangles = SortPool(k=self.params['k_triangles'])

        # stride is same as kernel_size
        self.conv1d_nodes_1 = Conv1d(
            in_channels=1,
            out_channels=self.params['conv1d_nodes_1']['out_channels'],
            groups=1,
            kernel_size=self.params['conv1d_nodes_1']['kernel_size'],
            stride=self.params['conv1d_nodes_1']['stride'],
            padding=self.params['conv1d_nodes_1']['padding'],
            dilation=self.params['conv1d_nodes_1']['dilation'],
            bias=False
        )

        # stride is same as kernel_size
        self.conv1d_edges_1 = Conv1d(
            in_channels=1,
            out_channels=self.params['conv1d_edges_1']['out_channels'],
            groups=1,
            kernel_size=self.params['conv1d_edges_1']['kernel_size'],
            stride=self.params['conv1d_edges_1']['stride'],
            padding=self.params['conv1d_edges_1']['padding'],
            dilation=self.params['conv1d_edges_1']['dilation'],
            bias=False
        )

        # stride is same as kernel_size
        self.conv1d_triangles_1 = Conv1d(
            in_channels=1,
            out_channels=self.params['conv1d_triangles_1']['out_channels'],
            groups=1,
            kernel_size=self.params['conv1d_triangles_1']['kernel_size'],
            stride=self.params['conv1d_triangles_1']['stride'],
            padding=self.params['conv1d_triangles_1']['padding'],
            dilation=self.params['conv1d_triangles_1']['dilation'],
            bias=False
        )

        self.mlp_out_conv1d = MLP(
            h_sizes=[
                self.params['mlp_out_conv1d']['input_size'],
                self.params['mlp_out_conv1d']['h_size'],
            ],
            out_size=self.params['mlp_out_conv1d']['out_size'],
            h_dropouts=self.params['mlp_out_conv1d']['h_dropouts'],
            h_bias=self.params['mlp_out_conv1d']['h_bias'],
            out_bias=self.params['mlp_out_conv1d']['out_bias'],
        )

    def forward(
        self,
        L0,
        L1,
        L2,
        D1invB1,
        D2B1TD1inv,
        B2TD2inv,
        B2D3,
        X0,
        X1,
        X2,
        num_nodes,
        num_edges,
        num_triangles,
    ):
        X01, X11, X21 = self.scconv1(
            L0=L0,
            L1=L1,
            L2=L2,
            D1invB1=D1invB1,
            D2B1TD1inv=D2B1TD1inv,
            B2TD2inv=B2TD2inv,
            B2D3=B2D3,
            X0=X0,
            X1=X1,
            X2=X2,
        )

        features_to_concat_0 = [
            X01,
        ]
        features_to_concat_1 = [
            X11,
        ]
        features_to_concat_2 = [
            X21,
        ]
        if self.params['orig_features'] is True:
            features_to_concat_0.insert(0, X0)
            features_to_concat_1.insert(0, X1)
            features_to_concat_2.insert(0, X2)

        X0_cat = torch.cat(features_to_concat_0, dim=1)
        X1_cat = torch.cat(features_to_concat_1, dim=1)
        X2_cat = torch.cat(features_to_concat_2, dim=1)

        X0_batch = unpack_to_batch(X0_cat, num_nodes)
        X0_batch = [self.sortpool_nodes(X) for X in X0_batch]
        nodes = torch.nn.utils.rnn.pad_sequence(X0_batch, batch_first=True)
        nodes = nodes.flatten(start_dim=1).unsqueeze(1)
        nodes = self.conv1d_nodes_1(nodes)
        nodes = F.relu(nodes)
        nodes = nodes.flatten(start_dim=1)

        X1_batch = unpack_to_batch(X1_cat, num_edges)
        X1_batch = [self.sortpool_edges(X) for X in X1_batch]
        edges = torch.nn.utils.rnn.pad_sequence(X1_batch, batch_first=True)
        edges = edges.flatten(start_dim=1).unsqueeze(1)
        edges = self.conv1d_edges_1(edges)
        edges = F.relu(edges)
        edges = edges.flatten(start_dim=1)

        X2_batch = unpack_to_batch(X2_cat, num_triangles)
        X2_batch = [self.sortpool_triangles(X) for X in X2_batch]
        triangles = torch.nn.utils.rnn.pad_sequence(X2_batch, batch_first=True)
        triangles = triangles.flatten(start_dim=1).unsqueeze(1)
        triangles = self.conv1d_triangles_1(triangles)
        triangles = F.relu(triangles)
        triangles = triangles.flatten(start_dim=1)

        x = torch.cat([nodes, edges, triangles], dim=1)
        output = self.mlp_out_conv1d(x)

        return output
