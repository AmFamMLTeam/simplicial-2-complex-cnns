import sqlite3 as sql
import os
import numpy as np
import scipy
import scipy.sparse
from scipy.sparse import coo_matrix
import torch
from itertools import chain, zip_longest
import random


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def connect_to_db(path):
    con = sql.connect(path, detect_types=sql.PARSE_DECLTYPES)
    return con


def load_sc_data(
    project_dir,
    segment,  # either 'train' or 'test'
    kernel_size,
    stride,
    img_id,
):
    dataset_path = os.path.join(
        project_dir,
        f'ksize_{kernel_size}_stride_{stride}'
    )
    img_path = os.path.join(dataset_path, segment, f'{img_id}')

    label = np.load(os.path.join(img_path, 'label.npy'), allow_pickle=True)
    X0 = np.load(os.path.join(img_path, 'X0.npy'), allow_pickle=True)
    X1 = np.load(os.path.join(img_path, 'X1.npy'), allow_pickle=True)
    X2 = np.load(os.path.join(img_path, 'X2.npy'), allow_pickle=True)

    return {
        'label': label,
        'X0': X0,
        'X1': X1,
        'X2': X2,
    }


def load_sc_struct(struct_path):
    B1 = scipy.sparse.load_npz(os.path.join(struct_path, 'B1.npz'))
    B2 = scipy.sparse.load_npz(os.path.join(struct_path, 'B2.npz'))
    L0 = scipy.sparse.load_npz(os.path.join(struct_path, 'L0.npz'))
    L1 = scipy.sparse.load_npz(os.path.join(struct_path, 'L1.npz'))
    L2 = scipy.sparse.load_npz(os.path.join(struct_path, 'L2.npz'))
    B2D3 = scipy.sparse.load_npz(os.path.join(struct_path, 'B2D3.npz'))
    D2B1TD1inv = scipy.sparse.load_npz(
        os.path.join(struct_path, 'D2B1TD1inv.npz')
    )
    D1invB1 = scipy.sparse.load_npz(os.path.join(struct_path, 'D1invB1.npz'))
    B2TD2inv = scipy.sparse.load_npz(os.path.join(struct_path, 'B2TD2inv.npz'))

    return {
        'B1': B1,
        'B2': B2,
        'L0': L0,
        'L1': L1,
        'L2': L2,
        'B2D3': B2D3,
        'D2B1TD1inv': D2B1TD1inv,
        'D1invB1': D1invB1,
        'B2TD2inv': B2TD2inv,
    }


def scipy_csr_to_torch_coo(mtx):
    coo = coo_matrix(mtx)

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


# stratified sampling for features
def stratified_sample(
    feature_dict,
    test_pct,
):
    targets = np.unique([value for value in feature_dict.values()])
    target_key_lists = [
        [
            key
            for key, value in feature_dict.items()
            if (
                value == target
            )
        ]
        for target in targets
    ]

    for key_list in target_key_lists:
        np.random.shuffle(key_list)

    keys_train = []
    keys_test = []

    for key_list in target_key_lists:
        keys_train += key_list[int(len(key_list)*test_pct):]
        keys_test += key_list[:int(len(key_list)*test_pct)]

    return keys_train, keys_test


def extract_sc_batch_info(
    sc_batch,
    sc_struct,
    device,
):
    X0 = np.concatenate(
        [
            sc['X0']
            for sc
            in sc_batch
        ],
        axis=0
    )

    X1 = np.concatenate(
        [
            sc['X1']
            for sc
            in sc_batch
        ],
        axis=0
    )

    X2 = np.concatenate(
        [
            sc['X2']
            for sc
            in sc_batch
        ],
        axis=0
    )

    target = np.array([
        sc['label'][0]
        for sc
        in sc_batch
    ])
    target = torch.from_numpy(target).to(device)

    X0 = torch.from_numpy(X0).float().to(device)
    X1 = torch.from_numpy(X1).float().to(device)
    X2 = torch.from_numpy(X2).float().to(device)

    num_nodes = [
        sc['X0'].shape[0]
        for sc
        in sc_batch
    ]

    num_edges = [
        sc['X1'].shape[0]
        for sc
        in sc_batch
    ]

    num_triangles = [
        sc['X2'].shape[0]
        for sc
        in sc_batch
    ]

    L0 = [
        sc_struct['L0']
        for _
        in sc_batch
    ]

    L1 = [
        sc_struct['L1']
        for _
        in sc_batch
    ]

    L2 = [
        sc_struct['L2']
        for _
        in sc_batch
    ]

    D1invB1 = [
        sc_struct['D1invB1']
        for _
        in sc_batch
    ]

    D2B1TD1inv = [
        sc_struct['D2B1TD1inv']
        for _
        in sc_batch
    ]

    B2TD2inv = [
        sc_struct['B2TD2inv']
        for _
        in sc_batch
    ]

    B2D3 = [
        sc_struct['B2D3']
        for _
        in sc_batch
    ]

    L0 = scipy.sparse.block_diag(L0)
    L1 = scipy.sparse.block_diag(L1)
    L2 = scipy.sparse.block_diag(L2)
    D1invB1 = scipy.sparse.block_diag(D1invB1)
    D2B1TD1inv = scipy.sparse.block_diag(D2B1TD1inv)
    B2TD2inv = scipy.sparse.block_diag(B2TD2inv)
    B2D3 = scipy.sparse.block_diag(B2D3)

    L0 = scipy_csr_to_torch_coo(L0).to(device)
    L1 = scipy_csr_to_torch_coo(L1).to(device)
    L2 = scipy_csr_to_torch_coo(L2).to(device)
    D1invB1 = scipy_csr_to_torch_coo(D1invB1).to(device)
    D2B1TD1inv = scipy_csr_to_torch_coo(D2B1TD1inv).to(device)
    B2TD2inv = scipy_csr_to_torch_coo(B2TD2inv).to(device)
    B2D3 = scipy_csr_to_torch_coo(B2D3).to(device)

    return {
        'X0': X0,
        'X1': X1,
        'X2': X2,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_triangles': num_triangles,
        'L0': L0,
        'L1': L1,
        'L2': L2,
        'D1invB1': D1invB1,
        'D2B1TD1inv': D2B1TD1inv,
        'B2TD2inv': B2TD2inv,
        'B2D3': B2D3,
        'target': target,
    }


def shuffle_img_id_stratified(id_dict):
    for value in id_dict.values():
        random.shuffle(value)


def interleave_lists(lists):
    return [x for x in chain(*zip_longest(*lists)) if x is not None]


def augment_config(config):
    n_zero_cells_in_row = int(
        np.floor(
            (28 - config['kernel_size'])/config['stride']
        ) + 1
    )
    config['network']['n_zero_cells'] = n_zero_cells_in_row**2
    config['network']['n_one_cells'] = (
        2*(2*n_zero_cells_in_row - 1)*(n_zero_cells_in_row - 1)
    )
    config['network']['n_two_cells'] = (
        4*(n_zero_cells_in_row - 1)*(n_zero_cells_in_row - 1)
    )

    # config['project_dir'] = (
    #     f'mnist_sccnn_ocf_'
    #     f'{config["one_cell_features"]}_tcf_'
    #     f'{config["two_cell_features"]}'
    # )

    config['network']['num_node_feats'] = int(config['kernel_size']**2)

    if config['one_cell_features'] is True:
        config['network']['num_edge_feats'] = int(
            config['network']['num_node_feats']*2
        )
    else:
        config['network']['num_edge_feats'] = 1

    if config['two_cell_features'] is True:
        config['network']['num_triangle_feats'] = int(
            config['network']['num_node_feats']*3
        )
    else:
        config['network']['num_triangle_feats'] = 1

    if config['conv_type'] == 'no_conv':
        F_nodes = 0
        F_edges = 0
        F_triangles = 0
    elif config['conv_type'] == 'graph_conv':
        F_nodes = config['network']['conv1']['output_node_dim']
        F_edges = config['network']['conv1']['output_edge_dim']
        F_triangles = config['network']['conv1']['output_triangle_dim']
    elif config['conv_type'] == 'sc_conv':
        F_nodes = config['network']['conv1']['output_node_dim']
        F_edges = config['network']['conv1']['output_edge_dim']
        F_triangles = config['network']['conv1']['output_triangle_dim']

    if config['network']['orig_features'] is True:
        F_nodes += config['network']['num_node_feats']
        F_edges += config['network']['num_edge_feats']
        F_triangles += config['network']['num_triangle_feats']

    config['network']['F_nodes'] = F_nodes
    config['network']['F_edges'] = F_edges
    config['network']['F_triangles'] = F_triangles

    config['network']['conv1d_nodes_1']['kernel_size'] = config[
        'network'
    ]['F_nodes']
    config['network']['conv1d_edges_1']['kernel_size'] = config[
        'network'
    ]['F_edges']
    config['network']['conv1d_triangles_1']['kernel_size'] = config[
        'network'
    ]['F_triangles']

    config['network']['conv1d_nodes_1']['stride'] = config[
        'network'
    ]['F_nodes']
    config['network']['conv1d_edges_1']['stride'] = config[
        'network'
    ]['F_edges']
    config['network']['conv1d_triangles_1']['stride'] = config[
        'network'
    ]['F_triangles']

    config['network']['conv1d_nodes_2']['in_channels'] = config[
        'network'
    ]['conv1d_nodes_1']['out_channels']
    config['network']['conv1d_edges_2']['in_channels'] = config[
        'network'
    ]['conv1d_edges_1']['out_channels']
    config['network']['conv1d_triangles_2']['in_channels'] = config[
        'network'
    ]['conv1d_triangles_1']['out_channels']

    if config['conv_type'] == 'no_conv':
        config['network']['mlp_out_conv1d']['input_size'] = (
            config['network']['n_zero_cells']
            *
            config['network']['conv1d_nodes_1']['out_channels']
        )
    elif config['conv_type'] == 'graph_conv':
        config['network']['mlp_out_conv1d']['input_size'] = (
            config['network']['n_zero_cells']
            *
            config['network']['conv1d_nodes_1']['out_channels']
        )
    elif config['conv_type'] == 'sc_conv':
        config['network']['mlp_out_conv1d']['input_size'] = (
            config['network']['k_nodes']*config[
                'network'
            ]['conv1d_nodes_1']['out_channels']
            +
            config['network']['k_edges']*config[
                'network'
            ]['conv1d_edges_1']['out_channels']
            +
            config['network']['k_triangles']*config[
                'network'
            ]['conv1d_triangles_1']['out_channels']
        )

    return config
