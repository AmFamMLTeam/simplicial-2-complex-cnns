import argparse
import numpy as np
import os
import pickle
import scipy
import yaml
from scipy import sparse
from scipy.sparse import csr_matrix
from mnist import MNIST
from config import constants


parser = argparse.ArgumentParser()
parser.add_argument('--config_filepath', type=str)
args = parser.parse_args()

with open(args.config_filepath, 'rb') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


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
    D2B1TD1inv = scipy.sparse.load_npz(os.path.join(
        struct_path,
        'D2B1TD1inv.npz',
    ))
    D1invB1 = scipy.sparse.load_npz(os.path.join(
        struct_path,
        'D1invB1.npz',
    ))
    B2TD2inv = scipy.sparse.load_npz(os.path.join(
        struct_path,
        'B2TD2inv.npz',
    ))

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


def make_zero_cells(
    kernel_size,
    stride,
    img_arr=None,
):
    zero_cells = {}
    zero_cell_index = 0
    for x in range(0, 28 - kernel_size + 1,  stride):
        for y in range(0, 28 - kernel_size + 1, stride):
            if img_arr is not None:
                zero_cells[zero_cell_index] = {
                    'zero_cell': (zero_cell_index,),
                    'x_y_coords': (x, y),
                    'features': img_arr[
                        x:x+kernel_size, y:y+kernel_size
                    ].flatten(),
                    'nbrs': [],
                }
                zero_cell_index += 1
            else:
                zero_cells[zero_cell_index] = {
                    'zero_cell': (zero_cell_index,),
                    'x_y_coords': (x, y),
                    'nbrs': [],
                }
                zero_cell_index += 1

    return zero_cells


def make_one_cells(zero_cells, stride):
    one_cells = {}
    one_cell_index = 0
    for a in zero_cells.keys():
        (x1, y1) = zero_cells[a]['x_y_coords']
        zero_cell1 = zero_cells[a]['zero_cell']
        if 'features' in zero_cells[a].keys():
            zero_cell1_features = zero_cells[a]['features']
        for b in zero_cells.keys():
            (x2, y2) = zero_cells[b]['x_y_coords']
            zero_cell2 = zero_cells[b]['zero_cell']
            if 'features' in zero_cells[b].keys():
                zero_cell2_features = zero_cells[b]['features']
            if zero_cell1[0] < zero_cell2[0]:
                if (
                    'features' in zero_cells[a].keys()
                    and
                    'features' in zero_cells[b].keys()
                ):
                    one_cell_dict = {
                        'one_cell': (zero_cell1[0], zero_cell2[0]),
                        'features': np.concatenate(
                            (zero_cell1_features, zero_cell2_features),
                            axis=0
                        )
                    }
                else:
                    one_cell_dict = {
                        'one_cell':
                        (zero_cell1[0], zero_cell2[0])
                    }
                if y1 == y2 and np.abs(x1 - x2) == stride:
                    one_cells[one_cell_index] = one_cell_dict
                    one_cell_index += 1
                elif x1 == x2 and np.abs(y1 - y2) == stride:
                    one_cells[one_cell_index] = one_cell_dict
                    one_cell_index += 1
                elif np.abs(x1 - x2) == stride and np.abs(y1 - y2) == stride:
                    one_cells[one_cell_index] = one_cell_dict
                    one_cell_index += 1

    return one_cells


def make_two_cells(zero_cells):
    two_cells = {}
    two_cell_index = 0
    for a in zero_cells.keys():
        # zero_cell = zero_cells[a]['zero_cell'][0]
        nbrs = sorted(zero_cells[a]['nbrs'])
        for index, nbr1 in enumerate(nbrs):
            for nbr2 in nbrs[index+1:]:
                if nbr2 in set(zero_cells[nbr1]['nbrs']):
                    if (
                        'features' in zero_cells[a].keys()
                        and
                        'features' in zero_cells[nbr1].keys()
                        and
                        'features' in zero_cells[nbr2].keys()
                    ):
                        two_cells[two_cell_index] = {
                            'two_cell': (a, nbr1, nbr2),
                            'features': np.concatenate(
                                (
                                    zero_cells[a]['features'],
                                    zero_cells[nbr1]['features'],
                                    zero_cells[nbr2]['features'],
                                ),
                                axis=0
                            )
                        }
                        two_cell_index += 1
                    else:
                        two_cells[two_cell_index] = {
                            'two_cell': (a, nbr1, nbr2)
                        }
                        two_cell_index += 1

    return two_cells


def preprocess_mnist_images(
    images,
    project_dir,
    segment,  # either 'train' or 'test'
    kernel_size,
    stride,
):
    dataset_path = os.path.join(
        project_dir,
        f'ksize_{kernel_size}_stride_{stride}',
    )
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    path = os.path.join(dataset_path, segment)
    if not os.path.isdir(path):
        os.mkdir(path)

    for index, image in enumerate(images):
        img_path = os.path.join(path, f'{index}')
        if not os.path.isdir(img_path):
            os.mkdir(img_path)

        label = np.array([labels[index]])

        np.save(file=os.path.join(img_path, 'label.npy'), arr=label)

        img_arr = np.array(image).reshape(28, 28)

        # make zero cells
        zero_cells = make_zero_cells(
            kernel_size=kernel_size,
            stride=stride,
            img_arr=img_arr,
        )

        zero_cells_path = os.path.join(img_path, 'zero_cells.pickle')
        with open(zero_cells_path, 'wb') as f:
            pickle.dump(zero_cells, f)

        n_zero_cells_in_row = int(np.floor((28 - kernel_size)/stride) + 1)
        # n_zero_cells = n_zero_cells_in_row**2
        n_one_cells = 2*(2*n_zero_cells_in_row - 1)*(n_zero_cells_in_row - 1)
        n_two_cells = 4*(n_zero_cells_in_row - 1)*(n_zero_cells_in_row - 1)

        X0 = np.array([
            zero_cells[key]['features']
            for key
            in sorted(zero_cells.keys())
        ])
        X1 = np.ones(shape=(n_one_cells, 1))
        X2 = np.ones(shape=(n_two_cells, 1))

        zero_cell_features_path = os.path.join(img_path,  'X0.npy')
        np.save(file=zero_cell_features_path, arr=X0)

        one_cell_features_path = os.path.join(img_path,  'X1.npy')
        np.save(file=one_cell_features_path, arr=X1)

        two_cell_features_path = os.path.join(img_path,  'X2.npy')
        np.save(file=two_cell_features_path, arr=X2)


def form_sc_structure(
    project_dir,
    kernel_size,
    stride,
):
    dataset_path = os.path.join(
        project_dir,
        f'ksize_{kernel_size}_stride_{stride}',
    )
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    sc_struct_path = os.path.join(dataset_path, 'sc_struct')
    if not os.path.isdir(sc_struct_path):
        os.mkdir(sc_struct_path)

    # make zero cells
    zero_cells = make_zero_cells(kernel_size, stride)

    # make one cells
    one_cells = make_one_cells(zero_cells)

    # add neighbors to zero_cells
    for one_cell_vals in one_cells.values():
        (x, y) = one_cell_vals['one_cell']
        zero_cells[x]['nbrs'].append(y)

    # form two cells
    two_cells = make_two_cells(zero_cells)

    n_zero_cells_in_row = int(np.floor((28 - kernel_size)/stride) + 1)
    n_zero_cells = n_zero_cells_in_row**2
    n_one_cells = 2*(2*n_zero_cells_in_row - 1)*(n_zero_cells_in_row - 1)
    n_two_cells = 4*(n_zero_cells_in_row - 1)*(n_zero_cells_in_row - 1)

    one_cell_index_lookup = {
        value['one_cell']: key
        for key, value
        in one_cells.items()
    }

    # make B1
    col_index = []
    row_index = []
    data = []

    for key, value in one_cells.items():
        (x, y) = value['one_cell']
        x, y = int(x), int(y)
        col_index.append(key)
        row_index.append(x)
        data.append(1)

        col_index.append(key)
        row_index.append(y)
        data.append(-1)

    B1 = csr_matrix(
        (
            data,
            (row_index, col_index)
        ),
        shape=(n_zero_cells, n_one_cells),
        dtype=float,
    )

    # make B2
    col_index = []
    row_index = []
    data = []

    for key, value in two_cells.items():
        (x, y, z) = value['two_cell']
        x, y, z = int(x), int(y), int(z)
        col_index.append(key)
        row_index.append(one_cell_index_lookup[(y, z)])
        data.append(1)

        col_index.append(key)
        row_index.append(one_cell_index_lookup[(x, z)])
        data.append(-1)

        col_index.append(key)
        row_index.append(one_cell_index_lookup[(x, y)])
        data.append(1)

    B2 = csr_matrix(
        (
            data,
            (row_index, col_index)
        ),
        shape=(n_one_cells, n_two_cells),
        dtype=float,
    )

    L0 = B1@B1.T
    B1_sum = np.abs(B1).sum(axis=1)
    D0 = sparse.diags(B1_sum.A.reshape(-1), 0)
    B1_sum_inv = 1./B1_sum
    B1_sum_inv[np.isinf(B1_sum_inv) | np.isneginf(B1_sum_inv)] = 0
    D0_inv = sparse.diags(B1_sum_inv.A.reshape(-1), 0)
    L0 = D0_inv@L0
    L0factor = (-1)*sparse.diags((1/(B1_sum_inv + 1)).A.reshape(-1), 0)
    L0bias = sparse.identity(n=D0.shape[0])
    L0 = L0factor@L0 + L0bias

    D1_inv = sparse.diags((B1_sum_inv*0.5).A.reshape(-1), 0)
    D2diag = np.max(
        (
            abs(B2).sum(axis=1).A.reshape(-1),
            np.ones(shape=(n_one_cells))
        ),
        axis=0
    )
    D2 = sparse.diags(D2diag, 0)
    D2inv = sparse.diags(1/D2diag, 0)
    D3 = (1/3.)*sparse.identity(n=B2.shape[1])

    A_1u = D2 - B2@D3@B2.T
    A_1d = D2inv - B1.T@D1_inv@B1
    A_1u_norm = (
        A_1u
        +
        sparse.identity(n=A_1u.shape[0])
    )@sparse.diags(1/(D2.diagonal() + 1), 0)
    A_1d_norm = (
        D2
        +
        sparse.identity(n=D2.shape[0])
    )@(A_1d + sparse.identity(n=A_1d.shape[0]))
    # not really L1, but easy to drop in; normalized adjacency
    L1 = A_1u_norm + A_1d_norm

    B2_sum = abs(B2).sum(axis=1)
    B2_sum_inv = 1/(B2_sum + 1)
    D5inv = sparse.diags(B2_sum_inv, 0)

    A_2d = sparse.identity(n=B2.shape[1]) + B2.T@D5inv@B2
    A_2d_norm = (2*sparse.identity(n=B2.shape[1]))@(
        A_2d
        +
        sparse.identity(n=A_2d.shape[0])
    )
    L2 = A_2d_norm  # normalized adjacency

    B2D3 = B2@D3
    D2B1TD1inv = (1/np.sqrt(2.))*D2@B1.T@D1_inv
    D1invB1 = (1/np.sqrt(2.))*D1_inv@B1
    B2TD2inv = B2.T@D5inv

    B1_path = os.path.join(sc_struct_path, 'B1')
    scipy.sparse.save_npz(
        file=B1_path,
        matrix=B1,
    )

    B2_path = os.path.join(sc_struct_path, 'B2')
    scipy.sparse.save_npz(
        file=B2_path,
        matrix=B2,
    )

    L0_path = os.path.join(sc_struct_path, 'L0')
    scipy.sparse.save_npz(
        file=L0_path,
        matrix=L0,
    )

    L1_path = os.path.join(sc_struct_path, 'L1')
    scipy.sparse.save_npz(
        file=L1_path,
        matrix=L1,
    )

    L2_path = os.path.join(sc_struct_path, 'L2')
    scipy.sparse.save_npz(
        file=L2_path,
        matrix=L2,
    )

    B2D3_path = os.path.join(sc_struct_path, 'B2D3')
    scipy.sparse.save_npz(
        file=B2D3_path,
        matrix=B2D3,
    )

    D2B1TD1inv_path = os.path.join(sc_struct_path, 'D2B1TD1inv')
    scipy.sparse.save_npz(
        file=D2B1TD1inv_path,
        matrix=D2B1TD1inv,
    )

    D1invB1_path = os.path.join(sc_struct_path, 'D1invB1')
    scipy.sparse.save_npz(
        file=D1invB1_path,
        matrix=D1invB1,
    )

    B2TD2inv_path = os.path.join(sc_struct_path, 'B2TD2inv')
    scipy.sparse.save_npz(
        file=B2TD2inv_path,
        matrix=B2TD2inv,
    )


if __name__ == '__main__':
    mnist_path = os.path.join(constants.home, 'mnist')
    mndata = MNIST(mnist_path)

    if not os.path.isdir(config.project_dir):
        os.mkdir(config.project_dir)

    images, labels = mndata.load_training()
    preprocess_mnist_images(
        images=images,
        project_dir=config.project_dir,
        segment='train',
        kernel_size=config.kernel_size,
        stride=config.stride,
    )

    images, labels = mndata.load_testing()
    preprocess_mnist_images(
        images=images,
        project_dir=config.project_dir,
        segment='test',
        kernel_size=config.kernel_size,
        stride=config.stride,
    )

    form_sc_structure(
        project_dir=config.project_dir,
        kernel_size=config.kernel_size,
        stride=config.stride,
    )
