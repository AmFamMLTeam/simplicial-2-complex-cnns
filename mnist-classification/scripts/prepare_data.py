import argparse
import numpy as np
import os
import pickle
import scipy
from mnist import MNIST
home = os.path.expanduser('~')


parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', type=str)
parser.add_argument('--kernel_size', type=int)
parser.add_argument('--stride', type=int)
args = parser.parse_args()


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
    D2B1TD1inv = scipy.sparse.load_npz(os.path.join(struct_path, 'D2B1TD1inv.npz'))
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


def make_one_cells(zero_cells):
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
                if 'features' in zero_cells[a].keys() and 'features' in zero_cells[b].keys():
                    one_cell_dict = {
                        'one_cell': (zero_cell1[0], zero_cell2[0]),
                        'features': np.concatenate(
                            (zero_cell1_features, zero_cell2_features),
                            axis=0
                        )
                    }
                else:
                    one_cell_dict = {'one_cell': (zero_cell1[0], zero_cell2[0])}
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
        zero_cell = zero_cells[a]['zero_cell'][0]
        nbrs = sorted(zero_cells[a]['nbrs'])
        for index, nbr1 in enumerate(nbrs):
            for nbr2 in nbrs[index+1:]:
                if nbr2 in set(zero_cells[nbr1]['nbrs']):
                    if ('features' in zero_cells[a].keys()
                    and 'features' in zero_cells[nbr1].keys()
                    and 'features' in zero_cells[nbr2].keys()):
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
                        two_cells[two_cell_index] = {'two_cell': (a, nbr1, nbr2)}
                        two_cell_index += 1

    return two_cells


def preprocess_mnist_images(
    images,
    project_dir,
    segment, # either 'train' or 'test'
    kernel_size,
    stride,
):
    dataset_path = os.path.join(project_dir, f'ksize_{kernel_size}_stride_{stride}')
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    path = os.path.join(dataset_path, segment)
    if not os.path.isdir(path):
        os.mkdir(path)

    imgs = tqdm(images, desc='mnist')
    for index, image in enumerate(imgs):
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
        n_zero_cells = n_zero_cells_in_row**2
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
    dataset_path = os.path.join(project_dir, f'ksize_{kernel_size}_stride_{stride}')
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

    B1 = np.zeros(shape=(n_zero_cells, n_one_cells), dtype='float')
    B2 = np.zeros(shape=(n_one_cells, n_two_cells), dtype='float')

    for key, value in one_cells.items():
        one_cell = value['one_cell']
        B1[:,key][int(one_cell[0])] = 1
        B1[:,key][int(one_cell[1])] = -1

    for key, value in two_cells.items():
        two_cell = value['two_cell']
        B2[:,key][int(two_cell[0])] = 1
        B2[:,key][int(two_cell[1])] = -1
        B2[:,key][int(two_cell[2])] = 1

    L0 = B1@B1.T
    B1_sum = np.abs(B1).sum(axis=1)
    D0 = np.diag(B1_sum)
    B1_sum_inv = 1./B1_sum
    B1_sum_inv[np.isinf(B1_sum_inv) | np.isneginf(B1_sum_inv)] = 0
    D0_inv = np.diag(B1_sum_inv)
    # noramlize L0
    L0 = D0_inv@L0
    # equivalent of \tilde{D}^{-1}\tilde{A}
    L0factor = (-1.)*np.linalg.inv(D0_inv + np.identity(n=D0_inv.shape[0]))
    L0bias = np.identity(n=D0.shape[0])
    L0 = L0factor@L0 + L0bias

    D1_inv = np.diag(B1_sum_inv*0.5)
    D2 = np.max((np.diag(np.abs(B2).sum(axis=1)), np.identity(n=len(one_cells))), axis=0)

#     normalize L1
    D2inv = np.linalg.inv(D2)
    L1 = D2@B1.T@D1_inv@B1 + (1/3.)*B2@B2.T@D2inv
    L1factor = (-1/2.)*np.identity(n=B1.shape[1])
    L1bias = np.identity(n=B1.shape[1])
    L1 = L1factor@(L1) + L1bias

    D2B1TD1inv = (1/np.sqrt(2.))*D2@B1.T@D1_inv
    B2D3 = (1/3.)*B2
    L2 = (1/3.)*B2.T@D2inv@B2 # same as L2 in this case
    D1invB1 = (1/np.sqrt(2.))*D1_inv@B1
    B2TD2inv = B2.T@D2inv


    B1_path = os.path.join(sc_struct_path, 'B1')
    B1_csr = scipy.sparse.csr_matrix(B1)
    scipy.sparse.save_npz(
        file=B1_path,
        matrix=B1_csr,
    )

    B2_path = os.path.join(sc_struct_path, 'B2')
    B2_csr = scipy.sparse.csr_matrix(B2)
    scipy.sparse.save_npz(
        file=B2_path,
        matrix=B2_csr,
    )

    L0_path = os.path.join(sc_struct_path, 'L0')
    L0_csr = scipy.sparse.csr_matrix(L0)
    scipy.sparse.save_npz(
        file=L0_path,
        matrix=L0_csr,
    )

    L1_path = os.path.join(sc_struct_path, 'L1')
    L1_csr = scipy.sparse.csr_matrix(L1)
    scipy.sparse.save_npz(
        file=L1_path,
        matrix=L1_csr,
    )

    L2_path = os.path.join(sc_struct_path, 'L2')
    L2_csr = scipy.sparse.csr_matrix(L2)
    scipy.sparse.save_npz(
        file=L2_path,
        matrix=L2_csr,
    )

    B2D3_path = os.path.join(sc_struct_path, 'B2D3')
    B2D3_csr = scipy.sparse.csr_matrix(B2D3)
    scipy.sparse.save_npz(
        file=B2D3_path,
        matrix=B2D3_csr,
    )

    D2B1TD1inv_path = os.path.join(sc_struct_path, 'D2B1TD1inv')
    D2B1TD1inv_csr = scipy.sparse.csr_matrix(D2B1TD1inv)
    scipy.sparse.save_npz(
        file=D2B1TD1inv_path,
        matrix=D2B1TD1inv_csr,
    )

    D1invB1_path = os.path.join(sc_struct_path, 'D1invB1')
    D1invB1_csr = scipy.sparse.csr_matrix(D1invB1)
    scipy.sparse.save_npz(
        file=D1invB1_path,
        matrix=D1invB1_csr,
    )

    B2TD2inv_path = os.path.join(sc_struct_path, 'B2TD2inv')
    B2TD2inv_csr = scipy.sparse.csr_matrix(B2TD2inv)
    scipy.sparse.save_npz(
        file=B2TD2inv_path,
        matrix=B2TD2inv_csr,
    )


if __name__ == '__main__':
    mnist_path = os.path.join(home, 'mnist')
    mndata = MNIST(mnist_path)
    project_dir = os.path.join(home, 'mount', args.project_dir)

    if not os.path.isdir(project_dir):
        os.mkdir(project_dir)

    images, labels = mndata.load_training()
    preprocess_mnist_images(
        images=images,
        project_dir=args.project_dir,
        segment='train',
        kernel_size=args.kernel_size,
        stride=args.stride,
    )

    images, labels = mndata.load_testing()
    preprocess_mnist_images(
        images=images,
        project_dir=args.project_dir,
        segment='test',
        kernel_size=args.kernel_size,
        stride=args.stride,
    )

    form_sc_structure(
        project_dir=args.project_dir,
        kernel_size=args.kernel_size,
        stride=args.stride,
    )
