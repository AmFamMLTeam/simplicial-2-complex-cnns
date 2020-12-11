import argparse
import numpy as np
import os
# import sys
import json
import yaml
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from Networks import SCNet, GraphNet, ConvNet, standardize
import visdom
from collections import defaultdict
from utils import (
    connect_to_db,
    load_sc_data,
    load_sc_struct,
    extract_sc_batch_info,
    shuffle_img_id_stratified,
    interleave_lists,
    augment_config
)
from config import constants


home = os.path.expanduser('~')
# sys.path.append(os.path.join(home, 'sccnn-mnist-clfn'))

parser = argparse.ArgumentParser()
parser.add_argument('--config_filepath', type=str)
args = parser.parse_args()

with open(args.config_filepath, 'rb') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = augment_config(config)

if not os.path.exists(constants.output_dir_path):
    os.mkdir(constants.output_dir_path)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

vis = visdom.Visdom(server='http://10.160.198.77', port=8888)

con = connect_to_db(constants.db_path)
cur = con.cursor()

query = '''
INSERT INTO experiment (
    epochs,
    learning_rate
)
VALUES
(
    ?,
    ?
)
'''
cur.execute(
    query,
    (
        config['epochs'],
        config['learning_rate']
    )
)

experiment_id = cur.lastrowid
con.commit()
con.close()

experiment_path = os.path.join(constants.output_dir_path, f'{experiment_id}')
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

config_dict_path = os.path.join(experiment_path, 'config.json')
with open(config_dict_path, 'w') as f:
    json.dump(config, f, indent=4)

project_dir = os.path.join(home, 'mount', config['project_dir'])
data_path = os.path.join(
    project_dir,
    f"ksize_{config['kernel_size']}_stride_{config['stride']}"
)

train_filepath = os.path.join(data_path, 'train')
test_filepath = os.path.join(data_path, 'test')

# initialize confusion matrix heatmap plot
num_test = len(os.listdir(test_filepath))
results_dict = {
    'predictions': np.zeros(shape=(num_test,), dtype=float),
    'truth': np.zeros(shape=(num_test,), dtype=float)
}

cfn_mtx = confusion_matrix(
    y_true=np.array(results_dict['truth']),
    y_pred=np.array(results_dict['predictions']),
)

cfn_mtx_path = os.path.join(experiment_path, 'confusion_matrix.npy')
np.save(file=cfn_mtx_path, arr=cfn_mtx)

test_heatmap = vis.heatmap(
    X=cfn_mtx,
    opts=dict(
        # title=f"SCNet MNIST. Epochs: {config['epochs']}"
        # f" lr: {config['learning_rate']} bs: {config['batch_size']}",
        title=f"conv_type: {config['conv_type']}, ",
    )
)

if config['conv_type'] == 'no_conv':
    scnet = ConvNet(params=config['network']).to(device)
elif config['conv_type'] == 'graph_conv':
    scnet = GraphNet(params=config['network']).to(device)
elif config['conv_type'] == 'sc_conv':
    scnet = SCNet(params=config['network']).to(device)

losses = np.zeros(shape=(config['epochs'],), dtype=float)
losses_normalized = np.zeros(shape=(config['epochs'],), dtype=float)
losses_normalized_plot = vis.line(
    Y=losses_normalized,
    X=np.linspace(start=1, stop=len(losses_normalized), num=len(losses_normalized)),
    name='losses_normalized',
    opts=dict(
        # title=f"SCNet MNIST Loss Norm. Epochs: {config['epochs']}"
        # f" lr: {config['learning_rate']}, nspls: {config['n_samples']},"
        # f"bs: {config['batch_size']}",
        title=f"conv_type: {config['conv_type']},",
        xlabel='epoch'
    )
)

error_rate_list = []

train_target_img_id_arr_dict = defaultdict(lambda:[])
for img_id in [iid for iid in os.listdir(train_filepath) if not iid.startswith('.')]:
    target_path = os.path.join(train_filepath, f'{img_id}', 'label.npy')
    target = np.load(target_path)
    target = target[0]
    train_target_img_id_arr_dict[target].append(img_id)

shuffle_img_id_stratified(train_target_img_id_arr_dict)

img_ids_train = interleave_lists(
    [
        value
        for value
        in train_target_img_id_arr_dict.values()
    ]
)
img_ids_train = img_ids_train[:config['n_train']]

train_target_img_id_arr_dict = defaultdict(lambda: [])
for img_id in img_ids_train:
    target_path = os.path.join(train_filepath, f'{img_id}', 'label.npy')
    target = np.load(target_path)
    target = target[0]
    train_target_img_id_arr_dict[target].append(img_id)

img_train_id_dict = {}
n_images_loaded_train = 1
for img_id in img_ids_train:
    img_train_id_dict[img_id] = load_sc_data(
        project_dir=project_dir,
        segment='train', # either 'train' or 'test'
        kernel_size=config['kernel_size'],
        stride=config['stride'],
        img_id=img_id,
    )
    with open(os.path.join(home, 'n_images_loaded_train.txt'), 'w') as f:
        f.write(f'num images loaded train: {n_images_loaded_train}')
    n_images_loaded_train += 1


test_target_img_id_arr_dict = defaultdict(lambda:[])
for img_id in [iid for iid in os.listdir(test_filepath) if not iid.startswith('.')]:
    target_path = os.path.join(test_filepath, f'{img_id}', 'label.npy')
    target = np.load(target_path)
    target = target[0]
    test_target_img_id_arr_dict[target].append(img_id)

shuffle_img_id_stratified(test_target_img_id_arr_dict)
img_ids_test = interleave_lists(
    [
        value
        for value
        in test_target_img_id_arr_dict.values()
    ]
)
img_ids_test = img_ids_test[:config['n_test']]

img_test_id_dict = {}
n_images_loaded_test = 1
for img_id in img_ids_test:
    img_test_id_dict[img_id] = load_sc_data(
        project_dir=project_dir,
        segment='test', # either 'train' or 'test'
        kernel_size=config['kernel_size'],
        stride=config['stride'],
        img_id=img_id,
    )
    with open(os.path.join(home, 'n_images_loaded_test.txt'), 'w') as f:
        f.write(f'num images loaded test: {n_images_loaded_test}')
    n_images_loaded_test += 1

sc_struct_path = os.path.join(data_path, 'sc_struct')
sc_struct = load_sc_struct(sc_struct_path)

optimizer = optim.Adam(params=scnet.parameters(), lr=config['learning_rate'])
loss_function = nn.CrossEntropyLoss()

for epoch in range(config['epochs'] + 1):
    shuffle_img_id_stratified(train_target_img_id_arr_dict)

    img_ids_train = interleave_lists(
        [
            value
            for value
            in train_target_img_id_arr_dict.values()
        ]
    )

    for batch_index_start in range(0, config['n_samples'], config['batch_size']):
        sc_batch = [
            img_train_id_dict[img_id]
            for img_id
            in img_ids_train[batch_index_start:batch_index_start+config['batch_size']]
        ]
        batch_info = extract_sc_batch_info(sc_batch, sc_struct, device)

        output = scnet(
            L0=batch_info['L0'],
            L1=batch_info['L1'],
            L2=batch_info['L2'],
            D1invB1=batch_info['D1invB1'],
            D2B1TD1inv=batch_info['D2B1TD1inv'],
            B2TD2inv=batch_info['B2TD2inv'],
            B2D3=batch_info['B2D3'],
            X0=standardize(batch_info['X0']),
            X1=standardize(batch_info['X1']),
            X2=standardize(batch_info['X2']),
            num_nodes=batch_info['num_nodes'],
            num_edges=batch_info['num_edges'],
            num_triangles=batch_info['num_triangles'],
        )
        # output = output.unsqueeze(0)
        loss = loss_function(output, batch_info['target'])

        # update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = (
            loss.cpu().data.numpy().reshape(1)[0].item()
            *
            float(config['batch_size'])
        )
        losses[epoch] += loss_value
        losses_normalized[epoch] += loss_value/float(config['n_samples'])

    # for each epoch
    losses_filepath = os.path.join(experiment_path, 'losses.npy')
    losses_normalized_filepath = os.path.join(experiment_path, 'losses_normalized.npy')
    np.save(
        file=losses_filepath,
        arr=losses
    )
    np.save(
        file=losses_normalized_filepath,
        arr=losses_normalized
    )
    vis.line(
        Y=losses_normalized,
        X=np.linspace(start=1, stop=len(losses_normalized), num=len(losses_normalized)),
        win=losses_normalized_plot,
        update='replace',
    )

    # periodically evaluate model
    if epoch % 20 == 0 and epoch > 0:
        predictions = []
        y_true = []

        for img_id in img_ids_test:
            sc_batch = [img_test_id_dict[img_id]]
            batch_info = extract_sc_batch_info(sc_batch, sc_struct, device)

            y_true.append(int(batch_info['target'][0]))

            output = scnet(
                L0=batch_info['L0'],
                L1=batch_info['L1'],
                L2=batch_info['L2'],
                D1invB1=batch_info['D1invB1'],
                D2B1TD1inv=batch_info['D2B1TD1inv'],
                B2TD2inv=batch_info['B2TD2inv'],
                B2D3=batch_info['B2D3'],
                X0=standardize(batch_info['X0']),
                X1=standardize(batch_info['X1']),
                X2=standardize(batch_info['X2']),
                num_nodes=batch_info['num_nodes'],
                num_edges=batch_info['num_edges'],
                num_triangles=batch_info['num_triangles'],
            )
            _, predicted = torch.max(output, 1)
            predictions.extend(predicted.cpu().numpy().tolist())

        results_dict = {
            'predictions': predictions,
            'truth': y_true
        }

        results_dict_path = os.path.join(experiment_path, 'results_dict.pickle')
        with open(results_dict_path, 'wb') as f:
            pickle.dump(results_dict, f)

        cfn_mtx = confusion_matrix(
            y_true=np.array(results_dict['truth']),
            y_pred=np.array(results_dict['predictions']),
        )
        np.save(file=cfn_mtx_path, arr=cfn_mtx)

        acc_score  = accuracy_score(
            y_true=results_dict['truth'],
            y_pred=results_dict['predictions']
        )
        error_rate = (1 - acc_score)*100
        error_rate_list.append(
            {
                'epoch': epoch,
                'error_rate': error_rate,
            }
        )
        error_rate_list_path = os.path.join(experiment_path, 'error_rate_list.pickle')
        with open(error_rate_list_path, 'wb') as f:
            pickle.dump(error_rate_list, f)
        test_heatmap = vis.heatmap(
            X=cfn_mtx,
            win=test_heatmap,
            opts=dict(
                # title=f"SCNet MNIST. Epochs: {epoch}/{config['epochs']}"
                # f" lr: {config['learning_rate']} "
                # f"er: {error_rate:.2f} bs: {config['batch_size']},"
                # f"nspls: {config['n_samples']}",
                title=f"{epoch}/{config['epochs']} "
                f"er: {error_rate:.2f}"
                f"conv_type: {config['conv_type']}, ",
            )
        )

    if epoch % 20 == 0 and epoch > 0:
        intermediate_model_filepath = os.path.join(
            experiment_path,
            'intermediate_scnet_mnist.pt'
        )
        torch.save(
            scnet.state_dict(),
            intermediate_model_filepath
        )

results_dict_path = os.path.join(experiment_path, 'results_dict.pickle')
with open(results_dict_path, 'wb') as f:
    pickle.dump(results_dict, f)

error_rate_list_path = os.path.join(experiment_path, 'error_rate_list.pickle')
with open(error_rate_list_path, 'wb') as f:
    pickle.dump(error_rate_list, f)

final_model_filepath = os.path.join(experiment_path, 'final_model_scnet_mnist.pt')
torch.save(
    scnet.state_dict(),
    final_model_filepath
)
