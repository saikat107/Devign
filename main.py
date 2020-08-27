import argparse
import os
import pickle
import sys

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel
from trainer import train
from utils import tally_param, debug


if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset for experiment.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the parser')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default=None)
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default=None)
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default=None)
    args = parser.parse_args()

    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, 'processed.bin')
    if os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    else:
        dataset = DataSet(train_src=os.path.join(input_dir, 'train_GGNNinput.json'),
                      valid_src=os.path.join(input_dir, 'valid_GGNNinput.json'),
                      test_src=os.path.join(input_dir, 'test_GGNNinput.json'),
                      batch_size=128, n_ident=args.node_tag, g_ident=args.graph_tag, l_ident=args.label_tag)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()

    model = DevignModel(input_dim=dataset.feature_size, output_dim=200,
                        num_steps=6, max_edge_types=dataset.max_edge_type)

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    model.cuda()
    loss_function = BCELoss(reduction='sum')
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    train(model=model, dataset=dataset, max_steps=1000000, dev_every=128, loss_function=loss_function, optimizer=optim,
          save_path=model_dir+'/DevignModel', max_patience=100, log_every=None)
