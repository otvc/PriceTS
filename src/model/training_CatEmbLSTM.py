import sys

sys.path.insert(0, './src/')

import os

import yaml

import pandas as pd

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset

from data.prepare_data import MilkTSDataset

from training_instruments import train_CatEmbLSTM

from models import CatEmbLSTM

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_config_params(path_to_config = 'config/CatEmbLSTM.yaml'):
    with open(path_to_config, 'r') as f:
        params = yaml.load(f)
    print(params)
    return params

def load_dataset(path_to_train, path_to_test):
    train_dataset = torch.load(path_to_train)
    test_dataset = torch.load(path_to_test)

    # train_dataset.train_data.to(device)
    # train_dataset.train_labels.to(device)

    # test_dataset.train_data.to(device)
    # test_dataset.train_labels.to(device)

    return train_dataset, test_dataset

def choose_model(params):
    milk_clean = pd.read_csv(params['dataset']['path_to_clean'])


    N_EMB_CLS = milk_clean[params['dataset']['cat_feature']].unique().shape[0]
    EMB_H = params['model']['EMB_H']#32

    NUM_INPUT_SIZE = len(test_dataset[0].keys()) - 2
    LSTM_H = params['model']['LSTM_H']#128
    LSTM_NUM_LAYERS = params['model']['LSTM_NUM_LAYERS']#1

    if params['fine-tune']['use_saved_model'] and os.path.isfile(params['fine-tune']['model_path']):
        model = torch.load(params['fine-tune']['model_path'])
    else:
        model = CatEmbLSTM(NUM_INPUT_SIZE, LSTM_H, LSTM_NUM_LAYERS, N_EMB_CLS, EMB_H).to(device)


    LR = params['model']['LR'] #1e-3

    if params['fine-tune']['use_saved_model'] and os.path.isfile(params['fine-tune']['optimizer_path']):
        optimizer = torch.load(params['fine-tune']['optimizer_path'])
    else:
        if params['optimizer'] == 'Adam':
            BETAS = params['model']['BETAS']#(0.999, 0.999)
            optimizer = optim.Adam(model.parameters(), lr = LR, betas = BETAS)
        elif params['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr = LR)
    
    scheduler = LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.01, total_iters=params['model']['EPOCHS'])
    
    return model, optimizer, scheduler

if __name__ == '__main__':
    params = load_config_params()

    BATCH_SIZE = params['dataset']['BATCH_SIZE']#256
    is_shuffle = params['dataset']['is_shuffle']#True

    train_dataset, test_dataset = load_dataset(params['dataset']['path_to_train'], params['dataset']['path_to_test'])

    train_dataloader = DataLoader(train_dataset,  batch_size = BATCH_SIZE, shuffle = is_shuffle, num_workers = params['dataset']['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = is_shuffle, num_workers = params['dataset']['num_workers'])

    model, optimizer, scheduler = choose_model(params)

    REDUCTION = 'mean'

    if params['loss_type'] == 'L1':
        criterion = nn.L1Loss(reduce=REDUCTION)
    elif params['loss_type'] == 'L2':
        criterion = nn.MSELoss(reduction = REDUCTION)

    EPOCHS = params['model']['EPOCHS']#5

    train_CatEmbLSTM(model, train_dataloader, test_dataloader, optimizer, criterion, scheduler, EPOCHS, device = device, 
                     path_to_stages = params['path_to_save_stages'], model_name = params['name'],)

    torch.save(optimizer, params['path_to_save'] + params['name'] + '_optimizer.pt')
    torch.save(model, params['path_to_save'] + params['name'] + '.pt')