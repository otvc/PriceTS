import sys

sys.path.insert(0, './src/')

import os

import yaml

import pandas as pd

import torch
from torch import nn
from torch import optim 
from torch.utils.data import DataLoader, Dataset

from data.prepare_data import MilkTSDataset

from training_instruments import train_CatEmbLSTM, inference_CatEmbLSTM

from models import CatEmbLSTM

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_config_params(path_to_config = 'config/inference_CatEmbLSTM.yaml'):
    with open(path_to_config, 'r') as f:
        params = yaml.load(f)
    print(params)
    return params

def load_dataset(path_to_inference):
    inference_dataset = torch.load(path_to_inference)
    return inference_dataset

def choose_model(params):
    model = torch.load(params['model']['path'])
    return model

if __name__ == '__main__':
    params = load_config_params()

    batch_size = params['dataset']['batch_size']#256
    inference_dataset = load_dataset(params['dataset']['path_to_input'], params['dataset']['path_to_output'])
    inference_dataloader = DataLoader(inference_dataset,  batch_size = batch_size, num_workers = params['dataset']['num_workers'])

    model = choose_model(params)
    predicted_values = inference_CatEmbLSTM(model, inference_dataloader)
    
    output = pd.DataFrame({params['target']: predicted_values.detach().cpu().numpy()})
    output.to_csv(params['path_to_output'])