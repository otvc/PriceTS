import yaml
import os
import sys

from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch

import pandas as pd

from  tqdm import tqdm

from models import CatEmbLSTM
from training_instruments import train_CatEmbLSTM, inference_CatEmbLSTM

sys.path.insert(0, 'src/')

from data.prepare_data import MilkTSDataset


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_config_params(path_to_config='config/inference_CatEmbLSTM.yaml'):
    with open(path_to_config, 'r') as f:
        params = yaml.load(f)
    print(params)
    return params


def load_dataset(path_to_inference) -> Dataset:
    inference_dataset = torch.load(path_to_inference)
    return inference_dataset


def choose_model(params):
    model = torch.load(params['model']['path'])
    return model


if __name__ == '__main__':
    params = load_config_params()

    batch_size = params['dataset']['batch_size']  # 256
    inference_dataset = load_dataset(params['dataset']['path_to_input'])
    model = choose_model(params)
    output = None
    for part_dataset in tqdm(inference_dataset.datasets):
        inference_dataloader = DataLoader(part_dataset,
                                          batch_size=batch_size,
                                          num_workers=params['dataset']['num_workers'])
        predicted_values = inference_CatEmbLSTM(model, inference_dataloader)
        part_dataset.data['predicted'] = None
        part_dataset.data.index = range(part_dataset.data.shape[0])
        part_dataset.data.loc[params['dataset']['lag']:,
                               'predicted'] = torch.cat(predicted_values[0]).detach().cpu().numpy()
        output = pd.concat([output, part_dataset.data])

    output.to_csv(params['dataset']['path_to_output'])
