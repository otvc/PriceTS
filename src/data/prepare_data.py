from datetime import datetime
import yaml

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from tqdm import tqdm

class MilkTSDataset(Dataset):
    '''
    Use this Dataset for particular `class_name`, `store`, `item`
    And use ConcatDataset for concatination datasets from several `class_name` and `store`
    
    `params`:
        `class_name`:str: type names which should be choosed (value should be contained in column `class_name` from `data`);
        `store`:int: id of store timeserieses of which we want to create (value should be contained in column `store` from `data`);
        `item`:int: id of product timeserieses of which we want to create (value should be contained in column `item` from `data`);
        `data`:pd.DataFrame: cleaned dataframe;
        `target`:str: target column which contained in `data`;
        `n_prev_days`:int: value of lag;
        `features`:list[str]: which features we should return to output.
    '''
    def __init__(self, class_name:str, store:int, item:int, data:pd.DataFrame,
                 target:str = 'price', n_prev_days:int = 5,
                 features:list[str] = ['cost', 'price', 'sales_units', 'sales_values', 'wasted_units']):
        self.features = features
        self.n_prev_days = n_prev_days
        self.target = target
        
        self.data = data[(data['class_name'] == class_name) & (data['store'] == store) & (data['item'] == item)]
        self.total_count = max(0, len(self.data) - n_prev_days)
    
    def __len__(self):
        return self.total_count
    
    def __getitem__(self, idx):
        output = {'target_price':self.data.iloc[idx + self.n_prev_days][self.target]}
        output = dict(self.data[self.features].iloc[idx:idx + self.n_prev_days].to_dict('list'), **output)
        return output
        

def preparing_dataframe(path_to_data,
                   to_float_point_feat = ['StoreInventory', 'wasted_units', 'wasted_cost'],
                   nan_features = ['sales_cost_x', 'sales_units', 'sales_value', 'wasted_units', 'wasted_cost', 'sales_cost_y'],
                   numeric_features = ['StoreInventory', 'sales_cost_x', 'sales_value', 'price', 'cost', 'sales_cost_y'],
                   cat_features = ['price_zone_&_class_name'],
                   date_feature = 'date',
                   ts_by_features = ['class_name', 'item', 'store']
                   ):
    data = pd.read_csv(path_to_data, sep = ',')

    data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace = True)

    for float_feat in to_float_point_feat:
        data[float_feat] = data[float_feat].str.replace(',','.')
        data[float_feat] = data[float_feat].astype(float)
    
    data[nan_features] = data[nan_features].fillna(0)
    data['price_zone_&_class_name'] = data['price_zone'] + '+' + data['class_name']
    data.drop(columns = ['price_zone', 'class'], inplace = True)
    milk_num_cat, milk_str_cat = pd.factorize(data['price_zone_&_class_name'])
    data['price_zone_&_class_name'] = milk_num_cat

    data['date'] = pd.to_datetime(data['date'])
    data.sort_values(by = 'date', inplace = True)

    needed_features = numeric_features + [date_feature] + cat_features + ts_by_features
    data = data[needed_features].copy()

    data = data[~data.price.isna()]
    data = data[~data.cost.isna()]

    return data, needed_features

def create_dataset_by_concat(data,
                             n_prev_days = 5,
                             numeric_features = ['StoreInventory', 'sales_cost_x', 'sales_value', 'price', 'cost', 'sales_cost_y'],
                             cat_features = ['price_zone_&_class_name']
                   ):
    df_cn_item = data[['item', 'class_name', 'store']].groupby(['class_name', 'store']).agg(set)
    datasets = []
    for cls_store in tqdm(df_cn_item.index):
        for item in df_cn_item.loc[cls_store].values[0]:
            cur_ds = MilkTSDataset(str(cls_store[0]), cls_store[1], 
                                   item, 
                                   data, 
                                   features = numeric_features + cat_features,
                                   n_prev_days = n_prev_days)
            if len(cur_ds) > 0:
                datasets.append(cur_ds)

    dataset = ConcatDataset(datasets)
    return dataset

def train_test_split(data,
                     feat_date = 'date',
                     date_split = datetime(2023,3,1),
                     n_prev_days = 5,
                     numeric_features = ['StoreInventory', 'sales_cost_x', 'sales_value', 'price', 'cost', 'sales_cost_y'],
                     cat_features = ['price_zone_&_class_name']
                     ):

    train_dataset = data[data[feat_date] < date_split]
    test_dataset = data[data[feat_date] >= date_split]

    train_dataset = create_dataset_by_concat(train_dataset, n_prev_days, numeric_features, cat_features)
    test_dataset = create_dataset_by_concat(test_dataset, n_prev_days, numeric_features, cat_features)

    return train_dataset, test_dataset

'''
    `params`:
             
    `return`:
             dict with keys:
             1. path_to_data;
             2. n_prev_days;
             3. date_split;
             4. cat_features;
             5. date_feature;
             6. nan_features;
             7. numeric_features;
             8. ts_by_features;
             9. to_float_point_feat;
             10. path_to_save.
'''
def get_args_from_config(path_to_config = 'config/dataset_v0.yaml'):
    with open(path_to_config, 'r') as f:
        config = yaml.load(f)
    config['date_split'] = datetime(config['date_split'].year, config['date_split'].month, config['date_split'].day)
    return config

def prepare_train_dataset_v0(config,
                             train_ds_name = 'train_dataset.pt',
                             test_ds_name = 'test_dataset.pt'):
    
    data, _ = preparing_dataframe(config['path_to_data'],
                                  config['to_float_point_feat'],
                                  config['nan_features'],
                                  config['numeric_features'],
                                  config['cat_features'],
                                  config['date_feature'],
                                  config['ts_by_features'])
    
    train_dataset, test_dataset = train_test_split(data,
                                                   config['date_feature'],
                                                   config['date_split'],
                                                   config['n_prev_days'],
                                                   config['numeric_features'],
                                                   config['cat_features'])
    
    data.to_csv(config['path_to_save'] + 'milk_clean.csv')
    torch.save(train_dataset, config['train_process']['path_to_save'] + train_ds_name)
    torch.save(test_dataset, config['train_process']['path_to_save'] + test_ds_name)

def prepare_inference_datset_v0(config,
                                inference_ds_name = 'milk_inference.pt'):
    data, _ = preparing_dataframe(config['path_to_data'],
                                  config['to_float_point_feat'],
                                  config['nan_features'],
                                  config['numeric_features'],
                                  config['cat_features'],
                                  config['date_feature'],
                                  config['ts_by_features'])
    inference_dataset = create_dataset_by_concat(data,
                                                 config['date_feature'],
                                                 config['date_split'],
                                                 config['n_prev_days'],
                                                 config['numeric_features'],
                                                 config['cat_features'])
    data.to_csv(config['path_to_save'] + 'milk_clean_inference.csv')
    torch.save(inference_dataset, config['inference_process']['path_to_save'] + inference_ds_name)

def prepare_dataset_v0(config):
    if config['is_train_process']:
        prepare_train_dataset_v0(config)
    else:
        prepare_inference_datset_v0(config)


if __name__ == '__main__':
    config = get_args_from_config()
    prepare_dataset_v0(config)

