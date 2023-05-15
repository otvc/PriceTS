import argparse

import pandas as pd
import numpy as np


def select_subseries(select_values = {},
                     path_to_load = 'data/input/milk_clean.csv',
                     path_to_save = 'data/input/milk_clean_small.csv'):
    test_df = pd.read_csv(path_to_load)
    print(select_values)
    subseries = test_df
    for feat_name, value in select_values.items():
        subseries = subseries[subseries[feat_name] == value]

    subseries.to_csv(path_to_save)

def get_params():
    parser = argparse.ArgumentParser(prog = 'TSSelecter', 
                                     description = 'Script for selecting unique timeserieses')
    parser.add_argument('-l', 
                        '--load', 
                        help = 'path to csv with timeserieses',
                        default = 'data/input/milk_clean.csv')
    
    parser.add_argument('-s', 
                        '--save', 
                        help = 'path to save unique subserieses by feature values',
                        default = 'data/input/milk_clean_small.csv')
    
    parser.add_argument('-p',
                        '--pzcls', 
                        help = 'price zone and cls numeric feature value',
                        default = 2)
    
    parser.add_argument('-i',
                        '--item', 
                        help = 'item numeric features value',
                        default = 542339)
    
    parser.add_argument('-st',
                        '--store', 
                        help = 'store numeric feature value',
                        default = 43)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_params()
    subset_uniq_feat = ['price_zone_&_class_name', 'item', 'store']
    select_values = {subset_uniq_feat[0]: args.pzcls, 
                     subset_uniq_feat[1]: args.item,
                     subset_uniq_feat[2]: args.store}
    select_subseries(select_values, args.load, args.save)
