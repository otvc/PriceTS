import argparse

import pandas as pd
import numpy as np


def get_unique_timeserieses(path_to_load = 'data/input/milk_clean.csv',
                            path_to_save = 'data/input/unique_timeserieses.csv',
                            subset_uniq_feat = ['price_zone_&_class_name', 'item', 'store']):
    test_df = pd.read_csv(path_to_load)
    np_val = test_df[subset_uniq_feat].values
    unique_values = np.unique(np_val, axis = 0)
    output_df = pd.DataFrame({feat: unique_values[:, i] 
                              for i, feat in enumerate(subset_uniq_feat)})
    output_df.to_csv(path_to_save)

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
                        default = 'data/input/unique_timeserieses.csv')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_params()
    get_unique_timeserieses(args.load, args.save)
