import pandas as pd

import numpy as np

import click

from utils import  plot_predict_process

@click.command()
@click.option("--read",
              default='data/output/inf_CatEmbLSTM.csv',
              help="Path to csv dataframe with\
              ground truth and predicted val.")
@click.option("--pzcn",
              default=8,
              help="Price zone and class_name,")
@click.option("--store",
              default=1,
              help="Id of store.")
@click.option("--item",
              default=70054,
              help="Id of item.")
def forecast_item_price_zone(read:str,
                             pzcn:int,
                             store:int,
                             item:int
                             ) -> None:
    """Plot a particular subseries. 

    Args:
        read (str): Path to csv dataframe with 
                    ground truth and predicted val;
        pzcn (int): Price zone and class name;
        store (int): id of store;
        item (int): id of item;
    """
    przc_column = 'price_zone_&_class_name'
    store_column = 'store'
    item_column = 'item'
    predict_column = 'predicted'
    target_column = 'sales_units'
    data = pd.read_csv(read)
    data = data[(data[przc_column] == pzcn) &
                (data[store_column] == store) &
                (data[item_column] == item)]
    title = f'Forecast [{pzcn},{store},{item}]'
    plot_predict_process(data[target_column], data[predict_column],
                         name=title)
    
if __name__ == '__main__':
    forecast_item_price_zone()
