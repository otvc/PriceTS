"Script for sample ts by particular item"

import pandas as pd

import click

@click.command()
@click.option("--load",
              default="data/input/pd_milk_data.csv",
              help="Path to dataset with ts")
@click.option("--save",
              default="data/output/pd_milk_data_item.csv",
              help="Path to save new dataset")
@click.option("--item",
              default=38,
              help="Id of particular item.")
def select_ts_by_item(load:str,
                      save:str,
                      item:int) -> pd.DataFrame:
    """Select timeseries by one item.

    Args:
        load (str): path to dataset with ts.
        save (str): path to save new dataset
        item (int): id of particular item.

    Returns:
        pd.DataFrame: new subseries.
    """
    data = pd.read_csv(load)
    target_column = 'item'
    subseria = data[data[target_column] == item]
    subseria.to_csv(save)
    return subseria

if __name__ == '__main__':
    select_ts_by_item()
