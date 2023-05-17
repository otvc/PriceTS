from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

def plot_predict_process(ground_truth:np.ndarray,
                       predicted:np.ndarray,
                       title_suffix:str='',
                       path:str = 'artefacts/predicts',
                       name:str = 'CatEmbLoss') -> None:
    """

    Args:
        ground_truth (np.ndarray): _description_
        predicted (np.ndarray): _description_
        title_suffix (str, optional): _description_. Defaults to ''.
        path (str, optional): _description_. Defaults to 'artefacts/predicts'.
        name (str, optional): _description_. Defaults to 'CatEmbLoss'.
    """
    fig, axes = plt.subplots(1, 1, figsize=(15, 5))

    axes.set_title(' '.join(['Loss', title_suffix]))
    axes.plot(ground_truth, label='ground truth')
    axes.plot(predicted, label='predicted')
    axes.legend()

    plt.savefig(path + name + '.png')
