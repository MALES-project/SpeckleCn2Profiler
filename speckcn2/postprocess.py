import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Optional
from torch import Tensor
from torch.utils.data import Dataset
from torch import device as Device


def tags_distribution(
        conf: dict,
        dataset: Dataset,
        test_tags: Tensor,
        device: Device,
        rescale: bool = False,
        recover_tag: Optional[Callable[[Tensor], Tensor]] = None) -> None:
    """Plots the distribution of the tags.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    dataset : torch.utils.data.Dataset
        The dataset used for training
    test_tags : torch.Tensor
        The predicted tags for the test dataset
    device : torch.device
        The device to use
    data_directory : str
        The directory where the data is stored
    rescale : bool, optional
        Whether to rescale the tags using recover_tag() or leave them between 0 and 1
    recover_tag : callable, optional
        Function to recover a tag
    """

    data_directory = conf['speckle']['datadirectory']
    model_name = conf['model']['name']

    # create a folder to store the plots
    if not os.path.isdir(f'{data_directory}/{model_name}_plots'):
        os.mkdir(f'{data_directory}/{model_name}_plots')

    train_tags = np.array([tag for _, tag in dataset])
    predic_tags = np.array([n.cpu().numpy() for n in test_tags])
    print(f'Data shape: {train_tags.shape}')
    print(f'Prediction shape: {predic_tags.shape}')
    print(f'Train mean: {train_tags.mean()}')
    print(f'Train std: {train_tags.std()}')
    print(f'Prediction mean: {predic_tags.mean()}')
    print(f'Prediction std: {predic_tags.std()}')
    # plot the distribution of each tag element
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(8):
        if rescale and recover_tag is not None:
            axs[i // 4, i % 4].hist(
                recover_tag(predic_tags[:, i]),
                bins=20,
                color='tab:red',
                density=True,
                #histtype='step',
                alpha=0.5,
                label='Model precitions')
            axs[i // 4, i % 4].hist(
                recover_tag(train_tags[:, i]),
                bins=20,
                color='tab:blue',
                density=True,
                #histtype='step',
                alpha=0.5,
                label='Training data')
        else:
            axs[i // 4, i % 4].hist(
                predic_tags[:, i],
                bins=20,
                color='tab:red',
                density=True,
                #histtype='step',
                alpha=0.5,
                label='Model precitions')
            axs[i // 4, i % 4].hist(
                train_tags[:, i],
                bins=20,
                color='tab:blue',
                density=True,
                #histtype='step',
                alpha=0.5,
                label='Training data')
        axs[i // 4, i % 4].set_title(f'Tag {i}')
    axs[0, 1].legend()
    plt.savefig(f'{data_directory}/{model_name}_plots/tags_distribution.png')
    plt.close()


def plot_loss(conf: dict, model, data_dir: str) -> None:
    """Plots the loss of the model.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    model : torch.nn.Module
        The model to plot the loss of
    data_dir : str
        The directory where the data is stored
    """

    model_name = conf['model']['name']

    # create a folder to store the plots
    if not os.path.isdir(f'{data_dir}/{model_name}_plots'):
        os.mkdir(f'{data_dir}/{model_name}_plots')

    # plot the loss
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(model.epoch, model.loss, label='Training loss')
    axs[0].plot(model.epoch, model.val_loss, label='Validation loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[1].plot(model.epoch, model.time, label='Time per epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Time [s]')
    axs[1].legend()
    plt.savefig(f'{data_dir}/{model_name}_plots/loss.png')
    plt.close()
