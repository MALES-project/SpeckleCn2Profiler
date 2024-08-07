"""This module provides utility functions for image processing and model
optimization.

It includes functions to plot original and preprocessed images along
with their tags, ensure the existence of specified directories, set up
optimizers based on configuration files, and create circular masks with
an inner "spider" circle removed. These utilities facilitate various
tasks in image analysis and machine learning model training.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def plot_preprocessed_image(image_orig: torch.tensor,
                            image: torch.tensor,
                            tags: torch.tensor,
                            counter: int,
                            datadirectory: str,
                            mname: str,
                            file_name: str,
                            polar: bool = False) -> None:
    """Plots the original and preprocessed image, and the tags.

    Parameters
    ----------
    image_orig : torch.tensor
        The original image
    image : torch.tensor
        The preprocessed image
    tags : torch.tensor
        The screen tags
    counter : int
        The counter of the image
    datadirectory : str
        The directory containing the data
    mname : str
        The name of the model
    file_name : str
        The name of the original image
    polar : bool, optional
        If the image is in polar coordinates, by default False
    """

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Plot the original image
    axs[0].imshow(image_orig.squeeze(), cmap='bone')
    axs[0].set_title(f'Training Image {file_name}')
    # Plot the preprocessd image
    axs[1].imshow(image.squeeze(), cmap='bone')
    axs[1].set_title('Processed as')
    if polar:
        axs[1].set_xlabel(r'$r$')
        axs[1].set_ylabel(r'$\theta$')

    # Plot the tags
    axs[2].plot(tags, 'o')
    axs[2].set_yscale('log')
    axs[2].set_title('Screen Tags')
    axs[2].legend()

    fig.subplots_adjust(wspace=0.3)
    plt.savefig(f'{datadirectory}/imgs_to_{mname}/{counter}.png')
    plt.close()


def ensure_directory(data_directory: str) -> None:
    """Ensure that the directory exists.

    Parameters
    ----------
    data_directory : str
        The directory to ensure
    """

    if not os.path.isdir(data_directory):
        os.mkdir(data_directory)


def setup_optimizer(config: dict, model: nn.Module) -> nn.Module:
    """Returns the optimizer specified in the configuration file.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration
    model : torch.nn.Module
        The model to optimize

    Returns
    -------
    optimizer : torch.nn.Module
        The optimizer with the loaded state
    """

    optimizer_name = config['hyppar']['optimizer']
    if optimizer_name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=config['hyppar']['lr'])
    elif optimizer_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=config['hyppar']['lr'])
    else:
        raise ValueError(f'Unknown optimizer {optimizer_name}')


def create_circular_mask_with_spider(resolution: int,
                                     bkg_value: int = 0) -> torch.Tensor:
    """Creates a circular mask with an inner "spider" circle removed.

    Parameters
    ----------
    resolution : int
        The resolution of the square mask.
    bkg_value : int
        The background value to set for the masked areas. Defaults to 0.

    Returns
    -------
    torch.Tensor : np.ndarray
        A 2D tensor representing the mask.
    """
    # Create a circular mask
    center = (int(resolution / 2), int(resolution / 2))
    radius = min(center)
    Y, X = np.ogrid[:resolution, :resolution]
    mask = (X - center[0])**2 + (Y - center[1])**2 > radius**2

    # Remove the inner circle (spider)
    spider_radius = int(0.22 * resolution)
    spider_mask = (X - center[0])**2 + (Y - center[1])**2 < spider_radius**2

    # Apply background value to the mask and spider mask
    final_mask = np.ones((resolution, resolution), dtype=np.uint8)
    final_mask[mask] = bkg_value
    final_mask[spider_mask] = bkg_value

    return torch.Tensor(final_mask)
