import torch
import pickle
import random
import os
import torch.nn as nn
import matplotlib.pyplot as plt


def plot_preprocessed_image(image_orig: torch.tensor, image: torch.tensor,
                            tags: torch.tensor, counter: int,
                            datadirectory: str, mname: str,
                            file_name: str, polar: bool=False) -> None:
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


def setup_loss(config: dict) -> nn.Module:
    """Returns the criterion specified in the configuration file.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration

    Returns
    -------
    criterion : torch.nn.Module
        The criterion with the loaded state
    """

    criterion_name = config['hyppar']['loss']
    if criterion_name == 'BCELoss':
        return torch.nn.BCELoss()
    elif criterion_name == 'MSELoss':
        return torch.nn.MSELoss()
    elif criterion_name == 'MAELoss':
        return torch.nn.L1Loss()
    elif criterion_name == 'Pearson':
        return PearsonCorrelationLoss()
    else:
        raise ValueError(f'Unknown criterion {criterion_name}')


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


class PearsonCorrelationLoss(nn.Module):

    def __init__(self):
        super(PearsonCorrelationLoss, self).__init__()

    def forward(self, x, y):
        # Calculate mean of x and y
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)

        # Calculate covariance
        cov_xy = torch.mean((x - mean_x) * (y - mean_y))

        # Calculate standard deviations
        std_x = torch.std(x)
        std_y = torch.std(y)

        # Calculate Pearson correlation coefficient
        corr = cov_xy / (std_x * std_y + 1e-8
                         )  # Add a small epsilon to avoid division by zero

        # The loss is 1 - correlation to be minimized
        loss = 1.0 - corr

        return loss
