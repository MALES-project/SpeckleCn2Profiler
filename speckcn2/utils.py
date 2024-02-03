import torch
import os
import torch.nn as nn
from typing import Tuple


def save(model: torch.nn.Module, datadirectory: str) -> None:
    """Save the model state and the model itself.

    Parameters
    ----------
    model : torch.nn.Module
        The model to save
    epoch : int
        The epoch of the model
    datadirectory : str
        The directory where the data is stored
    """

    model_state = {
        'epoch': model.epoch,
        'loss': model.loss,
        'val_loss': model.val_loss,
        'time': model.time,
        'model_state_dict': model.state_dict(),
    }

    torch.save(
        model_state,
        f'{datadirectory}/{model.name}_states/{model.name}_{model.epoch[-1]}.pth'
    )


def load(model: torch.nn.Module, datadirectory: str, epoch: int) -> None:
    """Load the model state and the model itself.

    Parameters
    ----------
    model : torch.nn.Module
        The model to load
    datadirectory : str
        The directory where the data is stored
    epoch : int
        The epoch of the model
    """

    model_state = torch.load(
        f'{datadirectory}/{model.name}_states/{model.name}_{epoch}.pth')

    model.epoch = model_state['epoch']
    model.loss = model_state['loss']
    model.val_loss = model_state['val_loss']
    model.time = model_state['time']
    model.load_state_dict(model_state['model_state_dict'])

    assert model.epoch[
        -1] == epoch, 'The epoch of the model is not the same as the one loaded'


def load_model_state(model: nn.Module,
                     datadirectory: str) -> Tuple[nn.Module, int]:
    """Loads the model state from the given directory.

    Parameters
    ----------
    model : torch.nn.Module
        The model to load the state into
    datadirectory : str
        The directory where the model states are stored

    Returns
    -------
    model : torch.nn.Module
        The model with the loaded state
    last_model_state : int
        The number of the last model state
    """

    # Print model informations
    print(model)
    model.nparams = sum(p.numel() for p in model.parameters())
    print(f'\n--> Nparams = {model.nparams}')

    # If no model is stored, create the folder
    if not os.path.isdir(f'{datadirectory}/{model.name}_states'):
        os.mkdir(f'{datadirectory}/{model.name}_states')

    # check what is the last model state
    try:
        last_model_state = sorted([
            int(file_name.split('.pth')[0].split('_')[-1])
            for file_name in os.listdir(f'{datadirectory}/{model.name}_states')
        ])[-1]
    except Exception as e:
        print(f'Warning: {e}')
        last_model_state = 0

    if last_model_state > 0:
        print(f'Loading model at epoch {last_model_state}')
        load(model, datadirectory, last_model_state)
        return model, last_model_state
    else:
        print('No pretrained model to load')

        # Initialize some model state measure
        model.loss = []
        model.val_loss = []
        model.time = []
        model.epoch = []

        return model, 0


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
