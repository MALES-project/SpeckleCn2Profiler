import torch
import torchvision
import os
from typing import Tuple
from torch import nn


def setup_model(config: dict) -> Tuple[nn.Module, int]:
    """Returns the model specified in the configuration file, with the last
    layer corresponding to the number of screens.

    Parameters
    ----------
    config : dict
        Dictionary containing the configuration

    Returns
    -------
    model : torch.nn.Module
        The model with the loaded state
    last_model_state : int
        The number of the last model state
    """

    model_name = config['model']['name']
    model_type = config['model']['type']
    pretrained = config['model']['pretrained']
    nscreens = config['speckle']['nscreens']
    data_directory = config['speckle']['datadirectory']

    print(f'^^^ Loading model {model_name} of type {model_type}')

    if model_type in ['resnet18', 'resnet50', 'resnet152']:
        return get_a_resnet(nscreens, data_directory, model_name, model_type,
                            pretrained)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_a_resnet(nscreens: int, datadirectory: str, model_name: str,
                 model_type: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """Returns a pretrained ResNet model, with the last layer corresponding to
    the number of screens.

    Parameters
    ----------
    nscreens : int
        Number of screens
    datadirectory : str
        Path to the directory containing the data
    model_name : str
        The name of the model
    model_type : str
        The type of the ResNet
    pretrained : bool
        Whether to use a pretrained model or not

    Returns
    -------
    model : torch.nn.Module
        The model with the loaded state
    last_model_state : int
        The number of the last model state
    """

    if model_type == 'resnet18':
        model = torchvision.models.resnet18(
            weights='IMAGENET1K_V1' if pretrained else None)
        finaloutsize = 512
    elif model_type == 'resnet50':
        model = torchvision.models.resnet50(
            weights='IMAGENET1K_V2' if pretrained else None)
        finaloutsize = 2048
    elif model_type == 'resnet152':
        model = torchvision.models.resnet152(
            weights='IMAGENET1K_V2' if pretrained else None)
        finaloutsize = 2048
    else:
        raise ValueError(f'Unknown model {model_type}')

    # Give it its name
    model.name = model_name
    # put it in evaluation mode
    model.eval()

    # Change the model to process black and white input
    model.conv1 = torch.nn.Conv2d(1,
                                  64,
                                  kernel_size=(7, 7),
                                  stride=(2, 2),
                                  padding=(3, 3),
                                  bias=False)
    # Add a final layer to predict the output
    model.fc = torch.nn.Sequential(torch.nn.Linear(finaloutsize, nscreens),
                                   torch.nn.Sigmoid())

    return load_model_state(model, datadirectory)


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
        model.load_state_dict(
            torch.load(
                f'{datadirectory}/{model.name}_states/{model.name}_{last_model_state}.pth'
            ))
        return model, last_model_state
    else:
        print('No pretrained model to load')
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
