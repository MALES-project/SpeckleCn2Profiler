"""This module provides utility functions for loading and saving model
configurations and states.

It includes functions to load configuration files, save model states,
load model states, and load the latest model state from a directory.
"""

from __future__ import annotations

import os

import torch
import yaml

from speckcn2.utils import ensure_directory


def load_config(config_file_path: str) -> dict:
    """Load the configuration file from a given path.

    This function reads a YAML configuration file and returns its contents as a dictionary.

    Parameters
    ----------
    config_file_path : str
        Path to the .yaml configuration file

    Returns
    -------
    config : dict
        Dictionary containing the configuration
    """
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save(model: torch.nn.Module, datadirectory: str) -> None:
    """Save the model state and the model itself to a specified directory.

    This function saves the model's state dictionary and other relevant information
    such as epoch, loss, validation loss, and time to a file in the specified directory.

    Parameters
    ----------
    model : torch.nn.Module
        The model to save
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
    """Load the model state and the model itself from a specified directory and
    epoch.

    This function loads the model's state dictionary and other relevant information
    such as epoch, loss, validation loss, and time from a file in the specified directory.

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
    model.load_state_dict(model_state['model_state_dict'], strict=False)

    assert model.epoch[
        -1] == epoch, 'The epoch of the model is not the same as the one loaded'


def load_model_state(model: torch.nn.Module,
                     datadirectory: str) -> tuple[torch.nn.Module, int]:
    """Loads the latest model state from the given directory.

    This function checks the specified directory for the latest model state file,
    loads it, and updates the model with the loaded state. If no state is found,
    it initializes the model state.

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
    # Print model information
    print(model)
    model.nparams = sum(p.numel() for p in model.parameters())
    print(f'\n--> Nparams = {model.nparams}')

    ensure_directory(f'{datadirectory}/{model.name}_states')

    # Check what is the last model state
    try:
        last_model_state = sorted([
            int(file_name.split('.pth')[0].split('_')[-1])
            for file_name in os.listdir(f'{datadirectory}/{model.name}_states')
        ])[-1]
    except Exception as e:
        print(f'Warning: {e}')
        last_model_state = 0

    if last_model_state > 0:
        print(
            f'Loading model at epoch {last_model_state}, from {datadirectory}')
        load(model, datadirectory, last_model_state)
        return model, last_model_state
    else:
        print('No pretrained model to load')

        # Initialize some model state measures
        model.loss = []
        model.val_loss = []
        model.time = []
        model.epoch = []

        return model, 0
