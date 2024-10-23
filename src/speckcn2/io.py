"""This module provides utility functions for loading and saving model
configurations and states.

It includes functions to load configuration files, save model states,
load model states, and load the latest model state from a directory.
"""

from __future__ import annotations

import os
import sys

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


def save(model: torch.nn.Module,
         datadirectory: str,
         early_stop: bool = False) -> None:
    """Save the model state and the model itself to a specified directory.

    This function saves the model's state dictionary and other relevant information
    such as epoch, loss, validation loss, and time to a file in the specified directory.

    Parameters
    ----------
    model : torch.nn.Module
        The model to save
    datadirectory : str
        The directory where the data is stored
    early_stop: bool
        If True, the model corresponds to the moment when early stop was triggered
    """
    model_state = {
        'epoch': model.epoch,
        'loss': model.loss,
        'val_loss': model.val_loss,
        'time': model.time,
        'model_state_dict': model.state_dict(),
    }

    if not early_stop:
        savename = f'{datadirectory}/{model.name}_states/{model.name}_{model.epoch[-1]}.pth'
    else:
        savename = f'{datadirectory}/{model.name}_states/{model.name}_{model.epoch[-1]}_earlystop.pth'

    torch.save(model_state, savename)


def load(model: torch.nn.Module,
         datadirectory: str,
         epoch: int,
         early_stop: bool = False) -> None:
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
    early_stop: bool
        If True, the last state reached the early stop condition
    """
    if early_stop:
        model_state = torch.load(
            f'{datadirectory}/{model.name}_states/{model.name}_{epoch}_earlystop.pth'
        )
        model.early_stop = True
    else:
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
    If the training was stopped after meeting an early stop condition, this function
    signals that the training should not be continued.

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

    fulldirname = f'{datadirectory}/{model.name}_states'
    ensure_directory(fulldirname)

    # First check if there was an early stop
    earlystop = [
        filename for filename in os.listdir(fulldirname)
        if 'earlystop' in filename
    ]
    if len(earlystop) == 0:
        # If there was no early stop, check what is the last model state
        try:
            last_model_state = sorted([
                int(file_name.split('.pth')[0].split('_')[-1])
                for file_name in os.listdir(fulldirname)
            ])[-1]
        except Exception as e:
            print(f'Warning: {e}')
            last_model_state = 0

        if last_model_state > 0:
            print(
                f'Loading model at epoch {last_model_state}, from {datadirectory}'
            )
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
    elif len(earlystop) == 1:
        filename = earlystop[0]
        print(f'Loading the early stop state {filename}')
        last_model_state = int(filename.split('_')[-2])
        load(model, datadirectory, last_model_state, early_stop=True)
        return model, last_model_state
    else:
        print(
            f'Error: more than one early stop state found. This is not correct. This is the list: {earlystop}'
        )
        sys.exit(0)
