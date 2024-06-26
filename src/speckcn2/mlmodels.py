from __future__ import annotations

import itertools

import numpy as np
import torch
import torchvision
from torch import nn

from speckcn2.io import load_model_state
from speckcn2.scnn import SteerableCNN, create_final_block


class EnsembleModel(nn.Module):
    """Wrapper that allows any model to be used for ensembled data."""

    def __init__(self,
                 ensemble_size: int,
                 device: torch.device,
                 uniform_ensemble: bool = False,
                 snr: str | None = None,
                 pixel_average: float = 1.,
                 resolution: int = 128):
        """Initializes the EnsembleModel.

        Parameters
        ----------
        ensemble_size : int
            The number of images that will be processed together as an ensemble
        device : torch.device
            The device to use
        uniform_ensemble : bool
            If true, all the images in the ensemble will have the same weight
        snr : str
            Signal to noise ratio. If >0 add noise to the input images.
            The noise is a Gaussian with mean=0 and std=1/snr
        pixel_average : float
            The average pixel value of the input images. It is used to calculate the noise
        resolution : int
            The resolution of the input images (assumed to be square)
        """
        super(EnsembleModel, self).__init__()
        self.ensemble_size = ensemble_size
        self.device = device
        self.uniform_ensemble = uniform_ensemble
        if snr:
            # For the noise, you have to get the average pixel value
            self.noise = (1. / eval(snr) * pixel_average).to(self.device)
            print(f'Adding noise with std={self.noise}')

        # Create a mask to ignore the spider
        self.mask = torch.ones(resolution, resolution).to(self.device)
        # Create a circular mask
        center = (int(resolution / 2), int(resolution / 2))
        radius = min(center)
        Y, X = np.ogrid[:resolution, :resolution]
        mask = (X - center[0])**2 + (Y - center[1])**2 > radius**2
        # Then the inner circle (called spider) is also removed
        # its default diameter, defined by the experimental setup, is 44% of the image width
        spider_radius = int(0.22 * resolution)
        spider_mask = (X - center[0])**2 + (Y -
                                            center[1])**2 < spider_radius**2
        bkg_value = 0
        self.mask[mask] = bkg_value
        self.mask[spider_mask] = bkg_value

    def forward(self, model, batch_ensemble):
        """Forward pass through the model.

        Parameters
        ----------
        model : torch.nn.Module
            The model to use
        batch_ensemble : list
            Each element is a batch of an ensemble of samples.
        """

        if self.ensemble_size == 1:
            batch = batch_ensemble
            # If no ensembling, each element of the batch is a tuple (image, tag, ensemble_id)
            images, tags, ensembles = zip(*batch)
            images = torch.stack(images).to(self.device)
            if self.noise:
                images += torch.randn_like(images) * self.noise * self.mask
            tags = torch.tensor(np.stack(tags)).to(self.device)

            return model(images), tags, images
        else:
            batch = list(itertools.chain(*batch_ensemble))
            # Like the ensemble=1 case, I can process independently each element of the batch
            images, tags, ensembles = zip(*batch)
            images = torch.stack(images).to(self.device)
            if self.noise:
                images += torch.randn_like(images) * self.noise * self.mask
            tags = torch.tensor(np.stack(tags)).to(self.device)

            model_output = model(images)

            # To average the self.ensemble_size outputs of the model I extract the confidence weights
            predictions = model_output[:, :-1]
            weights = model_output[:, -1]
            if self.uniform_ensemble:
                weights = torch.ones_like(weights)
            # multiply the prediction by the weights
            weighted_predictions = predictions * weights.unsqueeze(-1)
            # and sum over the ensembles
            weighted_predictions = weighted_predictions.view(
                model_output.size(0) // self.ensemble_size, self.ensemble_size,
                -1).sum(dim=1)
            # then normalize by the sum of the weights
            sum_weights = weights.view(
                weights.size(0) // self.ensemble_size,
                self.ensemble_size).sum(dim=1)
            ensemble_output = weighted_predictions / sum_weights.unsqueeze(-1)

            # and get the tags and ensemble_id of the first element of the ensemble
            tags = tags[::self.ensemble_size]
            ensembles = ensembles[::self.ensemble_size]

            return ensemble_output, tags, images


def setup_model(config: dict) -> tuple[nn.Module, int]:
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

    print(f'^^^ Initializing model {model_name} of type {model_type}')

    if model_type.startswith('resnet'):
        return get_a_resnet(config)
    elif model_type.startswith('scnnC'):
        return get_scnn(config)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_a_resnet(config: dict) -> tuple[nn.Module, int]:
    """Returns a pretrained ResNet model, with the last layer corresponding to
    the number of screens.

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
    ensemble = config['preproc'].get('ensemble', 1)

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

    # If the model uses multiple images as input,
    # add an extra channel as confidence weight
    # to average the final prediction
    if ensemble > 1:
        nscreens = nscreens + 1

    # Give it its name
    model.name = model_name

    # Change the model to process black and white input
    model.conv1 = torch.nn.Conv2d(1,
                                  64,
                                  kernel_size=(7, 7),
                                  stride=(2, 2),
                                  padding=(3, 3),
                                  bias=False)
    # Add a final fully connected piece to predict the output
    model.fc = create_final_block(config, finaloutsize, nscreens)

    return load_model_state(model, data_directory)


def get_scnn(config: dict) -> tuple[nn.Module, int]:
    """Returns a pretrained Spherical-CNN model, with the last layer
    corresponding to the number of screens."""

    model_name = config['model']['name']
    model_type = config['model']['type']
    datadirectory = config['speckle']['datadirectory']

    model_map = {
        'scnnC8': 'C8',
        'scnnC16': 'C16',
        'scnnC4': 'C4',
        'scnnC6': 'C6',
        'scnnC10': 'C10',
        'scnnC12': 'C12',
    }
    try:
        scnn_model = SteerableCNN(config, model_map[model_type])
    except KeyError:
        raise ValueError(f'Unknown model {model_type}')

    scnn_model.name = model_name

    return load_model_state(scnn_model, datadirectory)
