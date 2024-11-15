"""This module contains the definition of the EnsembleModel class and a
setup_model function.

The EnsembleModel class is a wrapper that allows any model to be used
for ensembled data. The setup_model function initializes and returns a
model based on the provided configuration.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import torch
import torchvision
from torch import nn

from speckcn2.io import load_model_state
from speckcn2.scnn import SteerableCNN, create_final_block


class EnsembleModel(nn.Module):
    """Wrapper that allows any model to be used for ensembled data."""

    def __init__(self, conf: dict, device: torch.device):
        """Initializes the EnsembleModel.

        Parameters
        ----------
        conf: dict
            The global configuration containing the model parameters.
        device : torch.device
            The device to use
        """
        super(EnsembleModel, self).__init__()

        self.ensemble_size = conf['preproc'].get('ensemble', 1)
        self.device = device
        self.uniform_ensemble = conf['preproc'].get('ensemble_unif', False)
        resolution = conf['preproc']['resize']
        self.D = conf['noise']['D']
        self.t = conf['noise']['t']
        self.snr = conf['noise']['snr']
        self.dT = conf['noise']['dT']
        self.dO = conf['noise']['dO']
        self.rn = conf['noise']['rn']
        self.fw = conf['noise']['fw']
        self.bit = conf['noise']['bit']
        self.discretize = conf['noise']['discretize']
        self.rot_sym = conf['noise'].get('rotation_sym', 0)
        if self.rot_sym > 0:
            self.rot_fold = 360 // self.rot_sym
        self.apply_masks = conf['noise'].get('apply_masks', False)
        if self.apply_masks:
            self.mask_D, self.mask_d, self.mask_X, self.mask_Y = self.create_masks(
                resolution)

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
            images = self.apply_noise(images)
            tags = torch.tensor(np.stack(tags)).to(self.device)

            return model(images), tags, images
        else:
            batch = list(itertools.chain(*batch_ensemble))
            # Like the ensemble=1 case, I can process independently each element of the batch
            images, tags, ensembles = zip(*batch)
            images = torch.stack(images).to(self.device)
            images = self.apply_noise(images)
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

    def apply_noise(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Processes a tensor of 2D images.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Tensor of 2D images with shape (batch, channels, width, height).

        Returns
        -------
        processed_tensor : torch.Tensor
            Tensor of processed 2D images.
        """
        batch, channels, height, width = image_tensor.shape
        processed_tensor = torch.zeros_like(image_tensor)

        # Apply rotation symmetry
        if self.rot_sym > 0:
            angle = random.randint(0, self.rot_fold)
            image_tensor = torch.rot90(image_tensor, angle, (2, 3))

        # Normalize wrt optical power
        image_tensor = image_tensor / torch.mean(
            image_tensor, dim=(2, 3), keepdim=True)

        amp = self.rn * 10**(self.snr / 20)

        for i in range(batch):
            for j in range(channels):
                B = image_tensor[i, j]

                ## Apply masks
                if self.apply_masks:
                    B[self.mask_D] = 0
                    B[self.mask_d] = 0
                    B[self.mask_X] = 0
                    B[self.mask_Y] = 0

                # Add noise sources
                A = self.rn + self.rn * torch.randn(
                    height, width, device=self.device) + amp * B + torch.sqrt(
                        amp * B) * torch.randn(
                            height, width, device=self.device)

                # Make a discretized version
                if self.discretize == 'on':
                    C = torch.round(A / self.fw * 2**self.bit)
                    C[A > self.fw] = self.fw
                    C[A < 0] = 0
                else:
                    C = A

                processed_tensor[i, j] = C

        return processed_tensor

    def create_masks(
        self, resolution: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Creates the masks for the circular aperture and the spider.

        Parameters
        ----------
        resolution : int
            Resolution of the images.

        Returns
        -------
        mask_D : torch.Tensor
            Mask for the circular aperture.
        mask_d : torch.Tensor
            Mask for the central obscuration.
        mask_X : torch.Tensor
            Mask for the horizontal spider.
        mask_Y : torch.Tensor
            Mask for the vertical spider.
        """
        # Coordinates
        x = torch.linspace(-1, 1, resolution, device=self.device)
        X, Y = torch.meshgrid(x, x, indexing='ij')
        d = self.dO * self.D  # Diameter obscuration

        R = torch.sqrt(X**2 + Y**2)

        # Masking image
        mask_D = R > self.D
        mask_d = R < d
        mask_X = torch.abs(X) < self.t / 2
        mask_Y = torch.abs(Y) < self.t / 2

        return mask_D, mask_d, mask_X, mask_Y


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


class EarlyStopper:
    """ Early stopping to stop the training when the validation loss does not decrease anymore.
    """

    def __init__(self, patience: int = 1, min_delta: float = 0):
        """Initializes the EarlyStopper.

        Parameters
        ----------
        patience: int
            Number of epochs of tolerance before stopping.
        min_delta: float
            Percentage of tolerance in considering the loss acceptable.
        """

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        """ Computes if the early stop condition is met at the current step.

        Parameters
        ----------
        validation_loss: float
            Current value of the validation loss

        Returns
        -------
        bool
            It returns True if the training has met the stop condition.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss * (1 + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
