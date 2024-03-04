import torch
import numpy as np
import itertools
import torchvision
from torch import nn
from speckcn2.io import load_model_state
from speckcn2.scnn import SteerableCNN

class EnsembleModel(nn.Module):
    """Wrapper that allows any model to be used for ensembled data."""

    def __init__(self,
                 ensemble_size: int,
                 device: torch.device,
                 uniform_ensemble: bool = False):
        """Initializes the EnsembleModel.

        Parameters
        ----------
        ensemble_size : int
            The number of images that will be processed together as an ensemble
        device : torch.device
            The device to use
        uniform_ensemble : bool
            If true, all the images in the ensemble will have the same weight
        """
        super(EnsembleModel, self).__init__()
        self.ensemble_size = ensemble_size
        self.device = device
        self.uniform_ensemble = uniform_ensemble

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
            tags = torch.tensor(np.stack(tags)).to(self.device)

            return model(images), tags, images
        else:
            batch = list(itertools.chain(*batch_ensemble))
            # Like the ensemble=1 case, I can process independently each element of the batch
            images, tags, ensembles = zip(*batch)
            images = torch.stack(images).to(self.device)
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
    pretrained = config['model']['pretrained']
    nscreens = config['speckle']['nscreens']
    data_directory = config['speckle']['datadirectory']
    img_res = config['preproc']['resize']
    ensemble = config['preproc']['ensemble']

    print(f'^^^ Initializing model {model_name} of type {model_type}')

    if model_type.startswith('resnet'):
        return get_a_resnet(nscreens, data_directory, model_name, model_type,
                            pretrained, ensemble)
    elif model_type.startswith('scnnC'):
        return get_scnn(config)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_a_resnet(nscreens: int,
                 datadirectory: str,
                 model_name: str,
                 model_type: str,
                 pretrained: bool,
                 ensemble: int = 1) -> tuple[nn.Module, int]:
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
    ensemble : int
        The number of input images that will be processd together as an ensemble corresponding to the same output

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

    # If the model uses multiple images as input, I add an extra channel as confidence weight to average the final prediction
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
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(finaloutsize, 64),
        torch.nn.BatchNorm1d(64),
        torch.nn.ELU(inplace=True),
        torch.nn.Linear(64, nscreens),
        torch.nn.Sigmoid(),
    )

    return load_model_state(model, datadirectory)



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
    }
    try:
        scnn_model = SteerableCNN(config, model_map[model_type])
    except KeyError:
        raise ValueError(f'Unknown model {model_type}')

    scnn_model.name = model_name

    return load_model_state(scnn_model, datadirectory)
