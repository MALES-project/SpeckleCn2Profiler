"""This module defines a Steerable Convolutional Neural Network (SteerableCNN)
using the escnn library for equivariant neural networks. The primary class,
SteerableCNN, allows for the creation of a convolutional neural network that
handles various symmetries, making it useful for tasks requiring rotational
invariance.

The module imports necessary libraries, including torch and escnn, and defines utility
functions such as `create_block`, `compute_new_features`, `create_pool`, and `create_final_block`.
These functions help construct convolutional blocks, calculate feature map sizes,
create anti-aliased pooling layers, and build fully connected layers.

The SteerableCNN class initializes with a configuration dictionary and a symmetry parameter,
setting up parameters like kernel sizes, paddings, strides, and feature fields.
It determines the symmetry group and initializes the input type for the network.

The network is built by iterating through specified kernel sizes, creating convolutional
blocks and pooling layers, adding a group pooling layer for invariance, and creating final
fully connected layers using `create_final_block`.

The forward method processes the input tensor through the network, applying each
equivariant block, performing group pooling, and classifying the output using the fully connected layers.

Overall, this module provides a flexible framework for building steerable
convolutional neural networks with configurable symmetries and architectures.
"""
from __future__ import annotations

import torch
from escnn import gspaces, nn


def create_block(in_type, out_type, kernel_size, padding, stride):
    return nn.SequentialModule(
        nn.R2Conv(in_type,
                  out_type,
                  kernel_size=kernel_size,
                  padding=padding,
                  stride=stride,
                  bias=False), nn.InnerBatchNorm(out_type),
        nn.ReLU(out_type, inplace=True))


def compute_new_features(nfeatures, kernel_size, padding, stride):
    # **Out dim of a convolution is:
    # (in_image_res - kernel_size + 2*pad)/stride + 1
    return int((nfeatures - kernel_size + 2 * padding) / stride) + 1


def create_pool(out_type, sigma, padding, stride):
    return nn.SequentialModule(
        nn.PointwiseAvgPoolAntialiased(out_type,
                                       sigma=sigma,
                                       stride=stride,
                                       padding=padding))


def create_final_block(config: dict, n_initial: int,
                       nscreens: int) -> nn.Sequential:
    """Creates a fully connected neural network block based on a predefined
    configuration.

    This function dynamically creates a sequence of PyTorch layers for a fully connected
    neural network. The configuration for the layers is read from a global `config` dictionary
    which should contain a 'final_block' key with a list of layer configurations. Each layer
    configuration is a dictionary that must include a 'type' key with the name of the layer
    class (e.g., 'Linear', 'Dropout', etc.) and can include additional keys for the layer
    parameters.

    The first 'Linear' layer in the configuration has its number of input features set to `n_in`,
    and any 'Linear' layer with 'out_features' set to 'nscreens' has its number of output features
    set to `nscreens`.

    Args:
    Parameters
    ----------
    config : dict
        The global configuration dictionary containing the layer configurations.
    n_initial : int
        The number of input features for the first 'Linear' layer.
    nscreens : int
        The number of output features for any 'Linear' layer with 'out_features' set to 'nscreens'.

    Returns
    ----------
    torch.nn.Sequential: A sequential container of the configured PyTorch layers.
    """
    layers = []

    for layer_config in config['final_block']:
        layer_type = layer_config.pop('type')

        if layer_type == 'Linear':
            # The first linear layer has a constrained number of input features
            if n_initial != -1:
                layer_config['in_features'] = n_initial
                n_initial = -1
            if layer_config['out_features'] == 'nscreens':
                layer_config['out_features'] = nscreens
            # Pass the number of features as args
            n_in = layer_config.pop('in_features')
            n_out = layer_config.pop('out_features')

            layers.append(torch.nn.Linear(n_in, n_out, **layer_config))
        else:
            layer_class = getattr(torch.nn, layer_type)
            layers.append(layer_class(**layer_config))

    return torch.nn.Sequential(*layers)


class SteerableCNN(torch.nn.Module):

    def __init__(self, config, symmetry):

        super(SteerableCNN, self).__init__()

        self.nscreens = config['speckle']['nscreens']
        self.in_image_res = config['preproc']['resize']
        self.ensemble = config['preproc'].get('ensemble', 1)

        self.KERNEL_SIZES = config['scnn']['KERNEL_SIZES']
        self.PADDINGS = config['scnn']['PADDINGS']
        self.STRIDES = config['scnn']['STRIDES']
        self.FEATURE_FIELDS = config['scnn']['FEATURE_FIELDS']
        self.POOL_INDICES = config['scnn']['POOL_INDICES']
        self.SIGMA = config['scnn']['SIGMA']
        self.POOL_STRIDES = config['scnn']['POOL_STRIDES']
        self.POOL_PADDINGS = config['scnn']['POOL_PADDINGS']

        # Decide the symmetry group
        symmetry_map = {
            'C8': 8,
            'C16': 16,
            'C4': 4,
            'C6': 6,
            'C10': 10,
            'C12': 12,
        }
        try:
            self.r2_act = gspaces.rot2dOnR2(N=symmetry_map[symmetry])
        except KeyError:
            raise ValueError('The symmetry must be one of ' +
                             ', '.join(symmetry_map.keys()))

        # Start computing the size of the feature map
        self.nfeatures = self.in_image_res

        # The input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        # This mask is to remove the 'corner' pixels in the input image,
        # that would get outside of the square while rotating
        self.mask = nn.MaskModule(in_type, self.in_image_res, margin=0),

        self.blocks = torch.nn.ModuleList()
        for i in range(len(self.KERNEL_SIZES)):
            # Specify the output type of the convolutional layer
            # we choose feature fields, each transforming under the regular representation of the group
            out_type = nn.FieldType(
                self.r2_act,
                self.FEATURE_FIELDS[i] * [self.r2_act.regular_repr])
            block = create_block(in_type, out_type, self.KERNEL_SIZES[i],
                                 self.PADDINGS[i], self.STRIDES[i])
            self.blocks.append(block)
            in_type = block.out_type
            self.nfeatures = compute_new_features(self.nfeatures,
                                                  self.KERNEL_SIZES[i],
                                                  self.PADDINGS[i],
                                                  self.STRIDES[i])
            # Once every two layers, add a pooling layer
            if i in self.POOL_INDICES:
                j = self.POOL_INDICES.index(i)
                pool = create_pool(out_type, self.SIGMA[j],
                                   self.POOL_PADDINGS[j], self.POOL_STRIDES[j])
                self.blocks.append(pool)
                # For gaussian blur filters, the kernel size is 2 * int(round(3 * sigma)) + 1
                kernel_size = 2 * int(round(3 * self.SIGMA[j])) + 1
                self.nfeatures = compute_new_features(self.nfeatures,
                                                      kernel_size,
                                                      self.POOL_PADDINGS[j],
                                                      self.POOL_STRIDES[j])
        if self.nfeatures < 1:
            raise ValueError(
                'The number of features is too small. Please, check the kernel sizes, paddings and strides'
            )

        # Apply a group pooling layer to impose the invariance
        self.gpool = nn.GroupPooling(out_type)

        # number of output channels
        nc = self.gpool.out_type.size * self.nfeatures * self.nfeatures

        # If the model uses multiple images as input,
        # I add an extra channel as confidence weight
        # to average the final prediction
        if self.ensemble > 1:
            out_size = self.nscreens + 1
        else:
            out_size = self.nscreens

        # Create the final fully connected layers
        self.fully_net = create_final_block(config, nc, out_size)

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)

        # apply each equivariant block
        for block in self.blocks:
            x = block(x)

        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # classify with the final fully connected layers
        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x
