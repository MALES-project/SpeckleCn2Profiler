# Import any modules or subpackages here
from __future__ import annotations

from .io import load, load_config, load_model_state, save
from .loss import ComposableLoss
from .mlmodels import EnsembleModel, get_a_resnet, setup_model
from .mlops import score, train
from .normalizer import Normalizer
from .plots import (
    plot_histo_losses,
    plot_J_error_details,
    plot_loss,
    plot_param_histo,
    plot_param_vs_loss,
    plot_samples_in_ensemble,
    plot_time,
)
from .postprocess import average_speckle_input, average_speckle_output, screen_errors, tags_distribution
from .preprocess import prepare_data, train_test_split
from .utils import setup_optimizer

__version__ = '1.0.0'
__all__ = [
    'load_config',
    'prepare_data',
    'Normalizer',
    'train_test_split',
    'setup_model',
    'get_a_resnet',
    'EnsembleModel',
    'ComposableLoss',
    'setup_optimizer',
    'train',
    'score',
    'tags_distribution',
    'screen_errors',
    'average_speckle_input',
    'average_speckle_output',
    'plot_J_error_details',
    'plot_loss',
    'plot_time',
    'save',
    'load',
    'load_model_state',
    'plot_histo_losses',
    'plot_param_vs_loss',
    'plot_param_histo',
    'plot_samples_in_ensemble',
]
