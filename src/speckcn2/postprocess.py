from __future__ import annotations

import itertools
from typing import Callable, Dict, Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch import device as Device

from speckcn2.loss import ComposableLoss
from speckcn2.mlmodels import EnsembleModel
from speckcn2.utils import ensure_directory


def tags_distribution(conf: dict,
                      train_set: list,
                      test_tags: Tensor,
                      device: Device,
                      rescale: bool = False,
                      recover_tag: Optional[list[Callable]] = None) -> None:
    """Plots the distribution of the tags.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    train_set : list
        The training set
    test_tags : torch.Tensor
        The predicted tags for the test dataset
    device : torch.device
        The device to use
    data_directory : str
        The directory where the data is stored
    rescale : bool, optional
        Whether to rescale the tags using recover_tag() or leave them between 0 and 1
    recover_tag : list, optional
        List of functions to recover each tag
    """

    data_directory = conf['speckle']['datadirectory']
    model_name = conf['model']['name']
    ensemble = conf['preproc'].get('ensemble', 1)

    ensure_directory(f'{data_directory}/result_plots')

    # Get the tags from the training set
    if ensemble > 1:
        train_set = list(itertools.chain(*train_set))
    _, tags, _ = zip(*train_set)
    tags = np.stack(tags)
    train_tags = np.array([n for n in tags])

    # Get the tags from the test set
    predic_tags = np.array([n.cpu().numpy() for n in test_tags])
    print(f'Data shape: {train_tags.shape}')
    print(f'Prediction shape: {predic_tags.shape}')
    print(f'Train mean: {train_tags.mean()}')
    print(f'Train std: {train_tags.std()}')
    print(f'Prediction mean: {predic_tags.mean()}')
    print(f'Prediction std: {predic_tags.std()}')

    # Keep track of J=sum(tags) for each sample
    J_pred = np.zeros(predic_tags.shape[0])
    J_true = np.zeros(train_tags.shape[0])

    # Plot the distribution of each tag element
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(train_tags.shape[1]):
        if rescale and recover_tag is not None:
            recovered_tag_model = np.asarray(
                [recover_tag[i](predic_tags[:, i], i)]).squeeze(0)
            recovered_tag_true = np.asarray(
                [recover_tag[i](train_tags[:, i], i)]).squeeze(0)
            axs[i // 4, i % 4].hist(recovered_tag_model,
                                    bins=20,
                                    color='tab:red',
                                    density=True,
                                    alpha=0.5,
                                    label='Model prediction')
            axs[i // 4, i % 4].hist(recovered_tag_true,
                                    bins=20,
                                    color='tab:blue',
                                    density=True,
                                    alpha=0.5,
                                    label='Training data')
            J_pred += 10**recovered_tag_model
            J_true += 10**recovered_tag_true
        else:
            axs[i // 4, i % 4].hist(predic_tags[:, i],
                                    bins=20,
                                    color='tab:red',
                                    density=True,
                                    alpha=0.5,
                                    label='Model prediction')
            axs[i // 4, i % 4].hist(train_tags[:, i],
                                    bins=20,
                                    color='tab:blue',
                                    density=True,
                                    alpha=0.5,
                                    label='Training data')
        axs[i // 4, i % 4].set_title(f'Tag {i}')
    axs[0, 1].legend()
    plt.tight_layout()
    if rescale:
        plt.savefig(f'{data_directory}/result_plots/{model_name}_tags.png')
    else:
        plt.savefig(
            f'{data_directory}/result_plots/{model_name}_tags_unscaled.png')
    plt.close()

    if rescale and recover_tag is not None:
        # Also plot the distribution of the sum of the tags
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs.hist(np.log10(J_pred),
                 bins=20,
                 color='tab:red',
                 density=True,
                 alpha=0.5,
                 label='Model prediction')
        axs.hist(np.log10(J_true),
                 bins=20,
                 color='tab:blue',
                 density=True,
                 alpha=0.5,
                 label='Training data')
        axs.set_title('Sum of J')
        axs.legend()
        plt.tight_layout()
        plt.savefig(f'{data_directory}/result_plots/{model_name}_sumJ.png')

        plt.close()


def average_speckle(conf: dict,
                    test_set: list,
                    device: Device,
                    model: nn.Torch,
                    criterion: ComposableLoss,
                    n_ensembles_to_plot: int = 100) -> None:
    """Test to see if averaging over speckle patterns improves the results.
    This function is then going to plot the relative error over the screen tags
    and the Fried parameter to make this evaluation.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    test_set : list
        The test set
    device : torch.device
        The device to use
    model : nn.Torch
        The trained model
    criterion : ComposableLoss
        The loss function
    n_ensembles_to_plot : int
        The number of ensembles to plot
    """

    data_directory = conf['speckle']['datadirectory']
    model_name = conf['model']['name']

    ensure_directory(f'{data_directory}/result_plots')

    # group the sets that have the same n[1]
    grouped_test_set: Dict = {}
    for n in test_set:
        key = tuple(n[1])
        if key not in grouped_test_set:
            grouped_test_set[key] = []
        grouped_test_set[key].append(n)
    print(
        '\nChecking if averaging speckle from the same turbulence improves results'
    )
    print(f'Number of samples: {len(test_set)}')
    print(f'Number of speckle groups: {len(grouped_test_set)}')

    # For each group compare the model prediction to the exact tag
    ensemble = EnsembleModel(conf, device)
    with torch.no_grad():
        model.eval()

        for ensemble_count, (key,
                             value) in enumerate(grouped_test_set.items()):
            avg_output = None
            cmap = cm.get_cmap('coolwarm')
            norm = plt.Normalize(1, len(value))

            if ensemble_count > n_ensembles_to_plot:
                continue

            for count, speckle in enumerate(value, 1):
                output, target, _ = ensemble(model, [speckle])

                if count == 1:
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

                if avg_output is None:
                    avg_output = output
                else:
                    avg_output += output

                loss, losses = criterion(avg_output / count, target)

                color = cmap(norm(count))
                ax[0].plot((torch.abs(avg_output / count - target) /
                            (target + 1e-7)).flatten(),
                           color=color)

                # Get the Cn2 profile and the recovered tags
                Cn2_pred = criterion.reconstruct_cn2(avg_output / count)
                Cn2_true = criterion.reconstruct_cn2(target)
                # and get all the measures
                all_measures = criterion._get_all_measures(
                    avg_output / count, target, Cn2_pred, Cn2_true)

                Fried_err = torch.abs(
                    all_measures['Fried_true'] -
                    all_measures['Fried_pred']) / all_measures['Fried_true']
                ax[1].scatter(count, Fried_err, color=color)

            ax[0].set_xlabel('# screen')
            ax[0].set_ylabel('J relative error ')
            ax[1].set_xlabel('# averaged speckles')
            ax[1].set_ylabel('Fried relative error')
            ax[0].set_yscale('log')
            ax[1].set_yscale('log')
            fig.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.suptitle('Effect of averaging speckles')
            plt.savefig(
                f'{data_directory}/{model_name}_score/average_ensemble{ensemble_count}.png'
            )
            plt.close()
