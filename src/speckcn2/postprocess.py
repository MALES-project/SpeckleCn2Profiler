from __future__ import annotations

import itertools
from typing import Callable, Dict, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from torch import Tensor, nn
from torch import device as Device

from speckcn2.loss import ComposableLoss
from speckcn2.mlmodels import EnsembleModel
from speckcn2.utils import ensure_directory


def tags_distribution(conf: dict,
                      train_set: list,
                      test_tags: Tensor,
                      device: Device,
                      nbins: int = 20,
                      rescale: bool = False,
                      recover_tag: Optional[list[Callable]] = None) -> None:
    """Function to plot the following:
    - distribution of the tags for unscaled results
    - distribution of the tags for rescaled results
    - distribution of the sum of the tags.

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
    nbins : int, optional
        Number of bins to use for the histograms
    rescale : bool, optional
        Whether to rescale the tags using recover_tag() or leave them between 0 and 1
    recover_tag : list, optional
        List of functions to recover each tag
    """

    data_directory = conf['speckle']['datadirectory']
    model_name = conf['model']['name']
    ensemble = conf['preproc'].get('ensemble', 1)

    dirname = f'{data_directory}/{model_name}_score'
    ensure_directory(dirname)

    # Get the tags from the training set
    if ensemble > 1:
        train_set = list(itertools.chain(*train_set))
    _, tags, _ = zip(*train_set)
    tags = np.stack(tags)
    train_tags = np.array([n for n in tags])

    # Get the tags from the test set
    predic_tags = np.array([n.cpu().numpy() for n in test_tags])

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
            J_pred += 10**recovered_tag_model
            J_true += 10**recovered_tag_true
            axs[i // 4, i % 4].hist(recovered_tag_model,
                                    bins=nbins,
                                    color='tab:red',
                                    density=True,
                                    alpha=0.5,
                                    label='Model prediction')
            axs[i // 4, i % 4].hist(recovered_tag_true,
                                    bins=nbins,
                                    color='tab:blue',
                                    density=True,
                                    alpha=0.5,
                                    label='Training data')
        else:
            axs[i // 4, i % 4].hist(predic_tags[:, i],
                                    bins=nbins,
                                    color='tab:red',
                                    density=True,
                                    alpha=0.5,
                                    label='Model prediction')
            axs[i // 4, i % 4].hist(train_tags[:, i],
                                    bins=nbins,
                                    color='tab:blue',
                                    density=True,
                                    alpha=0.5,
                                    label='Training data')
        axs[i // 4, i % 4].set_title(f'Screen {i}')
    axs[0, 1].legend()
    plt.tight_layout()
    figname = f'{dirname}/{model_name}_tags'
    if not rescale:
        figname += '_unscaled'
    plt.savefig(f'{figname}.png')
    plt.close()

    if rescale and recover_tag is not None:
        # Also plot the distribution of the sum of the tags
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs.hist(np.log10(J_pred),
                 bins=nbins,
                 color='tab:red',
                 density=True,
                 alpha=0.5,
                 label='Model prediction')
        axs.hist(np.log10(J_true),
                 bins=nbins,
                 color='tab:blue',
                 density=True,
                 alpha=0.5,
                 label='Training data')
        axs.set_title('Sum of J')
        axs.legend()
        plt.tight_layout()
        plt.savefig(f'{dirname}/{model_name}_sumJ.png')

        plt.close()


def average_speckle_output(conf: dict,
                           test_set: list,
                           device: Device,
                           model: nn.Torch,
                           criterion: ComposableLoss,
                           trimming: float = 0.1,
                           n_ensembles_to_plot: int = 100) -> None:
    """Test to see if averaging the prediction of multiple speckle patterns
    improves the results. This function is then going to plot the relative
    error over the screen tags and the Fried parameter to make this evaluation.

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
    n_screens = conf['speckle']['nscreens']

    dirname = f'{data_directory}/{model_name}_score/effect_averaging'
    ensure_directory(dirname)

    # group the sets that have the same n[1]
    grouped_test_set: Dict = {}
    for n in test_set:
        key = tuple(n[1])
        if key not in grouped_test_set:
            grouped_test_set[key] = []
        grouped_test_set[key].append(n)
    print('\nChecking if averaging speckle predictions improves results')
    print(f'Number of samples: {len(test_set)}')
    print(f'Number of speckle groups: {len(grouped_test_set)}')

    # In the end, we will plot groups that have uncommon values of loss
    loss_min = 1e10
    loss_max = 0
    ensemble_count = 0

    ensemble = EnsembleModel(conf, device)
    with torch.no_grad():
        model.eval()

        for key, value in grouped_test_set.items():
            _outputs = []
            _all_tags_pred = []
            cmap = plt.get_cmap('coolwarm')
            norm = plt.Normalize(1, len(value))

            for count, speckle in enumerate(value, 1):
                color = cmap(norm(count))
                output, target, _ = ensemble(model, [speckle])

                if count == 1:
                    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

                    # (1) Plot J vs nscreens
                    recovered_tag_true = criterion.get_J(target)
                    ax[0].plot(recovered_tag_true.squeeze(0).detach().cpu(),
                               '*',
                               label='True',
                               color='tab:green',
                               markersize=10,
                               markeredgecolor='black',
                               zorder=100)
                    ax[1].plot(recovered_tag_true.squeeze(0).detach().cpu(),
                               '*',
                               label='True',
                               color='tab:green',
                               markersize=10,
                               markeredgecolor='black',
                               zorder=100)
                    blue_patch = mpatches.Patch(color=color,
                                                label='One speckle')

                # Use the trimmed mean to get the average output
                # (trim_mean works only on cpu so you have to move back and forth)
                _outputs.append(output.detach().cpu().numpy())
                avg_output = torch.tensor(stats.trim_mean(_outputs,
                                                          trimming)).to(device)

                loss, losses = criterion(avg_output, target)

                # Get the Cn2 profile and the recovered tags
                Cn2_pred = criterion.reconstruct_cn2(avg_output)
                Cn2_true = criterion.reconstruct_cn2(target)
                recovered_tag_pred = criterion.get_J(avg_output)
                _all_tags_pred.append(recovered_tag_pred.detach().cpu())
                ax[1].plot(recovered_tag_pred.squeeze(0).detach().cpu(),
                           'o',
                           color=color)
                # and get all the measures
                all_measures = criterion._get_all_measures(
                    avg_output, target, Cn2_pred, Cn2_true)

                Fried_err = torch.abs(
                    all_measures['Fried_true'] -
                    all_measures['Fried_pred']) / all_measures['Fried_true']
                ax[2].scatter(count, Fried_err.detach().cpu(), color=color)
                if count == 1:
                    ax[2].plot(
                        [], [],
                        ' ',
                        label='(True) Fried = {:.3f}'.format(
                            all_measures['Fried_true'].detach().cpu().numpy()))

            # Now at the end of the loop, we decide if this set needs to plotted or not
            # by checking that the loss
            if loss > loss_max or loss < loss_min:
                avg_tags_trim = stats.trim_mean(_all_tags_pred,
                                                trimming).squeeze()
                percentiles_50 = np.percentile(_all_tags_pred, [25, 75],
                                               axis=0).squeeze()
                percentiles_68 = np.percentile(_all_tags_pred, [16, 84],
                                               axis=0).squeeze()
                percentiles_95 = np.percentile(_all_tags_pred, [2.5, 97.5],
                                               axis=0).squeeze()

                x_vals = np.arange(n_screens)
                alp = 1
                ax[0].plot(avg_tags_trim,
                           label='Mean',
                           color='tab:red',
                           zorder=50)
                ax[0].fill_between(x_vals,
                                   percentiles_50[0],
                                   percentiles_50[1],
                                   color='gold',
                                   alpha=alp,
                                   label='50% CI',
                                   zorder=5)
                ax[0].fill_between(x_vals,
                                   percentiles_68[0],
                                   percentiles_50[0],
                                   color='cadetblue',
                                   alpha=alp,
                                   label='68% CI',
                                   zorder=4)
                ax[0].fill_between(x_vals,
                                   percentiles_50[1],
                                   percentiles_68[1],
                                   color='cadetblue',
                                   alpha=alp,
                                   zorder=4)
                ax[0].fill_between(x_vals,
                                   percentiles_95[0],
                                   percentiles_68[0],
                                   color='blue',
                                   label='95% CI',
                                   alpha=alp,
                                   zorder=3)
                ax[0].fill_between(x_vals,
                                   percentiles_68[1],
                                   percentiles_95[1],
                                   color='blue',
                                   alpha=alp,
                                   zorder=3)

                ax[3].axis('off')  # Hide axis
                recap_info = f'LOSS TERMS:\nTotal Loss: {loss.item():.4g}\n'
                # the individual losses
                for key, value in losses.items():
                    recap_info += f'{key}: {value.item():.4g}\n'
                recap_info += '-------------------\nPARAMETERS:\n'
                # then the single parameters
                for key, value in all_measures.items():
                    recap_info += f'{key}: {value:.4g}\n'
                ax[3].text(0.5,
                           0.5,
                           recap_info,
                           horizontalalignment='center',
                           verticalalignment='center',
                           fontsize=10,
                           color='black')

                ax[0].set_yscale('log')
                ax[0].set_ylabel('J')
                ax[0].set_xlabel('# screen')
                ax[0].legend()
                ax[1].set_yscale('log')
                ax[1].set_ylabel('J')
                ax[1].set_xlabel('# screen')
                red_patch = mpatches.Patch(color=color, label='All speckles')
                handles, labels = ax[1].get_legend_handles_labels()
                handles.extend([blue_patch, red_patch])
                ax[1].legend(handles=handles)
                ax[2].set_xlabel('# averaged speckles')
                ax[2].set_ylabel('Fried relative error')
                ax[2].legend(frameon=False)
                ax[2].set_yscale('log')
                fig.tight_layout()
                plt.subplots_adjust(top=0.92)
                plt.suptitle('Effect of averaging speckle predictions')
                plt.savefig(
                    f'{dirname}/average_ensemble_loss{loss.item():.4g}.png')
                loss_max = max(loss, loss_max)
                loss_min = min(loss, loss_min)
                ensemble_count += 1

            plt.close()

            if ensemble_count > n_ensembles_to_plot:
                break


def average_speckle_input(conf: dict,
                          test_set: list,
                          device: Device,
                          model: nn.Torch,
                          criterion: ComposableLoss,
                          n_ensembles_to_plot: int = 100) -> None:
    """Test to see if averaging the speckle patterns (before the prediction)
    improves the results. This function is then going to plot the relative
    error over the screen tags and the Fried parameter to make this evaluation.

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

    dirname = f'{data_directory}/{model_name}_score/effect_averaging'
    ensure_directory(dirname)

    # group the sets that have the same n[1]
    grouped_test_set: Dict = {}
    for n in test_set:
        key = tuple(n[1])
        if key not in grouped_test_set:
            grouped_test_set[key] = []
        grouped_test_set[key].append(n)
    print('\nChecking if averaging speckle patterns improves results')
    print(f'Number of samples: {len(test_set)}')
    print(f'Number of speckle groups: {len(grouped_test_set)}')

    # For each group compare the model prediction to the exact tag
    ensemble = EnsembleModel(conf, device)
    with torch.no_grad():
        model.eval()

        for ensemble_count, (key,
                             value) in enumerate(grouped_test_set.items()):
            avg_speckle = None
            cmap = plt.get_cmap('coolwarm')
            norm = plt.Normalize(1, len(value))

            if ensemble_count > n_ensembles_to_plot:
                continue

            for count, speckle in enumerate(value, 1):
                color = cmap(norm(count))

                if avg_speckle is None:
                    avg_speckle = speckle
                else:
                    avg_speckle = (torch.add(avg_speckle[0],
                                             speckle[0]), *avg_speckle[1:])

                # Average only the speckle pattern (first element)
                avg_speckle_divided = (torch.div(avg_speckle[0],
                                                 count), *avg_speckle[1:])

                output, target, _ = ensemble(model, [avg_speckle_divided])
                loss, losses = criterion(output, target)

                if count == 1:
                    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

                    recovered_tag_true = criterion.get_J(target)
                    ax[0].plot(recovered_tag_true.squeeze(0).detach().cpu(),
                               '*',
                               label='True',
                               color='tab:green',
                               zorder=100)
                    blue_patch = mpatches.Patch(color=color,
                                                label='One speckle')

                ax[1].plot((torch.abs(output - target) /
                            (target + 1e-7)).flatten().detach().cpu(),
                           color=color)

                # Get the Cn2 profile and the recovered tags
                Cn2_pred = criterion.reconstruct_cn2(output)
                Cn2_true = criterion.reconstruct_cn2(target)
                recovered_tag_pred = criterion.get_J(output / count)
                ax[0].plot(recovered_tag_pred.squeeze(0).detach().cpu(),
                           '.',
                           color=color)
                # and get all the measures
                all_measures = criterion._get_all_measures(
                    output, target, Cn2_pred, Cn2_true)

                Fried_err = torch.abs(
                    all_measures['Fried_true'] -
                    all_measures['Fried_pred']) / all_measures['Fried_true']
                ax[2].scatter(count, Fried_err.detach().cpu(), color=color)
                if count == 1:
                    ax[2].plot(
                        [], [],
                        ' ',
                        label='(True) Fried = {:.3f}'.format(
                            all_measures['Fried_true'].detach().cpu().numpy()))

            ax[0].set_yscale('log')
            ax[0].set_ylabel('J')
            ax[0].set_xlabel('# screen')
            red_patch = mpatches.Patch(color=color, label='All speckles')
            handles, labels = ax[0].get_legend_handles_labels()
            handles.extend([blue_patch, red_patch])
            ax[0].legend(handles=handles)
            ax[1].set_xlabel('# screen')
            ax[1].set_ylabel('J relative error ')
            ax[2].set_xlabel('# averaged speckles')
            ax[2].set_ylabel('Fried relative error')
            ax[1].set_yscale('log')
            ax[2].set_yscale('log')
            ax[2].legend(frameon=False)
            fig.tight_layout()
            plt.subplots_adjust(top=0.92)
            plt.suptitle('Effect of averaging speckle patterns')
            plt.savefig(f'{dirname}/average_speckle_loss{loss.item():.4g}.png')
            plt.close()


def screen_errors(conf: dict,
                  device: Device,
                  J_pred: Tensor,
                  J_true: Tensor,
                  nbins: int = 20,
                  trimming: float = 0.1) -> None:
    """Plot the relative error of Cn2 for each screen.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    device : torch.device
        The device on which the data are stored
    J_pred : torch.Tensor
        The predicted J profile
    J_true : torch.Tensor
        The true J profile
    nbins : int, optional
        Number of bins to use for the histograms
    trimming : float, optional
        The fraction of data to trim from each end of the distribution
    """
    data_directory = conf['speckle']['datadirectory']
    model_name = conf['model']['name']
    n_screens = conf['speckle']['nscreens']

    dirname = f'{data_directory}/{model_name}_score'
    ensure_directory(dirname)

    # Plot the distribution of each tag element
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for si in range(n_screens):

        if device == 'cpu':
            data_pred = np.asarray(J_pred)[:, 0, si]
            data_true = np.asarray(J_true)[:, 0, si]
        else:
            data_pred = np.asarray([d.detach().cpu().numpy()
                                    for d in J_pred])[:, 0, si]
            data_true = np.asarray([d.detach().cpu().numpy()
                                    for d in J_true])[:, 0, si]
        # Define the bins
        bins = np.linspace(min(data_true), max(data_true), nbins + 1)

        # Digitize the data to get bin indices
        bin_indices = np.digitize(data_true, bins)

        # Compute the relative error for each bin
        relative_errors = []
        percentiles_50 = []
        percentiles_68 = []
        percentiles_95 = []

        for bi in range(1, len(bins)):
            model_bin_values = data_pred[bin_indices == bi]
            true_bin_values = data_true[bin_indices == bi]

            if len(true_bin_values) > 0:
                relative_error = np.abs(
                    (model_bin_values - true_bin_values) / true_bin_values)
                relative_errors.append(
                    stats.trim_mean(relative_error, trimming))
                percentiles_50.append(
                    np.percentile(relative_error, [25, 75], axis=0))
                percentiles_68.append(
                    np.percentile(relative_error, [16, 84], axis=0))
                percentiles_95.append(
                    np.percentile(relative_error, [2.5, 97.5], axis=0))
            else:
                relative_errors.append(0)
                percentiles_50.append([0, 0])
                percentiles_68.append([0, 0])
                percentiles_95.append([0, 0])

        percentiles_50 = np.asarray(percentiles_50)
        percentiles_68 = np.asarray(percentiles_68)
        percentiles_95 = np.asarray(percentiles_95)
        # Plot the relative errors
        axs[si // 4, si % 4].plot(bins[:-1],
                                  relative_errors,
                                  label='Mean',
                                  color='tab:red',
                                  zorder=50)
        axs[si // 4, si % 4].set_title(f'Screen {si}')
        # and the percentiles
        axs[si // 4, si % 4].fill_between(bins[:-1],
                                          percentiles_50[:, 0],
                                          percentiles_50[:, 1],
                                          color='gold',
                                          alpha=0.5,
                                          label='50% CI',
                                          zorder=5)
        axs[si // 4, si % 4].fill_between(bins[:-1],
                                          percentiles_68[:, 0],
                                          percentiles_50[:, 0],
                                          color='cadetblue',
                                          alpha=0.5,
                                          label='68% CI',
                                          zorder=4)
        axs[si // 4, si % 4].fill_between(bins[:-1],
                                          percentiles_50[:, 1],
                                          percentiles_68[:, 1],
                                          color='cadetblue',
                                          alpha=0.5,
                                          zorder=4)
        axs[si // 4, si % 4].fill_between(bins[:-1],
                                          percentiles_95[:, 0],
                                          percentiles_68[:, 0],
                                          color='blue',
                                          label='95% CI',
                                          alpha=0.5,
                                          zorder=3)
        axs[si // 4, si % 4].fill_between(bins[:-1],
                                          percentiles_68[:, 1],
                                          percentiles_95[:, 1],
                                          color='blue',
                                          alpha=0.5,
                                          zorder=3)
        axs[si // 4, si % 4].set_yscale('symlog', linthresh=0.1)
    axs[0, 1].legend()
    for ax in axs.flatten():
        ax.axhline(
            y=0,
            linestyle='--',
            color='black',
        )
    vals = axs[0, 0].get_yticks()
    axs[0, 0].set_yticklabels(['{:.0f}'.format(x * 100) + '%' for x in vals])
    vals = axs[-1, -1].get_yticks()
    axs[-1, -1].set_yticklabels(['{:.0f}'.format(x * 100) + '%' for x in vals])
    for i in range(1, n_screens - 1):
        axs.flat[i].sharey(axs.flat[0])
    plt.suptitle('Relative Error of J')
    plt.tight_layout()
    figname = f'{dirname}/{model_name}_Jerrors'
    plt.savefig(f'{figname}.png')
    plt.close()
