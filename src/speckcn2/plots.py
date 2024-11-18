from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from torch import device as Device
from torch import nn

from speckcn2.loss import ComposableLoss
from speckcn2.mlmodels import EnsembleModel
from speckcn2.utils import ensure_directory


def score_plot(
    conf: dict,
    inputs: torch.Tensor,
    tags: list,
    loss: torch.Tensor,
    losses: dict,
    i: int,
    counter: int,
    measures: dict,
    Cn2_pred: torch.Tensor,
    Cn2_true: torch.Tensor,
    recovered_tag_pred: torch.Tensor,
    recovered_tag_true: torch.Tensor,
) -> None:
    """Plots side by side:
    - [0:Nensemble] the input images (single or ensemble)
    - [-3] the predicted/exact tags J
    - [-2] the Cn2 profile
    - [-1] the different information of the loss
    normalize value in model units.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    inputs : torch.Tensor
        The input speckle patterns
    tags : list
        The exact tags of the data
    loss : torch.Tensor
        The total loss of the model (for this prediction)
    losses : dict
        The individual losses of the model
    i : int
        The batch index of the image
    counter : int
        The global index of the image
    measures : dict
        The different measures of the model
    Cn2_pred : torch.Tensor
        The predicted Cn2 profile
    Cn2_true : torch.Tensor
        The true Cn2 profile
    recovered_tag_pred : torch.Tensor
        The predicted tags
    recovered_tag_true : torch.Tensor
        The true tags
    """
    model_name = conf['model']['name']
    data_dir = conf['speckle']['datadirectory']
    ensemble = conf['preproc'].get('ensemble', 1)
    hs = conf['speckle']['splits']
    nscreens = conf['speckle']['nscreens']
    if len(hs) != nscreens:
        print(
            'WARNING: The number of screens does not match the number of splits'
        )
        return

    dirname = f'{data_dir}/{model_name}_score/single-shot_predictions'
    ensure_directory(dirname)

    fig, axs = plt.subplots(1, 3 + ensemble, figsize=(4 * (2 + ensemble), 3.5))

    # (1) Plot the input images
    for n in range(ensemble):
        img = inputs[ensemble * i + n].detach().cpu().squeeze().abs()
        axs[n].imshow(img, cmap='bone')
    title_string = f'Input {ensemble} images' if ensemble > 1 else 'Input single image'
    axs[1].set_title(title_string)

    # (2) Plot J vs nscreens
    axs[-3].plot(recovered_tag_true.squeeze(0).detach().cpu(),
                 'o',
                 label='True')
    axs[-3].plot(recovered_tag_pred.squeeze(0).detach().cpu(),
                 '.',
                 color='tab:red',
                 label='Predicted')
    axs[-3].set_yscale('log')
    axs[-3].set_ylabel('J')
    axs[-3].set_xlabel('# screen')
    axs[-3].legend()

    # (3) Plot Cn2 vs altitude
    axs[-2].plot(hs, Cn2_true.squeeze(0).detach().cpu(), 'o', label='True')
    axs[-2].plot(hs,
                 Cn2_pred.squeeze(0).detach().cpu(),
                 '.',
                 color='tab:red',
                 label='Predicted')
    axs[-2].set_xscale('log')
    axs[-2].set_yscale('log')
    axs[-2].set_ylabel(r'$Cn^2$')
    axs[-2].set_xlabel('Altitude [m]')

    # (4) Plot the recap information
    axs[-1].axis('off')  # Hide axis
    recap_info = f'LOSS TERMS:\nTotal Loss: {loss.item():.4g}\n'
    # the individual losses
    for key, value in losses.items():
        recap_info += f'{key}: {value.item():.4g}\n'
    recap_info += '-------------------\nPARAMETERS:\n'
    # then the single parameters
    for key, value in measures.items():
        recap_info += f'{key}: {value:.4g}\n'
    axs[-1].text(0.5,
                 0.5,
                 recap_info,
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=10,
                 color='black')

    plt.tight_layout()
    plt.savefig(f'{dirname}/single_speckle_loss{loss.item():.4g}.png')
    plt.close()


def plot_loss(conf: dict, model, data_dir: str) -> None:
    """Plots the loss of the model.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    model : torch.nn.Module
        The model to plot the loss of
    data_dir : str
        The directory where the data is stored
    """

    model_name = conf['model']['name']
    data_dir = conf['speckle']['datadirectory']

    dirname = f'{data_dir}/{model_name}_score'
    ensure_directory(dirname)

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.plot(model.epoch, model.loss, label='Training loss')
    axs.plot(model.epoch, model.val_loss, label='Validation loss')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    axs.set_yscale('log')
    axs.legend()
    plt.title(f'Model: {model_name}')
    plt.tight_layout()
    plt.savefig(f'{dirname}/loss_{model_name}.png')
    plt.close()


def plot_time(conf: dict, model, data_dir: str) -> None:
    """Plots the time per epoch of the model.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    model : torch.nn.Module
        The model to plot the loss of
    data_dir : str
        The directory where the data is stored
    """

    model_name = conf['model']['name']
    data_dir = conf['speckle']['datadirectory']

    dirname = f'{data_dir}/{model_name}_score'
    ensure_directory(dirname)

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.plot(model.epoch, model.time, label='Time per epoch')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Time [s]')
    axs.legend()
    plt.title(f'Model: {model_name}')
    plt.tight_layout()
    plt.savefig(f'{dirname}/time_{model_name}.png')
    plt.close()


def plot_histo_losses(conf: dict, test_losses: list[dict],
                      data_dir: str) -> None:
    """Plots the histogram of the losses.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    test_losses : list[dict]
        List of all the losses of the test set
    data_dir : str
        The directory where the data is stored
    """
    model_name = conf['model']['name']
    data_dir = conf['speckle']['datadirectory']

    dirname = f'{data_dir}/{model_name}_score/histo_losses'
    ensure_directory(dirname)

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    for key in ['MAE', 'Fried', 'Isoplanatic', 'Scintillation_w']:
        loss = [d[key].detach().cpu() for d in test_losses]
        bins = np.logspace(np.log10(min(loss)), np.log10(max(loss)),
                           num=50).tolist()
        axs.hist(loss, bins=bins, alpha=0.5, label=key, density=True)
    axs.set_xlabel('Loss')
    axs.set_ylabel('Frequency')
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.legend()
    plt.title(f'Model: {model_name}')
    plt.tight_layout()
    plt.savefig(f'{dirname}/histo_losses_{model_name}.png')
    plt.close()


def plot_param_vs_loss(conf: dict,
                       test_losses: list[dict],
                       data_dir: str,
                       measures: list,
                       no_sign: bool = False,
                       nbins: int = 10,
                       linear_bins: bool = False) -> None:
    """Plots the parameter vs the loss. Optionally, it also plots the detailed
    histo for all the bins for the desired metrics.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    test_losses : list[dict]
        List of all the losses of the test set
    data_dir : str
        The directory where the data is stored
    measures : list
        The measures of the model
    no_sign : bool
        If True, it will plot the abs of the relative error
    nbins : int
        The number of bins in which to partition the data
    linear_bins : bool
        If True, the bins are linearly spaced, otherwise they are log spaced
    """
    model_name = conf['model']['name']
    data_dir = conf['speckle']['datadirectory']

    dirname = f'{data_dir}/{model_name}_score'
    ensure_directory(dirname)

    for param, lname, name, units in zip(
        ['Fried_true', 'Isoplanatic_true', 'Scintillation_w_true'],
        ['Fried', 'Isoplanatic', 'Scintillation_w'],
        ['Fried parameter', 'Isoplanatic angle', 'Rytov index'],
        ['[m]', '[rad]', '[1]'],
    ):

        dirname = f'{data_dir}/{model_name}_score'
        p_data = [d[param].detach().cpu() for d in measures]
        if no_sign:
            l_data = [d[lname].detach().cpu() for d in test_losses]
        else:
            pname = lname.split('_true')[0] + '_pred'
            l_data = [((d[pname] - d[param]) / d[param]).detach().cpu()
                      for d in measures]

        pairs = sorted(zip(p_data, l_data))
        params, loss = zip(*pairs)
        params = np.array(params)
        loss = np.array(loss)

        if linear_bins:
            bins = np.linspace(min(params), max(params), num=nbins)
        else:
            bins = np.logspace(np.log10(min(params)),
                               np.log10(max(params)),
                               num=nbins)
        bin_indices = np.digitize(params, bins)
        bin_means = [
            loss[bin_indices == i].mean() if np.any(bin_indices == i) else 0
            for i in range(1, len(bins))
        ]
        bin_stds = [
            loss[bin_indices == i].std() if np.any(bin_indices == i) else 0
            for i in range(1, len(bins))
        ]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Plotting the results
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        axs.errorbar(bin_centers,
                     bin_means,
                     yerr=bin_stds,
                     marker='o',
                     linestyle='-',
                     alpha=0.75)

        # Plot error reference lines+shade
        axs.axhline(y=1.0, linestyle='--', color='tab:red', label='100% error')
        axs.axhline(y=0.5,
                    linestyle='--',
                    color='tab:orange',
                    label='50% error')
        axs.axhline(y=0.1,
                    linestyle='--',
                    color='tab:green',
                    label='10% error')
        axs.axhline(y=-1.0, linestyle='--', color='tab:red')
        axs.axhline(
            y=-0.5,
            linestyle='--',
            color='tab:orange',
        )
        axs.axhline(
            y=-0.1,
            linestyle='--',
            color='tab:green',
        )
        axs.axhline(
            y=0,
            linestyle='--',
            color='black',
        )
        plt.tight_layout()
        x_min, x_max = axs.get_xlim()
        axs.fill_between([x_min, x_max],
                         -0.1,
                         0.1,
                         color='tab:green',
                         alpha=0.1)
        axs.fill_between([x_min, x_max],
                         0.1,
                         0.5,
                         color='tab:orange',
                         alpha=0.1)
        axs.fill_between([x_min, x_max],
                         -0.5,
                         -0.1,
                         color='tab:orange',
                         alpha=0.1)
        axs.fill_between([x_min, x_max], 0.5, 1.0, color='tab:red', alpha=0.1)
        axs.fill_between([x_min, x_max],
                         -1.0,
                         -0.5,
                         color='tab:red',
                         alpha=0.1)
        axs.set_xlabel(f'{name} {units}')
        axs.set_xscale('log')
        axs.set_yscale('symlog', linthresh=0.1)
        axs.set_ylabel('Relative error')
        yticks = [-1, -0.5, -0.1, 0, 0.1, 0.5, 1]
        plt.yticks(yticks)
        yticklabels = ['-100%', '-50%', '-10%', '0', '10%', '50%', '100%']
        plt.gca().set_yticklabels(yticklabels)
        plt.title(f'Model: {model_name}')
        plt.tight_layout()
        plt.savefig(f'{dirname}/{param}_vs_sum_{model_name}.png')
        plt.close()

        # If specified, plot the histogram per single bin
        if conf['preproc'].get(lname + '_details', False):
            print(f'\nComputing {lname} details')
            dirname = f'{data_dir}/{model_name}_score/{lname}_bin_details'
            ensure_directory(dirname)

            for idx, single_bin in enumerate(bin_centers):
                l_data = loss[bin_indices == idx]
                if len(l_data) == 0:
                    continue
                fig, axs = plt.subplots(1, 1, figsize=(5, 5))
                axs.hist(l_data, bins=50, alpha=0.5, density=True)
                mu = np.mean(l_data)
                sigma = np.std(l_data)
                if no_sign:
                    print(
                        'Warning: you are requesting the analysis of absolute value using'
                        + ' normal gaussian assumption.' +
                        'This is not correct and the error will be overestimated.'
                    )
                print(
                    f'{lname} = {single_bin:.3f} -> mu = {mu:.3f}, sigma = {sigma:.3f}'
                )
                if sigma > 0:
                    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
                    x = x[x > min(l_data)]
                    x = x[x < max(l_data)]
                    axs.plot(x,
                             stats.norm.pdf(x, mu, sigma),
                             label=f'Average err: {mu:.3f}, Std: {sigma:.3f}')
                axs.set_xlabel(f'Relative error {lname}')
                axs.set_ylabel('Frequency')
                axs.legend()
                plt.title(f'{lname} value = {single_bin:.3f} {units}')
                plt.tight_layout()
                plt.savefig(f'{dirname}/{lname}_bin{idx}.png')
                plt.close()


def plot_param_histo(conf: dict, test_losses: list[dict], data_dir: str,
                     measures: list) -> None:
    """Plots the histograms of different parameters.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    test_losses : list[dict]
        List of all the losses of the test set
    data_dir : str
        The directory where the data is stored
    measures : list
        The measures of the model
    """
    model_name = conf['model']['name']
    data_dir = conf['speckle']['datadirectory']

    dirname = f'{data_dir}/{model_name}_score'
    ensure_directory(dirname)

    for param_model, param_true, name, units in zip(
        ['Fried_pred', 'Isoplanatic_pred', 'Scintillation_w_pred'],
        ['Fried_true', 'Isoplanatic_true', 'Scintillation_w_true'],
        ['Fried parameter', 'Isoplanatic angle', 'Rytov index'],
        ['[m]', '[rad]', '[1]'],
    ):
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))

        params_model = [d[param_model].detach().cpu() for d in measures]
        params_true = [d[param_true].detach().cpu() for d in measures]

        pairs = sorted(zip(params_true, params_model))
        params_true, params_model = zip(*pairs)
        params_true = np.array(params_true)
        params_model = np.array(params_model)

        bins = np.logspace(np.log10(min(params_true)),
                           np.log10(max(params_true)),
                           num=50).tolist()
        axs.hist(params_true,
                 bins=bins,
                 alpha=0.5,
                 label=param_true,
                 density=True)

        bins = np.logspace(np.log10(min(params_model)),
                           np.log10(max(params_model)),
                           num=50).tolist()
        axs.hist(params_model,
                 bins=bins,
                 alpha=0.5,
                 label=param_model,
                 density=True)

        axs.set_xlabel(f'{name} {units}')
        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.set_ylabel('Frequency')
        axs.legend()
        plt.title(f'Model: {model_name}')
        plt.tight_layout()
        plt.savefig(f'{dirname}/histo_{param_true}_{model_name}.png')
        plt.close()


def plot_J_error_details(conf: dict,
                         tags_true: list,
                         tags_pred: list,
                         nbins: int = 10,
                         linear_bins: bool = False) -> None:
    """Function to plot the histograms per single bin of each single screen tag
    to quantify the relative error as a function of J.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    tags_true : list
        The true tags of the validation set
    tags_pred : list
        The predicted tags of the validation set
    nbins : int
        The number of bins in which to partition the data
    linear_bins : bool
        If True, the bins are linearly spaced, otherwise they are log spaced
    """

    nscreens = conf['speckle']['nscreens']
    data_dir = conf['speckle']['datadirectory']
    model_name = conf['model']['name']

    dirname = f'{data_dir}/{model_name}_score/J_bin_details'
    ensure_directory(dirname)

    if conf['preproc'].get('J_details', False):
        for screen_id in range(nscreens):
            print(f'\nComputing screen-{screen_id} details')

            # Collect the data
            params = []
            loss = []
            for i in range(len(tags_true)):
                params.append(tags_true[i][0,
                                           screen_id].detach().cpu().numpy())
                loss.append((
                    (tags_pred[i][0, screen_id] - tags_true[i][0, screen_id]) /
                    (tags_true[i][0, screen_id])).detach().cpu().numpy())
            params = np.array(params)
            loss = np.array(loss)

            if linear_bins:
                bins = np.linspace(min(params), max(params), num=nbins)
            else:
                bins = np.logspace(np.log10(min(params)),
                                   np.log10(max(params)),
                                   num=nbins)
            bin_indices = np.digitize(params, bins)
            # get the average and std of the error per bin of J[screen_id]
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            for idx, single_bin in enumerate(bin_centers):
                l_data = loss[bin_indices == idx]
                if len(l_data) == 0:
                    continue
                fig, axs = plt.subplots(1, 1, figsize=(5, 5))
                axs.hist(l_data, bins=50, alpha=0.5, density=True)
                mu = np.mean(l_data)
                sigma = np.std(l_data)
                print(
                    f'J-{screen_id} = {single_bin:.3g} -> mu = {mu:.3f}, sigma = {sigma:.3f}'
                )
                if sigma > 0:
                    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
                    x = x[x > min(l_data)]
                    x = x[x < max(l_data)]
                    axs.plot(x,
                             stats.norm.pdf(x, mu, sigma),
                             label=f'Average err: {mu:.3f}, Std: {sigma:.3f}')
                axs.set_xlabel(f'Relative error J (screen-{screen_id})')
                axs.set_ylabel('Frequency')
                axs.legend()
                plt.title(f'J (screen-{screen_id}) value = {single_bin:.3g}')
                plt.tight_layout()
                plt.savefig(f'{dirname}/Jscreen{screen_id}_bin{idx}.png')
                plt.close()


def plot_samples_in_ensemble(conf: dict,
                             test_set: list,
                             device: Device,
                             model: nn.Torch,
                             criterion: ComposableLoss,
                             trimming: float = 0.2,
                             n_max_plots: int = 100) -> None:
    """Plot the prediction over a sample and compare it with the ones
    from its ensemble.

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
    trimming : float
        The trimming to use for the mean
    n_max_plots : int
        The maximum number of plots
    """

    data_directory = conf['speckle']['datadirectory']
    model_name = conf['model']['name']
    n_screens = conf['speckle']['nscreens']

    dirname = f'{data_directory}/{model_name}_score/single-shot_predictions'
    ensure_directory(dirname)

    # group the sets that have the same n[1]
    grouped_test_set: Dict = {}
    for n in test_set:
        key = tuple(n[1])
        if key not in grouped_test_set:
            grouped_test_set[key] = []
        grouped_test_set[key].append(n)
    print('\nAnalysis of single shot predictions')
    print(f'Number of samples: {len(test_set)}')
    print(f'Number of speckle groups: {len(grouped_test_set)}')
    # Define a random probability to plot each ensemble
    p_plot = n_max_plots / len(grouped_test_set)

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

            for count, speckle in enumerate(value, 1):
                output, target, _ = ensemble(model, [speckle])
                _outputs.append(output.detach().cpu().numpy())
                loss, losses = criterion(output, target)

                # Get the Cn2 profile and the recovered tags
                Cn2_pred = criterion.reconstruct_cn2(output)
                Cn2_true = criterion.reconstruct_cn2(target)
                recovered_tag_pred = criterion.get_J(output)
                _all_tags_pred.append(recovered_tag_pred.detach().cpu())
                # and get all the measures
                all_measures = criterion._get_all_measures(
                    output, target, Cn2_pred, Cn2_true)

                if count == 1:
                    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                    loss_0 = loss

                    # (0) Plot the speckle pattern
                    ax[0].axis('off')  # Hide axis
                    ax[0].imshow(speckle[0][0, :, :], cmap='bone')

                    # (1) Plot J vs nscreens
                    recovered_tag_true = criterion.get_J(target)
                    ax[1].plot(recovered_tag_true.squeeze(0).detach().cpu(),
                               '*',
                               label='True',
                               color='tab:green',
                               markersize=10,
                               markeredgecolor='black',
                               zorder=100)
                    ax[1].plot(recovered_tag_pred.squeeze(0).detach().cpu(),
                               'o',
                               label='This speckle',
                               color='tab:red',
                               markersize=7,
                               markeredgecolor='black',
                               zorder=90)

                    # (2) Plot the parameters of this speckle prediction
                    ax[2].axis('off')  # Hide axis
                    recap_info = f'LOSS TERMS:\nTotal Loss: {loss.item():.4g}\n'
                    # the individual losses
                    for key, value in losses.items():
                        recap_info += f'{key}: {value.item():.4g}\n'
                    recap_info += '-------------------\nPARAMETERS:\n'
                    # then the single parameters
                    for key, value in all_measures.items():
                        recap_info += f'{key}: {value:.4g}\n'
                    ax[2].text(0.5,
                               0.5,
                               recap_info,
                               horizontalalignment='center',
                               verticalalignment='center',
                               fontsize=10,
                               color='black')

            # Now at the end of the loop, we decide if this set needs to plotted or not
            # by checking that the loss is uncommon, or via a random probability
            if loss_0 > loss_max or loss_0 < loss_min or np.random.rand(
            ) < p_plot:
                avg_tags_trim = stats.trim_mean(_all_tags_pred,
                                                trimming).squeeze()
                percentiles_50 = np.percentile(_all_tags_pred, [25, 75],
                                               axis=0).squeeze()
                percentiles_68 = np.percentile(_all_tags_pred, [16, 84],
                                               axis=0).squeeze()
                percentiles_95 = np.percentile(_all_tags_pred, [2.5, 97.5],
                                               axis=0).squeeze()

                x_vals = np.arange(n_screens)
                alp = 0.3
                ax[1].plot(avg_tags_trim,
                           label='Mean',
                           color='tab:red',
                           zorder=50)
                ax[1].fill_between(x_vals,
                                   percentiles_50[0],
                                   percentiles_50[1],
                                   color='gold',
                                   alpha=alp,
                                   label='50% CI',
                                   zorder=5)
                ax[1].fill_between(x_vals,
                                   percentiles_68[0],
                                   percentiles_50[0],
                                   color='cadetblue',
                                   alpha=alp,
                                   label='68% CI',
                                   zorder=4)
                ax[1].fill_between(x_vals,
                                   percentiles_50[1],
                                   percentiles_68[1],
                                   color='cadetblue',
                                   alpha=alp,
                                   zorder=4)
                ax[1].fill_between(x_vals,
                                   percentiles_95[0],
                                   percentiles_68[0],
                                   color='blue',
                                   label='95% CI',
                                   alpha=alp,
                                   zorder=3)
                ax[1].fill_between(x_vals,
                                   percentiles_68[1],
                                   percentiles_95[1],
                                   color='blue',
                                   alpha=alp,
                                   zorder=3)

                ax[1].set_yscale('log')
                ax[1].set_ylabel('J')
                ax[1].set_xlabel('# screen')
                ax[1].legend()
                fig.tight_layout()
                plt.subplots_adjust(top=0.92)
                plt.suptitle(
                    'Prediction from a single speckle, compared to similar')
                plt.savefig(
                    f'{dirname}/single_speckle_loss{loss_0.item():.4g}.png')
                loss_max = max(loss_0, loss_max)
                loss_min = min(loss_0, loss_min)
                ensemble_count += 1

            plt.close()

            if ensemble_count >= n_max_plots:
                break
