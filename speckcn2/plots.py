import torch
import numpy as np
import matplotlib.pyplot as plt
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
    ensemble = conf['preproc']['ensemble']
    hs = conf['speckle']['splits']
    nscreens = conf['speckle']['nscreens']
    if len(hs) != nscreens:
        print(
            'WARNING: The number of screens does not match the number of splits'
        )
        return

    fig, axs = plt.subplots(1, 3 + ensemble, figsize=(4 * (2 + ensemble), 3.5))

    # (1) Plot the input images
    for n in range(ensemble):
        axs[n].imshow(inputs[ensemble * i + n].detach().cpu().squeeze(),
                      cmap='bone')
    axs[1].set_title(f'Input {ensemble} images')

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
    axs[-2].plot(Cn2_true.squeeze(0).detach().cpu(), hs, 'o', label='True')
    axs[-2].plot(Cn2_pred.squeeze(0).detach().cpu(),
                 hs,
                 '.',
                 color='tab:red',
                 label='Predicted')
    axs[-2].set_xscale('log')
    axs[-2].set_yscale('log')
    axs[-2].set_xlabel(r'$Cn^2$')
    axs[-2].set_ylabel('Altitude [m]')

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
    plt.savefig(f'{data_dir}/{model_name}_score/{counter}.png')
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

    ensure_directory(f'{data_dir}/result_plots')

    # plot the loss
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.plot(model.epoch, model.loss, label='Training loss')
    axs.plot(model.epoch, model.val_loss, label='Validation loss')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    axs.legend()
    plt.title(f'Model: {model_name}')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/result_plots/loss_{model_name}.png')
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

    ensure_directory(f'{data_dir}/result_plots')

    # plot the loss
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    axs.plot(model.epoch, model.time, label='Time per epoch')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Time [s]')
    axs.legend()
    plt.title(f'Model: {model_name}')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/result_plots/time_{model_name}.png')
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

    ensure_directory(f'{data_dir}/result_plots')

    # plot the loss
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    for key in ['JMAE', 'Fried', 'Isoplanatic', 'Scintillation_w']:
        loss = [d[key].detach().cpu() for d in test_losses]
        # Generate bin edges on a log scale
        bins = np.logspace(np.log10(min(loss)), np.log10(max(loss)), num=30)
        axs.hist(loss, bins=bins, alpha=0.5, label=key)
    axs.set_xlabel('Loss')
    axs.set_ylabel('Frequency')
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.legend()
    plt.title(f'Model: {model_name}')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/result_plots/histo_losses_{model_name}.png')
    plt.close()


    #this is receiveing the loss but it would like the _measures for this
def plot_param_vs_loss(conf: dict, test_losses: list[dict], data_dir: str,
                       measures: list) -> None:
    """Plots the parameter vs the loss.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    test_losses : list[dict]
        List of all the losses of the test set
    data_dir : str
        The directory where the data is stored
    param : str
        The parameter to plot
    """
    model_name = conf['model']['name']

    ensure_directory(f'{data_dir}/result_plots')

    for param, name in zip(
        ['Fried_true', 'Isoplanatic_true', 'Scintillation_w_true'],
        ['Fried parameter', 'Isoplanatic angle', '(weak) Scintillation index'
         ]):
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))

        # Extract the param and the sum of all values from each dictionary
        params = [d[param].detach().cpu() for d in measures]
        sums = [sum(d.values()).detach().cpu() for d in test_losses]

        # Pair these values together and sort them based on the value of param
        pairs = sorted(zip(params, sums))

        # Unzip the pairs back into two lists
        params, sums = zip(*pairs)

        # Plot the data
        axs.plot(params, sums, 'o')
        axs.set_xlabel(name)
        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.set_ylabel('Total loss')
        plt.title(f'Model: {model_name}')
        plt.tight_layout()
        plt.savefig(f'{data_dir}/result_plots/{param}_vs_sum_{model_name}.png')
        plt.close()
