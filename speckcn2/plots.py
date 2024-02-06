import numpy as np
import torch
from typing import Callable
import matplotlib.pyplot as plt
from speckcn2.utils import ensure_directory


def score_plot(
    model_name: str,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    tags: list,
    loss: torch.Tensor,
    i: int,
    counter: int,
    data_dir: str,
    recover_tag: list[Callable],
) -> None:
    """Plots side by side the input image, the predicted/exact tags and their
    normalize value in model units.

    Parameters
    ----------
    model_name : str
        The name of the model
    inputs : torch.Tensor
        The input speckle patterns
    outputs : torch.Tensor
        The predicted screen tags
    tags : list
        The exact tags of the data
    loss : torch.Tensor
        The loss of the model
    i : int
        The batch index of the image
    counter : int
        The global index of the image
    data_dir : str
        The directory where the data is stored
    recover_tag : list
        List of functions to recover each tag
    """

    # Plot the image and output side by side
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    axs[0].imshow(inputs[i].detach().cpu().squeeze(), cmap='bone')
    axs[0].set_title('Loss: {:.4f}'.format(loss.item()))
    recovered_tag_true = np.asarray([
        recover_tag[j](tags[i][j].detach().cpu().numpy())
        for j in range(len(tags[i]))
    ])
    axs[1].plot(10**(recovered_tag_true), 'o', label='True')
    recovered_tag_model = np.asarray([
        recover_tag[j](outputs[i][j].detach().cpu().numpy())
        for j in range(len(tags[i]))
    ])
    axs[1].plot(10**(recovered_tag_model),
                '.',
                color='tab:red',
                label='Predicted')
    axs[1].set_yscale('log')
    axs[1].set_title('Screen Tags')
    axs[1].legend()
    axs[2].plot(tags[i].detach().cpu().numpy(), 'o', label='True')
    axs[2].plot(outputs[i].detach().cpu().numpy(),
                '.',
                color='tab:red',
                label='Predicted')
    axs[2].set_title('Unnormalized out')
    axs[2].set_ylim(0, 1)
    axs[2].legend()
    plt.savefig(f'{data_dir}/{model_name}_score/{counter}.png')
    plt.close()


def altitude_profile_plot(
    model_name: str,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    tags: list,
    i: int,
    counter: int,
    data_dir: str,
    recover_tag: list[Callable],
) -> None:
    """Plots side by side the input image and the predicted/exact altitude
    profile.

    Parameters
    ----------
    model_name : str
        The name of the model
    inputs : torch.Tensor
        The input speckle patterns
    outputs : torch.Tensor
        The predicted screen tags
    tags : list
        The exact tags of the data
    i : int
        The batch index of the image
    counter : int
        The global index of the image
    data_dir : str
        The directory where the data is stored
    recover_tag : list
        List of functions to recover each tag
    """

    # Plot the image and output side by side
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    axs[0].imshow(inputs[i].detach().cpu().squeeze(), cmap='bone')
    recovered_tag_true = np.asarray([
        recover_tag[j](tags[i][j].detach().cpu().numpy())
        for j in range(len(tags[i]))
    ])
    h_true = 10**(recovered_tag_true)
    recovered_tag_model = np.asarray([
        recover_tag[j](outputs[i][j].detach().cpu().numpy())
        for j in range(len(tags[i]))
    ])
    h_predic = 10**(recovered_tag_model)

    axs[1].plot(h_true, range(len(h_true)), 'o--', label='True')

    plot_model_prediciton = False
    if plot_model_prediciton:
        axs[1].plot(h_predic,
                    range(len(h_predic)),
                    '.',
                    color='tab:red',
                    label='Predicted')

    axs[1].set_xscale('log')
    axs[1].set_xlabel(r'$C_{n}^{2}$')
    axs[1].set_ylabel('Altitude [m]')
    y_tick_labels = [f'{i*400}m' for i in range(0, 8)]
    axs[1].yaxis.set_ticks(range(0, 8))
    axs[1].set_yticklabels(y_tick_labels)

    plt.suptitle(r'$C_{n}^{2}(h)$ profile from Speckle Pattern')
    plt.tight_layout()
    plt.savefig(f'{data_dir}/{model_name}_score/h_{counter}.png')
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
