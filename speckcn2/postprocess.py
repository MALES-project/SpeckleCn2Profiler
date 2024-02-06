import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Optional
from torch import Tensor
from torch.utils.data import Dataset
from torch import device as Device
from speckcn2.utils import ensure_directory


def tags_distribution(
        conf: dict,
        dataset: Dataset,
        test_tags: Tensor,
        device: Device,
        rescale: bool = False,
        recover_tag: Optional[list[Callable[[Tensor],
                                            Tensor]]] = None) -> None:
    """Plots the distribution of the tags.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    dataset : torch.utils.data.Dataset
        The dataset used for training
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

    ensure_directory(f'{data_directory}/result_plots')

    train_tags = np.array([tag for _, tag in dataset])
    predic_tags = np.array([n.cpu().numpy() for n in test_tags])
    print(f'Data shape: {train_tags.shape}')
    print(f'Prediction shape: {predic_tags.shape}')
    print(f'Train mean: {train_tags.mean()}')
    print(f'Train std: {train_tags.std()}')
    print(f'Prediction mean: {predic_tags.mean()}')
    print(f'Prediction std: {predic_tags.std()}')
    # plot the distribution of each tag element
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(8):
        if rescale and recover_tag is not None:
            axs[i // 4, i % 4].hist(recover_tag[i](predic_tags[:, i]),
                                    bins=20,
                                    color='tab:red',
                                    density=True,
                                    alpha=0.5,
                                    label='Model precitions')
            axs[i // 4, i % 4].hist(recover_tag[i](train_tags[:, i]),
                                    bins=20,
                                    color='tab:blue',
                                    density=True,
                                    alpha=0.5,
                                    label='Training data')
        else:
            axs[i // 4, i % 4].hist(predic_tags[:, i],
                                    bins=20,
                                    color='tab:red',
                                    density=True,
                                    alpha=0.5,
                                    label='Model precitions')
            axs[i // 4, i % 4].hist(train_tags[:, i],
                                    bins=20,
                                    color='tab:blue',
                                    density=True,
                                    alpha=0.5,
                                    label='Training data')
        axs[i // 4, i % 4].set_title(f'Tag {i}')
    axs[0, 1].legend()
    plt.tight_layout()
    plt.savefig(f'{data_directory}/result_plots/{model_name}_tags.png')
    plt.close()
