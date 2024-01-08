import matplotlib.pyplot as plt
import numpy as np


def tags_distribution(dataset,
                      test_tags,
                      device,
                      rescale=False,
                      recover_tag=None):
    """Plots the distribution of the tags.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset used for training
    test_tags : torch.Tensor
        The predicted tags for the test dataset
    device : torch.device
        The device to use
    rescale : bool, optional
        Whether to rescale the tags using recover_tag() or leave them between 0 and 1
    """

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
        if rescale:
            axs[i // 4, i % 4].hist(recover_tag(predic_tags[:, i]),
                                    bins=20,
                                    color='tab:red',
                                    density=True,
                                    histtype='step',
                                    label='Model precitions')
            axs[i // 4, i % 4].hist(recover_tag(train_tags[:, i]),
                                    bins=20,
                                    color='tab:blue',
                                    density=True,
                                    histtype='step',
                                    label='Training data')
        else:
            axs[i // 4, i % 4].hist(predic_tags[:, i],
                                    bins=20,
                                    color='tab:red',
                                    density=True,
                                    histtype='step',
                                    label='Model precitions')
            axs[i // 4, i % 4].hist(train_tags[:, i],
                                    bins=20,
                                    color='tab:blue',
                                    density=True,
                                    histtype='step',
                                    label='Training data')
        axs[i // 4, i % 4].set_title(f'Tag {i}')
    axs[0, 1].legend()
    plt.show()
