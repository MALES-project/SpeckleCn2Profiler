import time
import random
import torch
from torch import nn, optim, Tensor
from speckcn2.io import save
from speckcn2.loss import ComposableLoss
from speckcn2.mlmodels import EnsembleModel
from speckcn2.preprocess import Normalizer
from speckcn2.utils import ensure_directory
from speckcn2.plots import score_plot


def train(model: nn.Module, last_model_state: int, conf: dict, train_set: list,
          test_set: list, device: torch.device, optimizer: optim.Optimizer,
          criterion: ComposableLoss) -> tuple[nn.Module, float]:
    """Trains the model for the given number of epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    last_model_state : int
        The number of the last model state
    conf : dict
        Dictionary containing the configuration
    train_set : list
        The training set
    test_set : list
        The testing set
    device : torch.device
        The device to use
    optimizer : torch.optim
        The optimizer to use
    criterion : ComposableLoss
        The loss function to use

    Returns
    -------
    model : torch.nn.Module
        The trained model
    average_loss : float
        The average loss of the last epoch
    """

    final_epoch = conf['hyppar']['maxepochs']
    save_every = conf['model']['save_every']
    datadirectory = conf['speckle']['datadirectory']
    batch_size = conf['hyppar']['batch_size']

    # Setup the EnsembleModel wrapper
    ensemble = EnsembleModel(conf['preproc']['ensemble'], device,
                             conf['preproc']['ensemble_unif'])

    print(f'Training the model from epoch {last_model_state} to {final_epoch}')
    average_loss = 0.0
    model.train()
    for epoch in range(last_model_state, final_epoch):
        total_loss = 0.0
        model.train()
        t_in = time.time()
        for i in range(0, len(train_set), batch_size):
            batch = train_set[i:i + batch_size]

            # Zero out the optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs, targets, _ = ensemble(model, batch)
            loss, _ = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

        # Suffle the training set
        random.shuffle(train_set)

        # Calculate average loss for the epoch
        average_loss = total_loss / len(train_set)

        # Log the important information
        t_fin = time.time() - t_in
        model.loss.append(average_loss)
        model.time.append(t_fin)
        model.epoch.append(epoch + 1)

        # And also the validation loss
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i in range(0, len(test_set), batch_size):
                batch = test_set[i:i + batch_size]
                # Forward pass
                outputs, targets, _ = ensemble(model, batch)
                loss, _ = criterion(outputs, targets)
                # sum loss
                val_loss += loss.item()
        val_loss = val_loss / len(test_set)
        model.val_loss.append(val_loss)

        # Print the average loss for every epoch
        print(
            f'Epoch {epoch+1}/{final_epoch} (in {t_fin:.3g}s),\tTrain-Loss: {average_loss:.5f},\tTest-Loss: {val_loss:.5f}',
            flush=True)

        if (epoch + 1) % save_every == 0 or epoch == final_epoch - 1:
            # Save the model state
            save(model, datadirectory)

    return model, average_loss


def score(model: nn.Module,
          test_set: list,
          device: torch.device,
          criterion: ComposableLoss,
          normalizer: Normalizer,
          nimg_plot: int = 100) -> list[Tensor]:
    """Tests the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to test
    test_set : list
        The testing set
    device : torch.device
        The device to use
    criterion : ComposableLoss
        The composable loss function, where I can access useful parameters
    normalizer : Normalizer
        The normalizer used to recover the tags
    nimg_plot : int
        Number of images to plot

    Returns
    -------
    test_tags : list
        List of all the predicted tags of the test set
    """
    counter = 0
    conf = normalizer.conf
    data_dir = conf['speckle']['datadirectory']
    batch_size = conf['hyppar']['batch_size']
    # Setup the EnsembleModel wrapper
    ensemble = EnsembleModel(conf['preproc']['ensemble'], device,
                             conf['preproc']['ensemble_unif'])

    with torch.no_grad():
        # Put model in evaluation mode
        model.eval()

        test_tags = []
        # create the directory where the images will be stored
        ensure_directory(f'{data_dir}/{model.name}_score')

        for idx in range(0, len(test_set), batch_size):
            batch = test_set[idx:idx + batch_size]

            # Forward pass
            outputs, targets, inputs = ensemble(model, batch)

            # Loop each input separately
            for i in range(len(outputs)):
                loss, losses = criterion(outputs[i], targets[i])
                # Print the loss for every epoch
                print(f'Item {counter} loss: {loss.item():.4f}')

                if counter < nimg_plot:
                    score_plot(conf, inputs, outputs, targets, loss, losses, i,
                               counter, criterion)

                # and get all the tags for statistic analysis
                for tag in outputs:
                    test_tags.append(tag)

                counter += 1

    return test_tags
