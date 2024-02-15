import time
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from speckcn2.io import save
from speckcn2.preprocess import Normalizer
from speckcn2.utils import ensure_directory
from speckcn2.plots import score_plot, altitude_profile_plot


def train(model: nn.Module, last_model_state: int, conf: dict,
          train_loader: DataLoader, test_loader: DataLoader,
          device: torch.device, optimizer: optim.Optimizer,
          criterion: nn.Module) -> tuple[nn.Module, float]:
    """Trains the model for the given number of epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    last_model_state : int
        The number of the last model state
    conf : dict
        Dictionary containing the configuration
    train_loader : torch.utils.data.DataLoader
        The training data loader
    test_loader : torch.utils.data.DataLoader
        The testing data loader
    device : torch.device
        The device to use
    optimizer : torch.optim
        The optimizer to use
    criterion : torch.nn
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

    print(f'Training the model from epoch {last_model_state} to {final_epoch}')
    average_loss = 0.0
    model.train()
    for epoch in range(last_model_state, final_epoch):
        total_loss = 0.0
        t_in = time.time()
        for i, (inputs, tags) in enumerate(train_loader):
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            tags = tags.to(device)

            # Zero out the optimizer
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, tags)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

        # Calculate average loss for the epoch
        average_loss = total_loss / len(train_loader)

        # Log the important information
        t_fin = time.time() - t_in
        model.loss.append(average_loss)
        model.time.append(t_fin)
        model.epoch.append(epoch + 1)
        # And also the validation loss
        val_loss = 0.0
        with torch.no_grad():
            for idx, (inputs, tags) in enumerate(test_loader):
                # Move input and label tensors to the device
                inputs = inputs.to(device)
                tags = tags.to(device)
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, tags)
                # sum loss
                val_loss += loss.item()
        val_loss = val_loss / len(test_loader)
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
          test_loader: DataLoader,
          device: torch.device,
          criterion: nn.Module,
          normalizer: Normalizer,
          nimg_plot: int = 100) -> list[Tensor]:
    """Tests the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to test
    test_loader : torch.utils.data.DataLoader
        The testing data loader
    device : torch.device
        The device to use
    criterion : torch.nn
        The loss function to use
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
    data_dir = normalizer.conf['speckle']['datadirectory']

    with torch.no_grad():
        # Put model in evaluation mode
        model.eval()

        test_tags = []
        # create the directory where the images will be stored
        ensure_directory(f'{data_dir}/{model.name}_score')

        for idx, (inputs, tags) in enumerate(test_loader):
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            tags = tags.to(device)

            # Forward pass
            outputs = model(inputs)

            # Loop each input separately
            for i in range(len(outputs)):
                loss = criterion(outputs[i], tags[i])
                # Print the loss for every epoch
                print(f'Item {counter} loss: {loss.item():.4f}')

                if counter < nimg_plot:
                    score_plot(model.name, inputs, outputs, tags, loss, i,
                               counter, data_dir, normalizer.recover_tag)
                    altitude_profile_plot(model.name, inputs, outputs, tags, i,
                                          counter, data_dir,
                                          normalizer.recover_tag)

                # and get all the tags for statistic analysis
                for tag in outputs:
                    test_tags.append(tag)

                counter += 1

    return test_tags
