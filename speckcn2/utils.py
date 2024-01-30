import torch


def save(model: torch.nn.Module, datadirectory: str) -> None:
    """Save the model state and the model itself.

    Parameters
    ----------
    model : torch.nn.Module
        The model to save
    epoch : int
        The epoch of the model
    datadirectory : str
        The directory where the data is stored
    """

    model_state = {
        'epoch': model.epoch,
        'loss': model.loss,
        'val_loss': model.val_loss,
        'time': model.time,
        'model_state_dict': model.state_dict(),
    }

    torch.save(
        model_state,
        f'{datadirectory}/{model.name}_states/{model.name}_{model.epoch[-1]}.pth'
    )


def load(model: torch.nn.Module, datadirectory: str, epoch: int) -> None:
    """Load the model state and the model itself.

    Parameters
    ----------
    model : torch.nn.Module
        The model to load
    datadirectory : str
        The directory where the data is stored
    epoch : int
        The epoch of the model
    """

    model_state = torch.load(
        f'{datadirectory}/{model.name}_states/{model.name}_{epoch}.pth')

    model.epoch = model_state['epoch']
    model.loss = model_state['loss']
    model.val_loss = model_state['val_loss']
    model.time = model_state['time']
    model.load_state_dict(model_state['model_state_dict'])

    assert model.epoch[
        -1] == epoch, 'The epoch of the model is not the same as the one loaded'
