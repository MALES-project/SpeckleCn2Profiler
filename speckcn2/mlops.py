import torch
import matplotlib.pyplot as plt


def train(model, last_model_state, final_epoch, train_loader, device,
          optimizer, criterion):
    """Trains the model for the given number of epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train
    last_model_state : int
        The number of the last model state
    final_epoch : int
        The final epoch
    train_loader : torch.utils.data.DataLoader
        The training data loader
    device : torch.device
        The device to use
    optimizer : torch.optim
        The optimizer to use
    criterion : torch.nn
        The loss function to use
    """

    average_loss = 0.0
    for epoch in range(last_model_state, final_epoch):
        total_loss = 0.0
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

        # Print the average loss for every epoch
        print(
            f'Epoch {epoch+1}/{final_epoch}, Average Loss: {average_loss:.4f}')

    return model, average_loss


def test(model, test_loader, device, criterion, recover_tag, nimg_plot=20):
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
    recover_tag : function
        Function to recover a tag
    nimg_plot : int
        Number of images to plot

    Returns
    -------
    test_tags : list
        List of all the predicted tags of the test set
    """

    with torch.no_grad():
        test_tags = []
        for idx, (inputs, tags) in enumerate(test_loader):
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            tags = tags.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, tags)

            # Print the loss for every epoch
            print(f'Loss: {loss.item():.4f}')

            if idx < nimg_plot:
                # Plot the image and output side by side
                fig, axs = plt.subplots(1, 3, figsize=(9, 2.5))
                axs[0].imshow(inputs[0].detach().cpu().squeeze(), cmap='bone')
                axs[0].set_title('Test Image')
                axs[1].plot(10**(recover_tag(tags[0].detach().cpu().numpy())),
                            'o',
                            label='True')
                axs[1].plot(10**(recover_tag(
                    outputs[0].detach().cpu().numpy())),
                            '.',
                            color='tab:red',
                            label='Predicted')
                axs[1].set_yscale('log')
                axs[1].set_title('Screen Tags')
                axs[1].legend()
                axs[2].plot(tags[0].detach().cpu().numpy(), 'o', label='True')
                axs[2].plot(outputs[0].detach().cpu().numpy(),
                            '.',
                            color='tab:red',
                            label='Predicted')
                axs[2].set_title('Unnormalized out')
                axs[2].set_ylim(0, 1)
                axs[2].legend()
                plt.show()

            # and get all the tags for statistic analysis
            for tag in tags:
                test_tags.append(tag)

    return test_tags
