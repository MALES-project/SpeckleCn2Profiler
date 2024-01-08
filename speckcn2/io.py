import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np


def prepare_data(datadirectory, transform, nimg_print=5, nreps=1):
    """If not already available, preprocesses the data by loading images and
    tags from the given directory, applying a transformation to the images.

    Parameters
    ----------
    datadirectory: str
        Path to the directory containing the data
    transform: torchvision.transforms
        Transformation to apply to the images
    nimg_print: int
        Number of images to print

    Returns
    -------
    all_images : list
        List of all images
    all_tags : list
        List of all tags
    """

    # First, check if the data has already been preprocessed
    if os.path.exists(os.path.join(datadirectory, 'all_images.pt')):
        print('*** Loading preprocessed data')
        # If so, load it
        all_images = torch.load(os.path.join(datadirectory, 'all_images.pt'))
        all_tags = torch.load(os.path.join(datadirectory, 'all_tags.pt'))
        return all_images, all_tags

    # Otherwise, preprocess the raw data:
    file_list = [
        file_name for file_name in os.listdir(datadirectory)
        if 'MALES' in file_name and 'tag' not in file_name
    ]
    # and randomly shuffle them
    np.random.shuffle(file_list)
    # I can also do data augmentation, by using each file multiple times (only if I am also doing a random rotation)
    file_list = file_list * nreps
    all_images = []
    all_tags = []
    if nimg_print > 0:
        show_image = True
    else:
        show_image = False

    # Load each text file as an image
    for counter, file_name in enumerate(file_list):
        # Construct the full path to the file
        file_path = os.path.join(datadirectory, file_name)

        # Open the text file as an image using PIL
        with open(file_path, 'r') as text_file:
            lines = text_file.readlines()

            pixel_values = np.array([[
                0 if value == 'NaN' or value == 'NaN\n' else float(value)
                for value in line.split(',')
            ] for line in lines],
                                    dtype=np.float32)

            # Create the image
            image = Image.fromarray(pixel_values, mode='F')

            # Apply the transformation
            image = transform(image)

            # and add it to the collection
            all_images.append(image)

            # Then load the screen tags
            tagname = file_name.replace('.txt', '_tag.txt')
            tags = np.loadtxt(os.path.join(datadirectory, tagname),
                              delimiter=',',
                              dtype=np.float32)

            # Plot the image using maplotlib
            if counter > nimg_print:
                show_image = False
            if show_image:
                fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                axs[0].imshow(image.squeeze(), cmap='bone')
                axs[0].set_title(f'Training Image {file_name}')

                # Plot the tags on the same plot
                axs[1].plot(tags, 'o')
                axs[1].set_yscale('log')
                axs[1].set_title('Screen Tags')
                axs[1].legend()

                fig.subplots_adjust(wspace=0.3)
                plt.show()
                plt.close()

            # Preprocess the tags
            tags = np.log10(tags)

            # Add the tag to the colleciton
            all_tags.append(tags)

    # Finally, store them before returning
    torch.save(all_images, os.path.join(datadirectory, 'all_images.pt'))
    torch.save(all_tags, os.path.join(datadirectory, 'all_tags.pt'))

    return all_images, all_tags


def normalize_tags(all_images, all_tags):
    """Normalize the tags to be between 0 and 1.

    Parameters
    ----------
    all_images : list
        List of all images
    all_tags : list
        List of all tags

    Returns:
    dataset : list
        List of tuples (image, normalized_tag)
    normalize_tag : function
        Function to normalize a tag
    recover_tag : function
        Function to recover a tag
    """

    # I process the arrays of tags, such that each tag is a number between 0 and 1
    tags = [tag for tag in all_tags]
    tags = np.array(tags)

    # I need to find the minimum and maximum value of each tag
    # I will use this to normalize the tags
    min_tags = np.min(tags, axis=0)
    max_tags = np.max(tags, axis=0)
    range_tags = max_tags - min_tags

    def normalize_tag(tag):
        return (tag - min_tags) / range_tags

    def recover_tag(ntag):
        return ntag * range_tags + min_tags

    # And normalize the tags
    normalized_tags = np.array([normalize_tag(tag) for tag in tags])

    # I can now create the dataset
    dataset = [(image, tag) for image, tag in zip(all_images, normalized_tags)]

    return dataset, normalize_tag, recover_tag


def train_test_split(dataset, batch_size=32, train_test_split=0.8):
    """Splits the data into training and testing sets.

    Parameters
    ----------
    dataset : list
        List of tuples (image, tag)
    batch_size : int
        Batch size for the data loaders
    train_test_split: float
        Fraction of the data to use for training

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Training data loader
    test_loader : torch.utils.data.DataLoader
        Testing data loader
    """

    train_size = int(train_test_split * len(dataset))
    print(
        f'*** There are {len(dataset)} images in the dataset, {train_size} for training and {len(dataset)-train_size} for testing.'
    )
    train_loader = torch.utils.data.DataLoader(dataset[:train_size],
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset[train_size:],
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=2)

    return train_loader, test_loader
