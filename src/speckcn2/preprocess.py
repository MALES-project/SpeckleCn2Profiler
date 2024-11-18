"""This module contains functions for training and evaluating a neural network
model using PyTorch. It includes the following key components:

1. `train`: Trains the model for a specified number of epochs, logs training
   and validation losses, and saves the model state at specified intervals.
2. `score`: Evaluates the model on a test dataset, calculates various metrics,
   and generates plots for a specified number of test samples.

The module relies on several external utilities and models from the `speckcn2`
package, including `EnsembleModel`, `ComposableLoss`, and `Normalizer`.
"""

from __future__ import annotations

import os
import pickle
import random
import time

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from speckcn2.normalizer import Normalizer
from speckcn2.transformations import PolarCoordinateTransform, ShiftRowsTransform, ToUnboundTensor
from speckcn2.utils import ensure_directory, plot_preprocessed_image


def assemble_transform(conf: dict) -> transforms.Compose:
    """Assembles the transformation to apply to each image.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration

    Returns
    -------
    transform : torchvision.transforms.Compose
        Transformation to apply to the images
    """
    list_transforms = []

    if conf['preproc']['centercrop'] > 0:
        # Take only the center of the image
        list_transforms.append(
            transforms.CenterCrop(conf['preproc']['centercrop']))

    if conf['preproc']['polarize']:
        # make the image larger for better polar conversion
        list_transforms.append(transforms.Resize(conf['preproc']['polresize']))
        # Convert to polar coordinates
        list_transforms.append(PolarCoordinateTransform())

    if conf['preproc']['randomrotate'] and not conf['preproc']['polarize']:
        # Randomly rotate the image, since it is symmetric (not if it is polarized)
        list_transforms.append(transforms.RandomRotation(degrees=(-180, 180)))

    if conf['preproc']['resize']:
        # Optionally, downscale it
        list_transforms.append(
            transforms.Resize(
                (conf['preproc']['resize'], conf['preproc']['resize'])))

    if conf['preproc']['equivariant'] and conf['preproc']['polarize']:
        # Apply the equivariant transform, which makes sense only in polar coordinates
        list_transforms.append(ShiftRowsTransform())

    list_transforms.append(ToUnboundTensor())

    return transforms.Compose(list_transforms)


def prepare_data(
    conf: dict,
    nimg_print: int = 5,
) -> tuple[list, list, list]:
    """If not already available, preprocesses the data by loading images and
    tags from the given directory, applying a transformation to the images.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    nimg_print: int
        Number of images to print

    Returns
    -------
    all_images : list
        List of all images
    all_tags : list
        List of all tags
    all_ensemble_ids : list
        List of all ensemble ids, representing images from the same Cn2 profile
    """
    datadirectory = conf['speckle']['datadirectory']
    dataname = conf['preproc']['dataname']
    tagname = dataname.replace('images', 'tags')
    ensemblename = dataname.replace('images', 'ensemble')

    # First, check if the data has already been preprocessed
    if os.path.exists(os.path.join(datadirectory, dataname)):
        print(f'*** Loading preprocessed data from {dataname}')
        # If so, load it
        all_images = torch.load(os.path.join(datadirectory, dataname))
        all_tags = torch.load(os.path.join(datadirectory, tagname))
        all_ensemble_ids = torch.load(os.path.join(datadirectory,
                                                   ensemblename))
        # print the info about the dataset
        print(f'*** There are {len(all_images)} images in the dataset.')
    else:
        # Check if there is at least one image file in the directory
        if not any('.h5' in file_name
                   for file_name in os.listdir(datadirectory)):
            raise FileNotFoundError(
                'No image files found in the directory. Please provide the '
                'correct path to the data directory.')
        # Otherwise, preprocess the raw data separating the single images
        all_images, all_tags, all_ensemble_ids = imgs_as_single_datapoint(
            conf, nimg_print)

    # Get the average value of the pixels, excluding the 0 values
    non_zero_pixels = 0
    sum_pixels = 0
    for image in all_images:
        non_zero_pixels_in_image = image[image != 0]
        non_zero_pixels += non_zero_pixels_in_image.numel()
        sum_pixels += torch.sum(non_zero_pixels_in_image)
    pixel_average = sum_pixels / non_zero_pixels
    print('*** Pixel average:', pixel_average)
    # and store it in the config
    conf['preproc']['pixel_average'] = pixel_average

    return all_images, all_tags, all_ensemble_ids


def imgs_as_single_datapoint(
    conf: dict,
    nimg_print: int = 5,
) -> tuple[list, list, list]:
    """Preprocesses the data by loading images and tags from the given
    directory, applying a transformation to the images. Each image is treated
    as a single data point.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    nimg_print: int
        Number of images to print

    Returns
    -------
    all_images : list
        List of all images
    all_tags : list
        List of all tags
    all_ensemble_ids : list
        List of all ensemble ids, representing images from the same Cn2 profile
    """
    time_start = time.time()
    datadirectory = conf['speckle']['datadirectory']
    mname = conf['model']['name']
    dataname = conf['preproc']['dataname']
    tagname = dataname.replace('images', 'tags')
    ensemblename = dataname.replace('images', 'ensemble')
    nreps = conf['preproc']['speckreps']

    # Dummy transformation to get the original image
    transform_orig = transforms.Compose([
        transforms.ToTensor(),
    ])

    # and get the transformation to apply to each image
    transform = assemble_transform(conf)

    # Get the list of images
    file_list = [
        file_name for file_name in os.listdir(datadirectory)
        if '.h5' in file_name and 'tag' not in file_name
    ]
    np.random.shuffle(file_list)
    # Optionally, do data augmentation by using each file multiple times
    # (It makes sense only in combination with random rotations)
    file_list = file_list * nreps

    all_images, all_tags, all_ensemble_ids = [], [], []
    show_image = nimg_print > 0
    if show_image:
        ensure_directory(f'{datadirectory}/imgs_to_{mname}')

    tag_files = get_tag_files(file_list, datadirectory)
    ensemble_dict = get_ensemble_dict(tag_files)

    # Load each text file as an image
    for counter, file_name in enumerate(file_list):
        # Process only if tags available
        if file_name in tag_files:

            # Construct the full path to the file
            file_path = os.path.join(datadirectory, file_name)

            # Open the HDF5 file
            print(file_path, flush=True)
            with h5py.File(file_path, 'r') as f:
                # Load the data from the 'data' dataset
                pixel_values = np.float32(f['data'][:])
                # and replace nans with 0
                np.nan_to_num(pixel_values, copy=False)

            # Create the image
            image_orig = Image.fromarray(pixel_values, mode='F')

            # Apply the transformation
            image = transform(image_orig)
            if show_image:
                image_orig = transform_orig(image_orig)

            # and add the img to the collection
            all_images.append(image)

            # Load the tags
            with h5py.File(tag_files[file_name], 'r') as f:
                if 'sample' in file_name:
                    tags = f['J'][:].reshape(8, 1)
                else:
                    tags = f['data'][:]

            # Plot the image using maplotlib
            if counter > nimg_print:
                show_image = False
            if show_image:
                plot_preprocessed_image(image_orig, image, tags, counter,
                                        datadirectory, mname, file_name)

            # Preprocess the tags
            np.log10(tags, out=tags)
            # Add the tag to the collection
            all_tags.append(tags.squeeze())

            # Get the ensemble ID
            ensemble_id = ensemble_dict[file_name]
            # and add it to the collection
            all_ensemble_ids.append(ensemble_id)

    # Finally, store them before returning
    torch.save(all_images, os.path.join(datadirectory, dataname))
    torch.save(all_tags, os.path.join(datadirectory, tagname))
    torch.save(all_ensemble_ids, os.path.join(datadirectory, ensemblename))

    print('*** Preprocessing complete.', flush=True)
    print('It took',
          time.time() - time_start, 'seconds to preprocess the data.')

    return all_images, all_tags, all_ensemble_ids


def get_tag_files(file_list: list, datadirectory: str) -> dict:
    """Function to check the existence of tag files for each image file.

    Parameters
    ----------
    file_list : list
        List of image files
    datadirectory : str
        The directory containing the data

    Returns
    -------
    tag_files : dict
        Dictionary of image files and their corresponding tag files
    """
    tag_files = {}
    for file_name in file_list:
        if 'MALES' in file_name:
            ftagname = file_name.replace('.h5', '_tag.h5')
        elif 'sample' in file_name:
            ftagname = file_name.rpartition('-')[0] + '_tag.h5'
        else:
            ftagname = file_name.rpartition('_')[0] + '_tag.h5'

        tag_path = os.path.join(datadirectory, ftagname)
        if os.path.exists(tag_path):
            tag_files[file_name] = tag_path
        else:
            print(f'*** Warning: tag file {ftagname} not found.')
    return tag_files


def get_ensemble_dict(tag_files: dict) -> dict:
    """Function to associate each Cn2 profile to an ensemble ID for parallel
    processing.

    Parameters
    ----------
    tag_files : dict
        Dictionary of image files and their corresponding tag files

    Returns
    -------
    ensemble_dict : dict
        Dictionary of image files and their corresponding ensemble IDs
    """
    ensembles = {}
    ensemble_counter = 1
    for value in tag_files.values():
        # Check if the value is already assigned an ID
        if value not in ensembles:
            # Assign a new ID if it's a new value
            ensembles[value] = ensemble_counter
            ensemble_counter += 1
    return {key: ensembles[value] for key, value in tag_files.items()}


def train_test_split(
    all_images: list[torch.tensor],
    all_tags: list[np.ndarray],
    all_ensemble_ids: list[int],
    nz: Normalizer,
) -> tuple[list, list]:
    """Splits the data into training and testing sets.

    Parameters
    ----------
    all_images : list
        List of images
    all_tags : list
        List of tags
    all_ensemble_ids : list
        List of ensemble ids
    nz: Normalizer
        The normalizer object to preprocess the data

    Returns
    -------
    train_set : list
        Training dataset
    test_set : list
        Testing dataset
    """
    # Get the config dict
    config = nz.conf
    # extract the model parameters
    modelname = config['model']['name']
    datadirectory = config['speckle']['datadirectory']
    ttsplit = config['hyppar'].get('ttsplit', 0.8)
    ensemble_size = config['preproc'].get('ensemble', 1)
    average_size = config['preproc'].get('average', 0)

    # Check if the training and test set are already prepared
    train_file = f'{datadirectory}/train_set_{modelname}.pickle'
    test_file = f'{datadirectory}/test_set_{modelname}.pickle'

    if os.path.isfile(train_file) and os.path.isfile(test_file):
        print('Loading the training and testing set...', flush=True)
        nz._normalizing_functions(all_images, all_tags, all_ensemble_ids)
        train_data = pickle.load(open(train_file, 'rb'))
        test_data = pickle.load(open(test_file, 'rb'))
        print(f'*** There are {len(train_data)} images in the training set, ')
        print(f'*** and {len(test_data)} images in the testing set.')
        return train_data, test_data

    # If the data are not already prepared, first I normalize them using the Normalizer object
    print('Normalizing the images and tags...', flush=True)
    dataset = nz.normalize_imgs_and_tags(all_images, all_tags,
                                         all_ensemble_ids)

    if average_size > 1 and ensemble_size > 1:
        raise ValueError(
            'The average_size and ensemble_size cannot be set at the same time.'
        )
    elif average_size > 1:
        dataset = create_average_dataset(dataset, average_size)
        print_average_info(dataset, average_size, ttsplit)
    elif ensemble_size > 1:
        dataset = create_ensemble_dataset(dataset, ensemble_size)
        print_ensemble_info(dataset, ensemble_size, ttsplit)
    else:
        print_dataset_info(dataset, ttsplit)

    train_set, test_set = split_dataset(dataset, ttsplit)

    pickle.dump(train_set, open(train_file, 'wb'))
    pickle.dump(test_set, open(test_file, 'wb'))

    return train_set, test_set


def create_average_dataset(dataset: list, average_size: int) -> list:
    """Creates a dataset of averages from a dataset of single images. The
    averages are created by grouping together average_size images.

    Parameters
    ----------
    dataset : list
        List of single images
    average_size : int
        The number of images that will be averaged together

    Returns
    -------
    average_dataset : list
        List of averages
    """
    split_averages: dict = {}
    for item in dataset:
        key = item[-1]
        split_averages.setdefault(key, []).append(item)

    average_dataset: list = []
    for average in split_averages.values():
        # * In each average, take n_groups groups of average_size datapoints
        n_groups = len(average) // average_size
        if n_groups < 1:
            raise ValueError(f'Average size {average_size} is too large '
                             f'for groups with size {len(average)}')
        # Extract the averages randomly
        sample = random.sample(average, n_groups * average_size)
        # Split the sample into groups of average_size
        list_to_avg = [
            sample[i:i + average_size]
            for i in range(0, n_groups * average_size, average_size)
        ]
        # Average the groups
        averages = [
            tuple(
                sum(element) / average_size
                for i, element in enumerate(zip(*group)))
            for group in list_to_avg
        ]
        average_dataset.extend(averages)

    random.shuffle(average_dataset)
    return average_dataset


def print_average_info(dataset: list, average_size: int, ttsplit: int):
    """Prints the information about the average dataset.

    Parameters
    ----------
    dataset : list
        The average dataset
    average_size : int
        The number of images in each average
    ttsplit : int
        The train-test split
    """
    train_size = int(ttsplit * len(dataset))
    print(
        f'*** There are {len(dataset)} average groups in the dataset, '
        f'that I split in {train_size} for training and '
        f'{len(dataset) - train_size} for testing. Each average is composed by '
        f'{average_size} images. This corresponds to {train_size * average_size} '
        f'for training and {(len(dataset) - train_size) * average_size} for testing.'
    )


def create_ensemble_dataset(dataset: list, ensemble_size: int) -> list:
    """Creates a dataset of ensembles from a dataset of single images. The
    ensembles are created by grouping together ensemble_size images. These
    images will be used to train the model in parallel.

    Parameters
    ----------
    dataset : list
        List of single images
    ensemble_size : int
        The number of images that will be processed together as an ensemble

    Returns
    -------
    ensemble_dataset : list
        List of ensembles
    """
    split_ensembles: dict = {}
    for item in dataset:
        key = item[-1]
        split_ensembles.setdefault(key, []).append(item)

    ensemble_dataset: list = []
    for ensemble in split_ensembles.values():
        # * In each ensemble, take n_groups groups of ensemble_size datapoints
        n_groups = len(ensemble) // ensemble_size
        if n_groups < 1:
            raise ValueError(f'Ensemble size {ensemble_size} is too large '
                             f'for ensembles with size {len(ensemble)}')
        # Extract the ensembles randomly
        sample = random.sample(ensemble, n_groups * ensemble_size)
        # Split the sample into groups of ensemble_size
        ensemble_dataset.extend(sample[i:i + ensemble_size]
                                for i in range(0, n_groups *
                                               ensemble_size, ensemble_size))

    random.shuffle(ensemble_dataset)
    return ensemble_dataset


def print_ensemble_info(dataset: list, ensemble_size: int, ttsplit: int):
    """Prints the information about the ensemble dataset.

    Parameters
    ----------
    dataset : list
        The ensemble dataset
    ensemble_size : int
        The number of images in each ensemble
    ttsplit : int
        The train-test split
    """

    train_size = int(ttsplit * len(dataset))
    print(
        f'*** There are {len(dataset)} ensemble groups in the dataset, '
        f'that I split in {train_size} for training and '
        f'{len(dataset) - train_size} for testing. Each ensemble is composed by '
        f'{ensemble_size} images. This corresponds to {train_size * ensemble_size} '
        f'for training and {(len(dataset) - train_size) * ensemble_size} for testing.'
    )


def print_dataset_info(dataset: list, ttsplit: int):
    """Prints the information about the dataset.

    Parameters
    ----------
    dataset : list
        The dataset
    ttsplit : int
        The train-test split
    """
    train_size = int(ttsplit * len(dataset))
    print(f'*** There are {len(dataset)} images in the dataset, '
          f'{train_size} for training and '
          f'{len(dataset) - train_size} for testing.')


def split_dataset(dataset: list, ttsplit: int) -> tuple[list, list]:
    """Splits the dataset into training and testing sets.

    Parameters
    ----------
    dataset : list
        The dataset
    ttsplit : int
        The train-test split

    Returns
    -------
    train_set : list
        The training set
    test_set : list
        The testing set
    """
    # First shuffle the dataset
    random.shuffle(dataset)
    train_size = int(ttsplit * len(dataset))
    return dataset[:train_size], dataset[train_size:]
