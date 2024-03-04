import torch
import pickle
import random
from dataclasses import dataclass
from PIL import Image
import os
import numpy as np
from typing import Callable
import torchvision.transforms as transforms
from speckcn2.utils import ensure_directory, plot_preprocessed_image
from speckcn2.transformations import PolarCoordinateTransform, ShiftRowsTransform, ToUnboundTensor
from speckcn2.normalizer import Normalizer


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

    if conf['preproc']['centercrop']>0:
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
    else:
        # Otherwise, preprocess the raw data separating the single images
        all_images, all_tags, all_ensemble_ids = imgs_as_single_datapoint(
            conf, nimg_print)

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

    # Get the list of images:
    file_list = [
        file_name for file_name in os.listdir(datadirectory)
        if '.txt' in file_name and 'tag' not in file_name
    ]
    # and randomly shuffle them
    np.random.shuffle(file_list)
    # I can also do data augmentation, by using each file multiple times (only if I am also doing a random rotation)
    file_list = file_list * nreps
    all_images = []
    all_tags = []
    all_ensemble_ids = []
    if nimg_print > 0:
        show_image = True
        # and create a folder to store the images
        ensure_directory(f'{datadirectory}/imgs_to_{mname}')
    else:
        show_image = False

    # Check existence of tag files
    tag_files = {}
    for file_name in file_list:
        if 'MALES' in file_name:
            ftagname = file_name.replace('.txt', '_tag.txt')
        else:
            base_name, _, _ = file_name.rpartition('_')
            ftagname = base_name + '_tag.txt'

        tag_path = os.path.join(datadirectory, ftagname)
        if os.path.exists(tag_path):
            tag_files[file_name] = tag_path
        else:
            print(f'*** Warning: tag file {ftagname} not found.')

    # I associate each Cn2 profile to an ensemble ID for parallel processing
    ensembles = {}
    ensemble_counter = 1
    for value in tag_files.values():
        # Check if the value is already assigned an ID
        if value not in ensembles:
            # Assign a new ID if it's a new value
            ensembles[value] = ensemble_counter
            ensemble_counter += 1
    # Create a new dictionary to associate IDs with original values
    ensemble_dict = {key: ensembles[value] for key, value in tag_files.items()}

    # Load each text file as an image
    for counter, file_name in enumerate(file_list):
        # Process only if tags available
        if file_name in tag_files:

            # Construct the full path to the file
            file_path = os.path.join(datadirectory, file_name)

            # Open the text file as an image using PIL
            pixel_values = np.loadtxt(file_path,
                                      delimiter=',',
                                      dtype=np.float32)

            # Create the image
            image_orig = Image.fromarray(pixel_values, mode='F')

            # Apply the transformation
            image = transform(image_orig)
            image_orig = transform_orig(image_orig)
            # and add the img to the collection
            all_images.append(image)

            # Load the tags
            tags = np.loadtxt(tag_files[file_name],
                              delimiter=',',
                              dtype=np.float32)

            # Plot the image using maplotlib
            if counter > nimg_print:
                show_image = False
            if show_image:
                plot_preprocessed_image(image_orig, image, tags, counter,
                                        datadirectory, mname, file_name)

            # Preprocess the tags
            np.log10(tags, out=tags)
            # Add the tag to the colleciton
            all_tags.append(tags)

            # Get the ensemble ID
            ensemble_id = ensemble_dict[file_name]
            # and add it to the collection
            all_ensemble_ids.append(ensemble_id)
        else:
            print(f'*** Warning: tag file {ftagname} not found.')

    # Finally, store them before returning
    torch.save(all_images, os.path.join(datadirectory, dataname))
    torch.save(all_tags, os.path.join(datadirectory, tagname))
    torch.save(all_ensemble_ids, os.path.join(datadirectory, ensemblename))

    print('*** Preprocessing complete.', flush=True)

    return all_images, all_tags, all_ensemble_ids


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
    train_test_split = config['hyppar']['ttsplit']
    ensemble_size = config['preproc']['ensemble']

    # Check if the training and test set are already prepared
    if os.path.isfile(f'{datadirectory}/train_set_{modelname}.pickle') and os.path.isfile(
            f'{datadirectory}/test_set_{modelname}.pickle'):
        print('Loading the training and testing set...', flush=True)
        train_set = pickle.load(
            open(f'{datadirectory}/train_set_{modelname}.pickle', 'rb'))
        test_set = pickle.load(
            open(f'{datadirectory}/test_set_{modelname}.pickle', 'rb'))
        return train_set, test_set

    # If the data are not already prepared, first I normalize them using the Normalizer object
    print('Normalizing the images and tags...', flush=True)
    dataset = nz.normalize_imgs_and_tags(all_images, all_tags, all_ensemble_ids)

    # If I am using ensembles, each data point is a tuple of images
    if ensemble_size > 1:
        ensemble_dataset = []

        split_ensembles = {}  # type: dict
        for item in dataset:
            key = item[-1]
            if key not in split_ensembles:
                split_ensembles[key] = []
            split_ensembles[key].append(item)

        for ensemble in split_ensembles:
            # * In each ensemble, take n_groups groups of ensemble_size datapoints
            this_e_size = len(split_ensembles[ensemble])
            n_groups = this_e_size // ensemble_size
            if n_groups < 1:
                raise ValueError(
                    f'Ensemble size {ensemble_size} is too large for ensemble {ensemble} with size {this_e_size}'
                )
            # Extact the ensembles randomly
            sample = random.sample(split_ensembles[ensemble],
                                   n_groups * ensemble_size)
            # split the sample into groups of ensemble_size
            sample = [
                sample[i:i + ensemble_size]
                for i in range(0, n_groups * ensemble_size, ensemble_size)
            ]
            # and append it to the ensemble_dataset as separate elements
            ensemble_dataset.extend(sample)

        # shuffle dimension 0 of the ensemble_dataset
        random.shuffle(ensemble_dataset)

        train_size = int(train_test_split * len(ensemble_dataset))
        train_set = ensemble_dataset[:train_size]
        test_set = ensemble_dataset[train_size:]

        print(
            f'*** There are {len(ensemble_dataset)} ensemble groups in the dataset, that I split in {len(train_set)} for training and {len(test_set)} for testing. Each ensemble is composed by {ensemble_size} images. This corresponds to {len(train_set)*ensemble_size} for training and {len(test_set)*ensemble_size} for testing.'
        )
    else:
        train_size = int(train_test_split * len(dataset))
        print(
            f'*** There are {len(dataset)} images in the dataset, {train_size} for training and {len(dataset)-train_size} for testing.'
        )

        random.shuffle(dataset)
        train_set = dataset[:train_size]
        test_set = dataset[train_size:]

    # Save the training and testing set
    pickle.dump(train_set,
                open(f'{datadirectory}/train_set_{modelname}.pickle', 'wb'))
    pickle.dump(test_set,
                open(f'{datadirectory}/test_set_{modelname}.pickle', 'wb'))

    return train_set, test_set