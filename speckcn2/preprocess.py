import torch
import h5py
import time
import pickle
import random
from PIL import Image
import os
import numpy as np
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

            ## Open the text file as an image using PIL
            #pixel_values = np.loadtxt(file_path,
            #                          delimiter=',',
            #                          dtype=np.float32)
            # Open the HDF5 file
            with h5py.File(file_path, 'r') as f:
                # Load the data from the 'data' dataset
                pixel_values = f['data'][:]

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
                tags = f['data'][:]

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

    # Finally, store them before returning
    torch.save(all_images, os.path.join(datadirectory, dataname))
    torch.save(all_tags, os.path.join(datadirectory, tagname))
    torch.save(all_ensemble_ids, os.path.join(datadirectory, ensemblename))

    print('*** Preprocessing complete.', flush=True)
    print('It took', time.time() - time_start, 'seconds to preprocess the data.')

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
        ftagname = file_name.replace(
            '.h5',
            '_tag.h5') if 'MALES' in file_name else file_name.rpartition(
                '_')[0] + '_tag.h5'
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
    ttsplit = config['hyppar']['ttsplit']
    ensemble_size = config['preproc']['ensemble']

    # Check if the training and test set are already prepared
    train_file = f'{datadirectory}/train_set_{modelname}.pickle'
    test_file = f'{datadirectory}/test_set_{modelname}.pickle'

    if os.path.isfile(train_file) and os.path.isfile(test_file):
        print('Loading the training and testing set...', flush=True)
        nz._normalizing_functions(all_images, all_tags, all_ensemble_ids)
        return pickle.load(open(train_file,
                                'rb')), pickle.load(open(test_file, 'rb'))

    # If the data are not already prepared, first I normalize them using the Normalizer object
    print('Normalizing the images and tags...', flush=True)
    dataset = nz.normalize_imgs_and_tags(all_images, all_tags,
                                         all_ensemble_ids)

    if ensemble_size > 1:
        dataset = create_ensemble_dataset(dataset, ensemble_size)
        print_ensemble_info(dataset, ensemble_size, ttsplit)
    else:
        print_dataset_info(dataset, ttsplit)

    train_set, test_set = split_dataset(dataset, ttsplit)

    pickle.dump(train_set, open(train_file, 'wb'))
    pickle.dump(test_set, open(test_file, 'wb'))

    return train_set, test_set


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
    split_ensembles = {}
    for item in dataset:
        key = item[-1]
        split_ensembles.setdefault(key, []).append(item)

    ensemble_dataset = []
    for ensemble in split_ensembles.values():
        # * In each ensemble, take n_groups groups of ensemble_size datapoints
        n_groups = len(ensemble) // ensemble_size
        if n_groups < 1:
            raise ValueError(
                f'Ensemble size {ensemble_size} is too large for ensemble {ensemble} with size {len(ensemble)}'
            )
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
        f'*** There are {len(dataset)} ensemble groups in the dataset, that I split in {train_size} for training and {len(dataset) - train_size} for testing. Each ensemble is composed by {ensemble_size} images. This corresponds to {train_size * ensemble_size} for training and {(len(dataset) - train_size) * ensemble_size} for testing.'
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
    print(
        f'*** There are {len(dataset)} images in the dataset, {train_size} for training and {len(dataset) - train_size} for testing.'
    )


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
