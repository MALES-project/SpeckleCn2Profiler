import torch
from dataclasses import dataclass
from PIL import Image
import os
import numpy as np
from typing import Callable
import torchvision.transforms as transforms
from speckcn2.utils import ensure_directory, plot_preprocessed_image
from speckcn2.transformations import PolarCoordinateTransform, ShiftRowsTransform, ToUnboundTensor


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

    if conf['preproc']['centercrop']:
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

    if conf['preproc']['equivariant']:
        # Apply the equivariant transform
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
        transforms.CenterCrop(conf['preproc']['centercrop']),
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


@dataclass
class Normalizer:
    """Class to handle the normalization of images and tags.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    """
    conf: dict

    def normalize_imgs_and_tags(
        self,
        all_images: list[torch.tensor],
        all_tags: list[np.ndarray],
        all_ensemble_ids: list[int],
    ) -> list[tuple[torch.tensor, np.ndarray, int]]:
        """Normalize both the input images and the tags to be between 0 and 1.

        Parameters
        ----------
        all_images : list
            List of all images
        all_tags : list
            List of all tags
        conf : dict
            Dictionary containing the configuration

        Returns
        -------
        dataset : list
            List of tuples (image, normalized_tag)
        """
        # Get a mask for the NaN values
        self._mask_img = 1 - np.isnan(all_images[0])
        # Then replace all the nan values with 0
        all_images = [torch.nan_to_num(image) for image in all_images]

        # Define the normalization functions for the images
        if not hasattr(self, 'normalize_img'):
            # Find the maximum between the maximum and minimum values of the images
            max_img = max([torch.max(image) for image in all_images])
            print('*** Image max:', max_img)
            min_img = min([torch.min(image) for image in all_images])
            print('*** Image min:', min_img)
            range_img = max_img - min_img

            self.normalize_img, self.recover_img = self._img_normalize_functions(
                min_img, range_img)

        # Normalize the images
        normalized_images = [self.normalize_img(image) for image in all_images]

        # Define the normalization functions for the tags
        if not hasattr(self, 'normalize_tag'):
            self.min_tags = np.min(all_tags, axis=0)
            print('*** Tag min:', self.min_tags)
            self.max_tags = np.max(all_tags, axis=0)
            print('*** Tag max:', self.max_tags)

            # get the std deviation if using Z-score
            if self.conf['preproc']['normalization'] == 'zscore':
                self.mean_tags = np.mean(all_tags, axis=0)
                print('*** Tag mean:', self.mean_tags)
                self.std_tags = np.std(all_tags, axis=0)
                print('*** Tag std:', self.std_tags)
            elif self.conf['preproc']['normalization'] == 'unif':
                raise NotImplementedError(
                    '*** uniform normalization is not fully implemented yet. The sorting messes up the ensemble IDs.'
                )
                array_tags = np.array(all_tags)

                # Find the indices that sort the tags
                _sorting_indices = np.argsort(array_tags, axis=0)
                # And the corresponding indices that unsort them
                self._unsorting_indices = np.argsort(np.argsort(array_tags,
                                                                axis=0),
                                                     axis=0)
                self._sorted_tags = np.stack([
                    array_tags[_sorting_indices[:, i], i]
                    for i in range(array_tags.shape[1])
                ]).T
                self.Ndata = array_tags.shape[0]

            self.normalize_tag, self.recover_tag = self._tag_normalize_functions(
            )

        # And normalize the tags
        normalized_tags = np.array([[
            self.normalize_tag[j](tag, tag_id) for j, tag in enumerate(tags)
        ] for tag_id, tags in enumerate(all_tags)],
                                   dtype=np.float32)

        # I can now create the dataset
        dataset = [(image, tag, ensemble_id) for image, tag, ensemble_id in
                   zip(normalized_images, normalized_tags, all_ensemble_ids)]

        return dataset

    def _img_normalize_functions(
            self, min_img: np.ndarray,
            range_img: np.ndarray) -> tuple[Callable, Callable]:
        """Create the normalization and recovery functions for the images. The
        images are normalized between 0 and 1 using global values.

        Parameters
        ----------
        min_img : np.ndarray
            Minimum value for all the images
        range_img : np.ndarray
            Maximum value for all the images

        Returns
        -------
        normalize_fn : Callable
            Function to normalize an image
        recover_fn : Callable
            Function to recover an image
        """
        normalize_fn = (
            lambda x, min_img=min_img, range_img=range_img: self._mask_img *
            (x - min_img) / range_img)

        recover_fn = (
            lambda y, min_img=min_img, range_img=range_img: self._mask_img *
            (y * range_img + min_img))

        return normalize_fn, recover_fn

    def _tag_normalize_functions(
            self) -> tuple[list[Callable], list[Callable]]:
        """Create the normalization and recovery functions for the tags.

        Returns
        -------
        normalize_functions : list
            List of functions to normalize each tag
        recover_functions : list
            List of functions to recover each tag
        """

        if self.conf['preproc']['normalization'] == 'unif':

            normalize_functions = [
                (lambda x, x_id, i=i: self._unsorting_indices[x_id, i] / self.
                 Ndata) for i in range(self.conf['speckle']['nscreens'])
            ]
            recover_functions = [
                (lambda y, i=i: self._sorted_tags[round(y * self.Ndata), i])
                for i in range(self.conf['speckle']['nscreens'])
            ]
            return normalize_functions, recover_functions
        elif self.conf['preproc']['normalization'] == 'zscore':
            normalize_functions = [
                (lambda x, mean=self.mean_tags[i], std=self.std_tags[i]:
                 (x - mean) / std)
                for i in range(self.conf['speckle']['nscreens'])
            ]
            recover_functions = [
                (lambda y, mean=self.mean_tags[i], std=self.std_tags[i]: y *
                 std + mean) for i in range(self.conf['speckle']['nscreens'])
            ]
            return normalize_functions, recover_functions
        elif self.conf['preproc']['normalization'] == 'log':
            # The log normalization follows this formula:
            #   f(x) = (log(x - min + 1) - log(max - min + 1)) / (log(c - min + 1) - log(max - min + 1))
            # where c=1 sets the range of 0<=f(x)<=c=1
            normalize_functions = [
                (lambda x, x_id, min_t=self.min_tags[i], max_t=self.max_tags[
                    i]: np.log((x - min_t + 1) / (max_t - min_t + 1)) / np.log(
                        (2 - min_t) / (max_t - min_t + 1)))
                for i in range(self.conf['speckle']['nscreens'].min_tags)
            ]
            recover_functions = [
                (lambda y, y_id, min_t=self.min_tags[i], max_t=self.max_tags[
                    i]: np.exp(y) * np.log((2 - min_t) / (max_t - min_t + 1)) *
                 (max_t - min_t + 1) + min_t - 1)
                for i in range(self.conf['speckle']['nscreens'].min_tags)
            ]
            return normalize_functions, recover_functions
        elif self.conf['preproc']['normalization'] == 'lin':
            normalize_functions = [
                (lambda x, x_id, min_t=self.min_tags[i], max_t=self.max_tags[
                    i]: (x - min_t) / (max_t - min_t))
                for i in range(self.conf['speckle']['nscreens'])
            ]
            recover_functions = [
                (lambda y, y_id, min_t=self.min_tags[i], max_t=self.max_tags[
                    i]: y * (max_t - min_t) + min_t)
                for i in range(self.conf['speckle']['nscreens'])
            ]
            return normalize_functions, recover_functions
        else:
            raise ValueError(
                f"*** Error in normalization: normalization {self.conf['preproc']['normalization']} unknown."
            )
