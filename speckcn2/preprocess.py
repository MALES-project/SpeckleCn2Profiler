import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
from typing import Callable, List, Tuple
import torchvision.transforms as transforms
from speckcn2.utils import ensure_directory


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


class PolarCoordinateTransform(torch.nn.Module):
    """Transform a Cartesian image to polar coordinates."""

    def __init__(self):
        super(PolarCoordinateTransform, self).__init__()

    def forward(self, img):
        """ forward method of the transform
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """

        img = np.array(img)  # Convert PIL image to NumPy array

        # Assuming img is a grayscale image
        height, width = img.shape
        polar_image = np.zeros_like(img)

        # Center of the image
        center_x, center_y = width // 2, height // 2

        # Maximum possible value of r
        max_r = np.sqrt(center_x**2 + center_y**2)

        # Create a grid of (x, y) coordinates
        y, x = np.ogrid[:height, :width]

        # Shift the grid so that the center of the image is at (0, 0)
        x = x - center_x
        y = y - center_y

        # Convert Cartesian to Polar coordinates
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)

        # Rescale r to [0, height)
        r = np.round(r * (height - 1) / max_r).astype(int)

        # Rescale theta to [0, width)
        theta = np.round((theta + 2 * np.pi) % (2 * np.pi) * (width - 1) /
                         (2 * np.pi)).astype(int)

        # Use a 2D histogram to accumulate all values that map to the same polar coordinate
        histogram, _, _ = np.histogram2d(theta.flatten(),
                                         r.flatten(),
                                         bins=[height, width],
                                         range=[[0, height], [0, width]],
                                         weights=img.flatten())

        # Count how many Cartesian coordinates map to each polar coordinate
        counts, _, _ = np.histogram2d(theta.flatten(),
                                      r.flatten(),
                                      bins=[height, width],
                                      range=[[0, height], [0, width]])

        # Take the average of all values that map to the same polar coordinate
        polar_image = histogram / counts

        # Handle any divisions by zero
        polar_image[np.isnan(polar_image)] = 0

        # Crop the large r part that is not used
        for x in range(width - 1, -1, -1):
            # If the column contains at least one non-black pixel
            if np.any(polar_image[:, x] != 0):
                # Crop at this x position
                polar_image = polar_image[:, :x]
                break

        # reconvert to PIL image before returning
        return Image.fromarray(polar_image)


class ShiftRowsTransform(torch.nn.Module):
    """Shift the rows of an image such that the row with the smallest sum is at
    the bottom."""

    def __init__(self):
        super(ShiftRowsTransform, self).__init__()

    def forward(self, img):
        img = np.array(img)  # Convert PIL image to NumPy array

        # Find the row with the largest sum
        row_sums = np.sum(img, axis=1)
        max_sum_row_index = np.argmin(row_sums)

        # Shift all rows respecting periodicity
        shifted_img = np.roll(img, -max_sum_row_index, axis=0)

        # reconvert to PIL image before returning
        return Image.fromarray(shifted_img)


class ToUnboundTensor(torch.nn.Module):
    """Transform the image into a tensor, but do not normalize it like
    torchvision.ToTensor."""

    def __init__(self):
        super(ToUnboundTensor, self).__init__()

    def forward(self, img):
        img = np.array(img)
        # add a color chanel in dim 0
        return torch.from_numpy(img).unsqueeze(0)


def prepare_data(
    conf: dict,
    nimg_print: int = 5,
) -> Tuple[List, List]:
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
    """
    datadirectory = conf['speckle']['datadirectory']
    dataname = conf['preproc']['dataname']
    tagname = dataname.replace('images', 'tags')

    # First, check if the data has already been preprocessed
    if os.path.exists(os.path.join(datadirectory, dataname)):
        print(f'*** Loading preprocessed data from {dataname}')
        # If so, load it
        all_images = torch.load(os.path.join(datadirectory, dataname))
        all_tags = torch.load(os.path.join(datadirectory, tagname))
    elif conf['preproc']['multichannel'] > 1:
        print(
            f'Preprocessing data as multichannel={conf["preproc"]["multichannel"]}'
        )
        raise NotImplementedError(
            '*** Error in preprocessing: multichannel>1 not implemented yet.')
    elif conf['preproc']['multichannel'] == 1:
        # Otherwise, preprocess the raw data separating the single images
        all_images, all_tags = imgs_as_single_datapoint(conf, nimg_print)
    else:
        raise ValueError(
            '*** Error in preprocessing: multichannel must be either 1 or >1.')

    return all_images, all_tags


def imgs_as_single_datapoint(
    conf: dict,
    nimg_print: int = 5,
) -> Tuple[List, List]:
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
    """
    datadirectory = conf['speckle']['datadirectory']
    mname = conf['model']['name']
    dataname = conf['preproc']['dataname']
    tagname = dataname.replace('images', 'tags')
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
    if nimg_print > 0:
        show_image = True
        # and create a folder to store the images
        ensure_directory(f'{datadirectory}/imgs_to_{mname}')
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
            image_orig = Image.fromarray(pixel_values, mode='F')

            # Apply the transformation
            image = transform(image_orig)
            image_orig = transform_orig(image_orig)

            # and add it to the collection
            all_images.append(image)

            # Then load the screen tags
            if 'MALES' in file_name:
                ftagname = file_name.replace('.txt', '_tag.txt')
            else:
                base_name, _, _ = file_name.rpartition('_')
                ftagname = base_name + '_tag.txt'

            if os.path.exists(os.path.join(datadirectory, ftagname)):
                tags = np.loadtxt(os.path.join(datadirectory, ftagname),
                                  delimiter=',',
                                  dtype=np.float32)

                # Plot the image using maplotlib
                if counter > nimg_print:
                    show_image = False
                if show_image:
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    # Plot the original image
                    axs[0].imshow(image_orig.squeeze(), cmap='bone')
                    axs[0].set_title(f'Training Image {file_name}')
                    # Plot the preprocessd image
                    axs[1].imshow(image.squeeze(), cmap='bone')
                    axs[1].set_title('Processed as')
                    axs[1].set_xlabel(r'$r$')
                    axs[1].set_ylabel(r'$\theta$')

                    # Plot the tags
                    axs[2].plot(tags, 'o')
                    axs[2].set_yscale('log')
                    axs[2].set_title('Screen Tags')
                    axs[2].legend()

                    fig.subplots_adjust(wspace=0.3)
                    plt.savefig(
                        f'{datadirectory}/imgs_to_{mname}/{counter}.png')
                    plt.close()

                # Preprocess the tags
                tags = np.log10(tags)

                # Add the tag to the colleciton
                all_tags.append(tags)
            else:
                print(f'*** Warning: tag file {ftagname} not found.')

    # Finally, store them before returning
    torch.save(all_images, os.path.join(datadirectory, dataname))
    torch.save(all_tags, os.path.join(datadirectory, tagname))

    print('*** Preprocessing complete.', flush=True)

    return all_images, all_tags


def imgs_as_channels(
    conf: dict,
    nimg_print: int = 5,
) -> Tuple[List, List]:
    """Preprocesses the data by loading images and tags from the given
    directory, applying a transformation to the images. Then the images are
    treated as color channels and they get grouped based on multiplicity value.

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
    """
    datadirectory = conf['speckle']['datadirectory']
    mname = conf['model']['name']
    dataname = conf['preproc']['dataname']
    tagname = dataname.replace('images', 'tags')
    nreps = conf['preproc']['speckreps']
    multiplicity = conf['preproc']['multichannel']

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
    # count them based on the corresponding tag
    print(multiplicity)
    ...  # TODO
    # optionally, randomly expand the groups in order to form groups of size multiplicity
    ...  # TODO
    # group them based on multiplicity
    ...  # TODO

    # and randomly shuffle them
    np.random.shuffle(file_list)
    # I can also do data augmentation, by using each file multiple times (only if I am also doing a random rotation)
    file_list = file_list * nreps
    all_images = []
    all_tags = []
    if nimg_print > 0:
        show_image = True
        # and create a folder to store the images
        ensure_directory(f'{datadirectory}/imgs_to_{mname}')
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
            image_orig = Image.fromarray(pixel_values, mode='F')

            # Apply the transformation
            image = transform(image_orig)
            image_orig = transform_orig(image_orig)

            # and add it to the collection
            all_images.append(image)

            # Then load the screen tags
            if 'MALES' in file_name:
                ftagname = file_name.replace('.txt', '_tag.txt')
            else:
                base_name, _, _ = file_name.rpartition('_')
                ftagname = base_name + '_tag.txt'

            if os.path.exists(os.path.join(datadirectory, ftagname)):
                tags = np.loadtxt(os.path.join(datadirectory, ftagname),
                                  delimiter=',',
                                  dtype=np.float32)

                # Plot the image using maplotlib
                if counter > nimg_print:
                    show_image = False
                if show_image:
                    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                    # Plot the original image
                    axs[0].imshow(image_orig.squeeze(), cmap='bone')
                    axs[0].set_title(f'Training Image {file_name}')
                    # Plot the preprocessd image
                    axs[1].imshow(image.squeeze(), cmap='bone')
                    axs[1].set_title('Processed as')
                    axs[1].set_xlabel(r'$r$')
                    axs[1].set_ylabel(r'$\theta$')

                    # Plot the tags
                    axs[2].plot(tags, 'o')
                    axs[2].set_yscale('log')
                    axs[2].set_title('Screen Tags')
                    axs[2].legend()

                    fig.subplots_adjust(wspace=0.3)
                    plt.savefig(
                        f'{datadirectory}/imgs_to_{mname}/{counter}.png')
                    plt.close()

                # Preprocess the tags
                tags = np.log10(tags)

                # Add the tag to the colleciton
                all_tags.append(tags)
            else:
                print(f'*** Warning: tag file {ftagname} not found.')

    # Finally, store them before returning
    torch.save(all_images, os.path.join(datadirectory, dataname))
    torch.save(all_tags, os.path.join(datadirectory, tagname))

    print('*** Preprocessing complete.', flush=True)

    return all_images, all_tags


def normalize_imgs_and_tags(
    all_images: List[torch.tensor], all_tags: List[np.ndarray]
) -> Tuple[
        List[Tuple[torch.tensor, np.ndarray]],
        Callable[[np.ndarray], np.ndarray],
        Callable[[np.ndarray], np.ndarray],
        List[Callable[[torch.tensor], torch.tensor]],
        List[Callable[[torch.tensor], torch.tensor]],
]:
    """Normalize both the input images and the tags to be between 0 and 1.

    Parameters
    ----------
    all_images : list
        List of all images
    all_tags : list
        List of all tags

    Returns
    -------
    dataset : list
        List of tuples (image, normalized_tag)
    normalize_img : function
        Function to normalize an image
    recover_img : function
        Function to recover an image
    normalize_tag : list
        List of functions to normalize each tag
    recover_tag : list
        List of functions to recover each tag
    """
    # Find the maximum between the maximum and minimum values of the images
    max_img = max([torch.max(image) for image in all_images])
    print('*** Image max:', max_img)
    min_img = min([torch.min(image) for image in all_images])
    print('*** Image min:', min_img)
    range_img = max_img - min_img

    def normalize_img(img):
        return (img - min_img) / range_img

    def recover_img(nimg):
        return nimg * range_img + min_img

    # Normalize the images
    normalized_images = [normalize_img(image) for image in all_images]

    # Then I normalize the tags
    min_tags = np.min(all_tags, axis=0)
    print('*** Tag min:', min_tags)
    max_tags = np.max(all_tags, axis=0)
    print('*** Tag max:', max_tags)

    # I create a lambda function for each tag
    def create_normalize_functions(min_t, max_t):
        return [(lambda x, min_t=min_t[i], max_t=max_t[i]: (x - min_t) /
                 (max_t - min_t)) for i in range(len(min_t))]

    normalize_tag = create_normalize_functions(min_tags, max_tags)

    # And the recover functions
    def create_recover_functions(min_t, max_t):
        return [(lambda x, min_t=min_t[i], max_t=max_t[i]: x *
                 (max_t - min_t) + min_t) for i in range(len(min_t))]

    recover_tag = create_recover_functions(min_tags, max_tags)

    # And normalize the tags
    normalized_tags = np.array(
        [[normalize_tag[j](tag) for j, tag in enumerate(tags)]
         for tags in all_tags])
    #    normalized_tags = np.array(normalized_tags)

    # I can now create the dataset
    dataset = [(image, tag)
               for image, tag in zip(normalized_images, normalized_tags)]

    return dataset, normalize_img, recover_img, normalize_tag, recover_tag
