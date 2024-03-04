import torch
from dataclasses import dataclass
from PIL import Image
import os
import numpy as np
from typing import Callable


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
