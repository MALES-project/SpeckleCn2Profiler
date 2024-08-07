"""This module defines the Normalizer class, which handles the normalization of
images and tags based on a given configuration.

The class includes methods to precompile normalization functions,
normalize images and tags, and define specific normalization strategies
such as Z-score and uniform normalization. The normalization process
involves replacing NaN values, creating masks, and scaling values to a
specified range. The module also provides functions to recover the
original values from the normalized data. The Normalizer class ensures
that both images and tags are consistently normalized according to the
specified configuration, facilitating further processing and analysis.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


class Normalizer:
    """Class to handle the normalization of images and tags.

    Parameters
    ----------
    conf : dict
        Dictionary containing the configuration
    """

    def __init__(self, conf: dict):
        self.conf = conf

    def _normalizing_functions(
        self,
        all_images: list[torch.tensor],
        all_tags: list[np.ndarray],
        all_ensemble_ids: list[int],
    ):
        """Precompiles the normalizing functions.

        Parameters
        ----------
        all_images : list
            List of all images
        all_tags : list
            List of all tags
        conf : dict
            Dictionary containing the configuration
        """
        # Define the normalization functions for the images
        if not hasattr(self, 'normalize_img'):
            # First we replace all NaN values by 0
            all_images = [torch.nan_to_num(image) for image in all_images]
            self._mask_img = 1 - np.isnan(all_images[0])
            # Then we create a mask corresponding to all the 0s of image 0
            self._mask_img = self._mask_img * (all_images[0] != 0)
            # as a sanity check, we check if the mask would the same using the last image
            assert np.array_equal(
                self._mask_img, self._mask_img *
                (all_images[-1] != 0)), 'Error: mask is not consistent'
            self._define_img_normalize_functions(all_images)

        # Define the normalization functions for the tags
        if not hasattr(self, 'normalize_tag'):
            self._define_tag_normalize_functions(all_tags)

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
        self._normalizing_functions(all_images, all_tags, all_ensemble_ids)

        # Normalize the images
        normalized_images = [self.normalize_img(image) for image in all_images]

        # And normalize the tags
        normalized_tags = np.array([[
            self.normalize_tag[j](tag, tag_id) for j, tag in enumerate(tags)
        ] for tag_id, tags in enumerate(all_tags)],
                                   dtype=np.float32)

        # I can now create the dataset
        dataset = [(image, tag, ensemble_id) for image, tag, ensemble_id in
                   zip(normalized_images, normalized_tags, all_ensemble_ids)]

        return dataset

    def _define_img_normalize_functions(self, all_images):
        """Define the normalization functions for the images."""
        if self.conf['preproc']['img_normalization']:
            # Find the maximum between the maximum and minimum values of the images
            max_img = max([torch.max(image) for image in all_images])
            print('*** Image max:', max_img)
            min_img = min([torch.min(image) for image in all_images])
            print('*** Image min:', min_img)
            range_img = max_img - min_img

            self.normalize_img, self.recover_img = self._img_normalize_functions(
                min_img, range_img)
        else:
            # The images do not need to be normalized
            print('-> [!] No img normalization.')
            self.normalize_img = lambda x: x
            self.recover_img = lambda x, y: x

    def _define_tag_normalize_functions(self, all_tags):
        """Define the normalization functions for the tags."""
        self.min_tags = np.min(all_tags, axis=0)
        print('*** Tag min:', self.min_tags)
        self.max_tags = np.max(all_tags, axis=0)
        print('*** Tag max:', self.max_tags)

        # get the std deviation if using Z-score
        if self.conf['preproc']['normalization'] == 'zscore':
            self._define_zscore_normalize_functions(all_tags)
        elif self.conf['preproc']['normalization'] == 'unif':
            self._define_unif_normalize_functions(all_tags)

        self.normalize_tag, self.recover_tag = self._tag_normalize_functions()

    def _define_zscore_normalize_functions(self, all_tags):
        """Define the normalization functions for the tags using Z-score."""
        self.mean_tags = np.mean(all_tags, axis=0)
        print('*** Tag mean:', self.mean_tags)
        self.std_tags = np.std(all_tags, axis=0)
        print('*** Tag std:', self.std_tags)

    def _define_unif_normalize_functions(self, all_tags):
        """Define the normalization functions for the tags using uniform
        normalization."""
        raise NotImplementedError(
            '*** uniform normalization is not fully implemented yet. The sorting messes up the ensemble IDs.'
        )
        array_tags = np.array(all_tags)

        # Find the indices that sort the tags
        _sorting_indices = np.argsort(array_tags, axis=0)
        # And the corresponding indices that unsort them
        self._unsorting_indices = np.argsort(np.argsort(array_tags, axis=0),
                                             axis=0)
        self._sorted_tags = np.stack([
            array_tags[_sorting_indices[:, i], i]
            for i in range(array_tags.shape[1])
        ]).T
        self.Ndata = array_tags.shape[0]

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

        def normalize_fn(x, min_img=min_img, range_img=range_img):
            return self._mask_img * (x - min_img) / range_img

        def recover_fn(y, y_id, min_img=min_img, range_img=range_img):
            return self._mask_img * (y * range_img + min_img)

        return normalize_fn, recover_fn

    def _tag_normalize_functions(
            self) -> tuple[list[Callable], list[Callable]]:
        """Create the normalization and recovery functions for the tags.
        Several alternatives are available.

        Returns
        -------
        normalize_functions : list
            List of functions to normalize each tag
        recover_functions : list
            List of functions to recover each tag
        """
        normalization = self.conf['preproc']['normalization']
        nscreens = self.conf['speckle']['nscreens']

        if normalization == 'unif':
            return self._unif_normalize_functions(nscreens)
        elif normalization == 'zscore':
            return self._zscore_normalize_functions(nscreens)
        elif normalization == 'log':
            return self._log_normalize_functions(nscreens)
        elif normalization == 'lin':
            return self._lin_normalize_functions(nscreens)
        else:
            raise ValueError(
                f'*** Error in normalization: normalization {normalization} unknown.'
            )

    def _unif_normalize_functions(self, nscreens):

        def create_normalize_functions(nscreens):

            def normalize_fn(x, x_id, i):
                return self._unsorting_indices[x_id, i] / self.Ndata

            return [normalize_fn for _ in range(nscreens)]

        def create_recover_functions(nscreens):

            def recover_fn(y, y_id, i):
                return self._sorted_tags[round(y * self.Ndata), i]

            return [recover_fn for _ in range(nscreens)]

        normalize_functions = create_normalize_functions(nscreens)
        recover_functions = create_recover_functions(nscreens)
        return normalize_functions, recover_functions

    def _zscore_normalize_functions(self, nscreens):
        # TODO if you still decide to support this,
        # return def functions and not lambdas
        normalize_functions = [
            (lambda x, x_id, mean=self.mean_tags[i], std=self.std_tags[i]:
             (x - mean) / std) for i in range(nscreens)
        ]
        recover_functions = [(lambda y, y_id, mean=self.mean_tags[
            i], std=self.std_tags[i]: y * std + mean) for i in range(nscreens)]
        return normalize_functions, recover_functions

    def _log_normalize_functions(self, nscreens):
        # TODO if you still decide to support this,
        # return def functions and not lambdas
        normalize_functions = [(lambda x, x_id, min_t=self.min_tags[
            i], max_t=self.max_tags[i]: np.log(
                (x - min_t + 1) / (max_t - min_t + 1)) / np.log(
                    (2 - min_t) / (max_t - min_t + 1)))
                               for i in range(nscreens)]
        recover_functions = [(lambda y, y_id, min_t=self.min_tags[
            i], max_t=self.max_tags[i]: np.exp(y) * np.log(
                (2 - min_t) / (max_t - min_t + 1)) *
                              (max_t - min_t + 1) + min_t - 1)
                             for i in range(nscreens)]
        return normalize_functions, recover_functions

    def _lin_normalize_functions(self, nscreens):
        # TODO if you still decide to support this,
        # return def functions and not lambdas
        normalize_functions = [
            (lambda x, x_id, min_t=self.min_tags[i], max_t=self.max_tags[i]:
             (x - min_t) / (max_t - min_t)) for i in range(nscreens)
        ]
        recover_functions = [(
            lambda y, y_id, min_t=self.min_tags[i], max_t=self.max_tags[i]: y *
            (max_t - min_t) + min_t) for i in range(nscreens)]
        return normalize_functions, recover_functions
