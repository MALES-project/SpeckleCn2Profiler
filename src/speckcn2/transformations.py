"""This module defines several image transformation classes using PyTorch and
NumPy.

The `PolarCoordinateTransform` class converts a Cartesian image to polar
coordinates, which can be useful for certain types of image analysis.
The `ShiftRowsTransform` class shifts the rows of an image so that the row with the
smallest sum is positioned at the bottom, which can help in aligning images for further processing.
The `ToUnboundTensor` class converts an image to a tensor without normalizing it,
preserving the original pixel values.
Lastly, the `SpiderMask` class applies a circular mask to the image, simulating
the effect of a spider by setting pixels outside the mask to a background value,
which can be useful in certain experimental setups.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


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


class SpiderMask(torch.nn.Module):
    """Apply a circular mask to the image, representing the effect of the
    spider.

    The pixels outside the spider are set to -0.01, such that their
    value is lower than no light in the detector (0).
    """

    def __init__(self):
        super(SpiderMask, self).__init__()

    def forward(self, img):
        # Convert the image to a numpy array
        img = np.array(img)

        # Create a circular mask
        h, w = img.shape[:2]
        center = (int(w / 2), int(h / 2))
        radius = min(center)
        Y, X = np.ogrid[:h, :w]
        mask = (X - center[0])**2 + (Y - center[1])**2 > radius**2

        # Then the inner circle (called spider) is also removed
        # its default diameter, defined by the experimental setup, is 44% of the image width
        spider_radius = int(0.22 * w)
        spider_mask = (X - center[0])**2 + (Y -
                                            center[1])**2 < spider_radius**2

        # Apply the mask to the image
        img = img.astype(float)
        # If you are normalizing the images, you can set the mask to x<0 such that
        # it will automatically be normalized to a value of 0
        bkg_value = -0.01
        # but if you are not normalizing the images, you can set the mask to 0
        bkg_value = 0

        img[mask] = bkg_value
        img[spider_mask] = bkg_value

        return Image.fromarray(img)
