import torch
import numpy as np
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
