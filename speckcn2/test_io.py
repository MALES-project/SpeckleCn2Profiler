import unittest
import os
import torch
import torchvision.transforms as transforms
import numpy as np
from speckcn2.io import prepare_data


class TestPrepareData(unittest.TestCase):

    def test_prepare_data(self):
        # Define the test data directory
        test_data_dir = 'speckcn2/assets/test'

        # Define the transformation to apply to the images
        transform = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.ToTensor()])

        # Call the function
        all_images, all_tags = prepare_data(test_data_dir, transform)

        # Assert the expected output
        self.assertIsInstance(all_images, list)
        self.assertIsInstance(all_tags, list)

        # Check if the preprocessed data files are saved
        self.assertTrue(
            os.path.exists(os.path.join(test_data_dir, 'all_images.pt')))
        self.assertTrue(
            os.path.exists(os.path.join(test_data_dir, 'all_tags.pt')))

        # Load the preprocessed data files
        loaded_images = torch.load(os.path.join(test_data_dir,
                                                'all_images.pt'))
        loaded_tags = torch.load(os.path.join(test_data_dir, 'all_tags.pt'))

        # Assert the loaded data matches the original data
        self.assertEqual(len(loaded_images), len(all_images))
        self.assertEqual(len(loaded_tags), len(all_tags))
        for i in range(len(all_images)):
            self.assertTrue(
                torch.all(torch.eq(loaded_images[i], all_images[i])))
            self.assertTrue(np.array_equal(loaded_tags[i], all_tags[i]))

        # Clean up the preprocessed data files
        os.remove(os.path.join(test_data_dir, 'all_images.pt'))
        os.remove(os.path.join(test_data_dir, 'all_tags.pt'))


if __name__ == '__main__':
    unittest.main()
