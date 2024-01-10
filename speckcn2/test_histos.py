import unittest
import io
import torch
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from speckcn2.histos import tags_distribution


class TestTagsDistribution(unittest.TestCase):

    def test_tags_distribution(self):
        # Create a dummy dataset
        dataset = [(torch.rand(64 * 64), torch.rand(8)) for _ in range(32)]
        test_tags = torch.rand((8, 32))
        device = torch.device('cpu')

        # Temporarily redirect stdout to a string buffer
        with io.StringIO() as buf, redirect_stdout(buf):
            # Call the function
            tags_distribution(dataset, test_tags, device, rescale=False)

            # Now we can check if the print statements in your function are as expected
            self.assertIn('Data shape:', buf.getvalue())
            self.assertIn('Prediction shape:', buf.getvalue())
            self.assertIn('Train mean:', buf.getvalue())
            self.assertIn('Train std:', buf.getvalue())
            self.assertIn('Prediction mean:', buf.getvalue())
            self.assertIn('Prediction std:', buf.getvalue())

        # Close the plot
        plt.close()


if __name__ == '__main__':
    unittest.main()
