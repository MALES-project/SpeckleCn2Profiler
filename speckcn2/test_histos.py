import unittest
import torch
import matplotlib.pyplot as plt
from histos import tags_distribution


class TestTagsDistribution(unittest.TestCase):

    def test_tags_distribution(self):
        # Create a dummy dataset
        dataset = [(torch.tensor([1, 2, 3]), torch.tensor([0.1, 0.2, 0.3])),
                   (torch.tensor([4, 5, 6]), torch.tensor([0.4, 0.5, 0.6])),
                   (torch.tensor([7, 8, 9]), torch.tensor([0.7, 0.8, 0.9]))]
        test_tags = torch.tensor([[0.11, 0.22, 0.33], [0.44, 0.55, 0.66],
                                  [0.77, 0.88, 0.99]])
        device = torch.device('cpu')

        # Call the function
        tags_distribution(dataset, test_tags, device, rescale=False)

        # Assert the expected output
        # (You can modify these assertions based on your actual data)
        self.assertEqual(len(plt.gcf().get_axes()), 8)
        self.assertEqual(plt.gcf().get_axes()[0].get_title(), 'Tag 0')
        self.assertEqual(plt.gcf().get_axes()[1].get_title(), 'Tag 1')
        self.assertEqual(plt.gcf().get_axes()[2].get_title(), 'Tag 2')
        self.assertEqual(plt.gcf().get_axes()[3].get_title(), 'Tag 3')
        self.assertEqual(plt.gcf().get_axes()[4].get_title(), 'Tag 4')
        self.assertEqual(plt.gcf().get_axes()[5].get_title(), 'Tag 5')
        self.assertEqual(plt.gcf().get_axes()[6].get_title(), 'Tag 6')
        self.assertEqual(plt.gcf().get_axes()[7].get_title(), 'Tag 7')

        # Close the plot
        plt.close()


if __name__ == '__main__':
    unittest.main()
