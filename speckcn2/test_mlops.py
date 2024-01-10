import unittest
import torch
from speckcn2.mlops import test


class TestMLOps(unittest.TestCase):

    def test_test_function(self):
        # Create a dummy model
        model = torch.nn.Sequential(torch.nn.Flatten(),
                                    torch.nn.Linear(32 * 32, 1024),
                                    torch.nn.ReLU(), torch.nn.Linear(1024, 8))
        # Create a dummy test loader
        dataset = [(torch.rand(1, 1, 32, 32), torch.rand(1, 8))
                   for _ in range(32)]
        test_loader = torch.utils.data.DataLoader(dataset)
        device = torch.device('cpu')
        criterion = torch.nn.MSELoss()

        def recover_tag(x):
            return x

        # Call the function
        test_tags = test(model,
                         test_loader,
                         device,
                         criterion,
                         recover_tag,
                         nimg_plot=1)

        # Assert the expected output
        self.assertEqual(len(test_tags), 32)
        self.assertEqual(test_tags[0].shape[1], 8)


if __name__ == '__main__':
    unittest.main()
