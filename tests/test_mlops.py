import torch
from speckcn2.mlops import score


def test_score():
    # Create a dummy model
    dmodel = torch.nn.Sequential(torch.nn.Flatten(),
                                 torch.nn.Linear(32 * 32, 1024),
                                 torch.nn.ReLU(), torch.nn.Linear(1024, 8))
    dmodel.name = 'test_model'
    # Create a dummy test loader
    dataset = [(torch.rand(1, 1, 32, 32), torch.rand(1, 8)) for _ in range(32)]
    test_loader = torch.utils.data.DataLoader(dataset)
    device = torch.device('cpu')
    criterion = torch.nn.MSELoss()

    recover_tag = [(lambda x, min_t=0, max_t=1: x * (max_t - min_t) + min_t)
                   for i in range(8)]

    # Call the function
    test_tags = score(dmodel,
                      test_loader,
                      device,
                      criterion,
                      recover_tag,
                      data_dir='tests/test_data',
                      nimg_plot=1)

    # Assert the expected output
    assert len(test_tags) == 32
    assert test_tags[0].shape[0] == 8
