import io  # type: ignore
import torch
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from speckcn2.postprocess import tags_distribution


def test_tags_distribution():
    # Create a dummy dataset
    dataset = [(torch.rand(64 * 64), torch.rand(8)) for _ in range(32)]
    test_tags = torch.rand((8, 32))
    device = torch.device('cpu')

    conf = {
        'model': {
            'name': 'test_model',
        },
        'speckle': {
            'datadirectory': 'speckcn2/assets/test'
        }
    }

    # Temporarily redirect stdout to a string buffer
    with io.StringIO() as buf, redirect_stdout(buf):
        # Call the function
        tags_distribution(conf, dataset, test_tags, device, rescale=False)

        # Now we can check if the print statements in your function are as expected
        assert 'Data shape:' in buf.getvalue()
        assert 'Prediction shape:' in buf.getvalue()
        assert 'Train mean:' in buf.getvalue()
        assert 'Train std:' in buf.getvalue()
        assert 'Prediction mean:' in buf.getvalue()
        assert 'Prediction std:' in buf.getvalue()

    # Close the plot
    plt.close()
