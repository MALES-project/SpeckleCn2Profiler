import os
import torch
import numpy as np
from speckcn2.preprocess import prepare_data


def test_prepare_data(tmpdir):
    # Define the test data directory
    test_data_dir = 'speckcn2/assets/test'

    conf = {
        'speckle': {
            'datadirectory': test_data_dir
        },
        'model': {
            'name': 'test_model'
        },
        'preproc': {
            'polarize': False,
            'equivariant': False,
            'randomrotate': False,
            'centercrop': 100,
            'resize': 100,
            'dataname': 'all_images_test_model.pt',
        }
    }

    # Call the function
    all_images, all_tags = prepare_data(conf)

    # Assert the expected output
    assert isinstance(all_images, list)
    assert isinstance(all_tags, list)

    # Check if the preprocessed data files are saved
    assert os.path.exists(
        os.path.join(test_data_dir, 'all_images_test_model.pt'))
    assert os.path.exists(os.path.join(test_data_dir,
                                       'all_tags_test_model.pt'))

    # Load the preprocessed data files
    loaded_images = torch.load(
        os.path.join(test_data_dir, 'all_images_test_model.pt'))
    loaded_tags = torch.load(
        os.path.join(test_data_dir, 'all_tags_test_model.pt'))

    # Assert the loaded data matches the original data
    assert len(loaded_images) == len(all_images)
    assert len(loaded_tags) == len(all_tags)
    for i in range(len(all_images)):
        assert torch.all(torch.eq(loaded_images[i], all_images[i]))
        assert np.array_equal(loaded_tags[i], all_tags[i])

    # Clean up the preprocessed data files
    os.remove(os.path.join(test_data_dir, 'all_images_test_model.pt'))
    os.remove(os.path.join(test_data_dir, 'all_tags_test_model.pt'))
