from __future__ import annotations

import pytest


@pytest.fixture(scope='module')
def my_test_conf():

    conf = {
        'speckle': {
            'datadirectory': 'tests/test_data',
            'hArray': [0.1, 300, 650, 1e3, 2e3, 5e3, 10e3, 20e3, 40e3],
            'splits': [150, 475, 825, 1500, 3500, 7500, 15000, 30000],
            'lambda': 550,
            'z': 0,
            'L': 40e3
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
            'speckreps': 1,
            'multichannel': 1
        },
        'loss': {
            'MAE': 1
        }
    }
    return conf
