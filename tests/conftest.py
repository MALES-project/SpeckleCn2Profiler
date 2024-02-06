import pytest


@pytest.fixture(scope='module')
def my_test_conf():

    conf = {
        'speckle': {
            'datadirectory': 'tests/test_data',
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
        }
    }
    return conf
