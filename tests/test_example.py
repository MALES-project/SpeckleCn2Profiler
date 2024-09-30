from __future__ import annotations

import os
import runpy
from pathlib import Path

import pytest

CONF_YAML = ('tests/test_data/conf_resnet.yaml',
             'tests/test_data/conf_scnn.yaml')


@pytest.mark.parametrize('conf', CONF_YAML)
@pytest.mark.dependency()
def test_example(conf):
    train = runpy.run_path('examples/example_train.py')
    train['main'](conf)

    post = runpy.run_path('examples/example_post.py')
    post['main'](conf)


@pytest.mark.parametrize('conf', CONF_YAML)
#@pytest.mark.dependency(depends=['test_example'])
def test_example_figures(conf, image_diff):
    basefolder = 'tests/test_data/speckles/expected_results/'
    os.walk(basefolder)
    for single_folder in os.walk(basefolder):
        for img in single_folder[2]:
            expected = single_folder[0] + '/' + img
            test_folder = 'tests/test_data/speckles/' + Path(
                single_folder[0]).parts[-1]
            test_img = test_folder + '/' + img
            print(f'test img {test_img}')
            print(f'expected img {expected}')
            image_diff(expected, test_img)
