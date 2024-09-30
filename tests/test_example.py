from __future__ import annotations

import os
import runpy
from pathlib import Path

import pytest
import torch

import speckcn2 as sp2

CONF_YAML = ('tests/test_data/conf_resnet.yaml',
             'tests/test_data/conf_scnn.yaml')


@pytest.mark.parametrize('conf', CONF_YAML)
@pytest.mark.dependency(name='test_example')
def test_example(conf):
    train = runpy.run_path('examples/example_train.py')
    train['main'](conf)

    post = runpy.run_path('examples/example_post.py')
    post['main'](conf)


@pytest.mark.parametrize('conf', CONF_YAML)
@pytest.mark.dependency(depends=['test_example'])
def test_example_figures(conf, image_diff):
    basefolder = 'tests/test_data/speckles/expected_results/'
    os.walk(basefolder)
    for single_folder in os.walk(basefolder):
        for img in single_folder[2]:
            if 'time' not in img and 'sum' not in img:
                expected = single_folder[0] + '/' + img
                test_folder = 'tests/test_data/speckles/' + Path(
                    single_folder[0]).parts[-1]
                test_img = test_folder + '/' + img
                print(f'test img {test_img}', flush=True)
                print(f'expected img {expected}', flush=True)
                image_diff(expected, test_img)


@pytest.mark.parametrize('conf', CONF_YAML)
@pytest.mark.dependency(depends=['test_example'])
def test_example_models(conf):
    config = sp2.load_config(conf)
    datadirectory = config['speckle']['datadirectory']
    test_model, _ = sp2.setup_model(config)
    test_model, _ = sp2.load_model_state(test_model, datadirectory)

    basefolder = 'tests/test_data/speckles/expected_results/'
    config = sp2.load_config(conf)
    expected_model, _ = sp2.setup_model(config)
    expected_model, _ = sp2.load_model_state(expected_model, basefolder)
    for p1, p2 in zip(expected_model.parameters(), test_model.parameters()):
        assert torch.equal(p1, p2)
