from __future__ import annotations

import glob
import os
import runpy
import shutil
from pathlib import Path

import pytest
import torch

import speckcn2 as sp2

CONF_YAML = (('tests/test_data/conf_resnet.yaml', 'resnet'),
             ('tests/test_data/conf_scnn.yaml', 'scnn'))


# Function to remove files or directories matching a pattern
def remove_files(pattern, is_dir=False):
    for path in glob.glob(pattern):
        if is_dir:
            shutil.rmtree(path)
        else:
            os.remove(path)


@pytest.mark.parametrize(('conf', 'model_type'), CONF_YAML)
@pytest.mark.dependency(name='test_train_and_score')
def test_train_and_score(conf, model_type):
    basefolder = 'tests/test_data/speckles/'
    patterns = {
        'model_folders': f'{basefolder}Model_test_{model_type}*',
        'datasets': f'{basefolder}all_*{model_type}*pt',
        'train_test_splits':
        f'{basefolder}t*_set_Model_test_{model_type}*pickle'
    }
    remove_files(patterns['model_folders'], is_dir=True)
    remove_files(patterns['datasets'])
    remove_files(patterns['train_test_splits'])

    train = runpy.run_path('examples/example_train.py')
    train['main'](conf)

    post = runpy.run_path('examples/example_post.py')
    post['main'](conf)


@pytest.mark.parametrize(('conf', 'model_type'), CONF_YAML)
@pytest.mark.dependency(depends=['test_train_and_score'])
def test_figures(conf, model_type, image_diff):
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


@pytest.mark.parametrize(('conf', 'model_type'), CONF_YAML)
@pytest.mark.dependency(depends=['test_train_and_score'])
def test_weights(conf, model_type):
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
