from __future__ import annotations

import glob
import os
import random
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import speckcn2 as sp2

CONF_YAML = (('tests/test_data/conf_resnet.yaml', 'resnet'),
             ('tests/test_data/conf_scnn.yaml', 'scnn'))
SEEDS = (42, 43, 44)
EXPECTED_VALUES_SEED = {
    42: {
        'random': 0.8554501933059546,
        'numpy': 0.44600577295795574,
        'torch': 0.02965700626373291,
    },
    43: {
        'random': 0.8160040884005317,
        'numpy': 0.38412578876288583,
        'torch': 0.15573692321777344,
    },
    44: {
        'random': 0.4159863274445267,
        'numpy': 0.7251349509702875,
        'torch': 0.06965392827987671,
    }
}


# test to check if numpy, random and pytorch generate the expected random numbers on this machine
@pytest.mark.parametrize('seed', SEEDS)
@pytest.mark.dependency(name='random_numbers')
def test_random_numbers(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for _ in range(1000):
        random_value = random.random()
        numpy_value = np.random.rand()
        torch_value = torch.rand(1).item()

    assert random_value == pytest.approx(
        EXPECTED_VALUES_SEED[seed]
        ['random']), f'random.random() mismatch for seed {seed}'
    assert numpy_value == pytest.approx(
        EXPECTED_VALUES_SEED[seed]
        ['numpy']), f'np.random.rand() mismatch for seed {seed}'
    assert torch_value == pytest.approx(
        EXPECTED_VALUES_SEED[seed]
        ['torch']), f'torch.rand(1).item() mismatch for seed {seed}'


# Function to remove files or directories matching a pattern
def remove_files(pattern, is_dir=False):
    for path in glob.glob(pattern):
        if is_dir:
            shutil.rmtree(path)
        else:
            os.remove(path)


# Helper function to generate image paths
def generate_image_paths(basefolder):
    image_pairs = []
    for single_folder in os.walk(basefolder):
        for img in single_folder[2]:
            if 'time' not in img and 'sum' not in img:
                expected = single_folder[0] + '/' + img
                test_folder = 'tests/test_data/speckles/' + Path(
                    single_folder[0]).parts[-1]
                test_img = test_folder + '/' + img
                image_pairs.append((expected, test_img))
    return image_pairs


@pytest.mark.parametrize(('conf', 'model_type'), CONF_YAML)
@pytest.mark.dependency(name='test_train_and_score',
                        depends=['random_numbers'])
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


@pytest.mark.skipif(
    (sys.version_info.major != 3 or sys.version_info.minor != 10),
    reason='Test only runs on Python 3.10')
@pytest.mark.parametrize(('conf', 'model_type'), CONF_YAML)
@pytest.mark.dependency(depends=['test_train_and_score'])
@pytest.mark.parametrize(
    ('expected', 'test_img'),
    generate_image_paths('tests/test_data/speckles/expected_results/'))
def test_figures(conf, model_type, expected, test_img, image_diff):
    print(f'test img {test_img}', flush=True)
    print(f'expected img {expected}', flush=True)
    image_diff(expected, test_img)


@pytest.mark.skipif(
    (sys.version_info.major != 3 or sys.version_info.minor != 10),
    reason='Test only runs on Python 3.10')
@pytest.mark.parametrize(('conf', 'model_type'), CONF_YAML)
@pytest.mark.dependency(depends=['test_train_and_score'])
def test_weights(conf, model_type):
    config = sp2.load_config(conf)
    datadirectory = config['speckle']['datadirectory']
    test_model, _ = sp2.setup_model(config)
    test_model, _ = sp2.load_model_state(test_model, datadirectory)

    basefolder = 'tests/test_data/speckles/expected_model_weights/'
    config = sp2.load_config(conf)
    config['speckle']['datadirectory'] = basefolder
    expected_model, _ = sp2.setup_model(config)
    expected_model, _ = sp2.load_model_state(expected_model, basefolder)
    assert len(list(test_model.parameters())) > 0
    assert len(list(expected_model.parameters())) == len(
        list(test_model.parameters()))
    for p1, p2 in zip(expected_model.parameters(), test_model.parameters()):
        assert torch.equal(p1, p2)
