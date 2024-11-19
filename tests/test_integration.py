from __future__ import annotations

import glob
import os
import random
import runpy
import shutil

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
    for root, dirs, files in os.walk(basefolder):
        for img in files:
            if 'time' not in img and 'sum' not in img:
                expected = os.path.join(root, img)
                relative_path = os.path.relpath(root, basefolder)
                test_folder = os.path.join('tests/test_data/speckles',
                                           relative_path)
                test_img = os.path.join(test_folder, img)
                image_pairs.append((expected, test_img))
    return image_pairs


# Helper function to check if required files exist
def check_required_files_exist(pattern):
    if not glob.glob(pattern):
        print(
            f"Required '{pattern}' missing. "
            'If this is the first time you run the test locally, use `python scripts/setup_test.py`'
        )
        pytest.fail(
            f"Required '{pattern}' missing. "
            'If this is the first time you run the test locally, use `python scripts/setup_test.py`'
        )


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


@pytest.mark.dependency(name='test_data_present',
                        depends=['test_train_and_score'])
def test_data_present():
    check_required_files_exist(
        'tests/test_data/speckles/expected_results/*resnet*')
    check_required_files_exist(
        'tests/test_data/speckles/expected_model_weights/*resnet*')
    check_required_files_exist(
        'tests/test_data/speckles/expected_results/*scnn*')
    check_required_files_exist(
        'tests/test_data/speckles/expected_model_weights/*scnn*')


@pytest.mark.parametrize(('conf', 'model_type'), CONF_YAML)
@pytest.mark.dependency(depends=['test_train_and_score', 'test_data_present'])
@pytest.mark.parametrize(
    ('expected', 'test_img'),
    generate_image_paths('tests/test_data/speckles/expected_results/'))
def test_figures(conf, model_type, expected, test_img, image_diff):
    print(f'test img {test_img}', flush=True)
    print(f'expected img {expected}', flush=True)
    image_diff(expected, test_img)


@pytest.mark.parametrize(('conf', 'model_type'), CONF_YAML)
@pytest.mark.dependency(depends=['test_train_and_score', 'test_data_present'])
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
