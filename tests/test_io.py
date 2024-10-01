from __future__ import annotations

import yaml

from speckcn2.io import load_config


def test_load_config():
    config_file_path = 'tests/test_data/conf_resnet.yaml'
    with open(config_file_path, 'r') as file:
        expected_config = yaml.safe_load(file)

    actual_config = load_config(config_file_path)
    assert actual_config == expected_config
