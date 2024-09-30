from __future__ import annotations

import pytest
import runpy

CONF_YAML = ('tests/test_data/test_resnet.yaml', 'tests/test_data/test_scnn.yaml')
@pytest.mark.parametrize('conf', CONF_YAML)
def test_example(conf):
    train = runpy.run_path('examples/example_train.py')
    train['main'](conf)

    post = runpy.run_path('examples/example_post.py')
    post['main'](conf)