import os
import torch
import pytest
from speckcn2.mlmodels import load_model_state


@pytest.fixture
def model_and_directory(tmpdir):
    model = torch.nn.Module()
    model.name = 'test_model'
    datadirectory = tmpdir.mkdir('test_data')
    return model, str(datadirectory)


def test_load_model_state_no_model_folder(model_and_directory):
    model, datadirectory = model_and_directory

    # Call the load_model_state function
    loaded_model, last_state = load_model_state(model, datadirectory)

    # Check if the model folder is created
    model_folder = os.path.join(datadirectory, 'test_model_states')
    assert os.path.isdir(model_folder)

    # Check if the loaded model is the same as the input model
    assert loaded_model is model

    # Check if the last model state is 0
    assert last_state == 0


def test_load_model_state_existing_model_states(model_and_directory):
    model, datadirectory = model_and_directory

    # Create some dummy model state files
    model_folder = os.path.join(datadirectory, 'test_model_states')
    os.mkdir(model_folder)
    torch.save(model.state_dict(),
               os.path.join(model_folder, 'test_model_1.pth'))
    torch.save(model.state_dict(),
               os.path.join(model_folder, 'test_model_2.pth'))

    # Call the load_model_state function
    loaded_model, last_state = load_model_state(model, datadirectory)

    # Check if the loaded model is the same as the input model
    assert loaded_model is model

    # Check if the last model state is the highest numbered state file
    assert last_state == 2


def test_load_model_state_no_model_states(model_and_directory):
    model, datadirectory = model_and_directory

    # Remove any existing model state files
    model_folder = os.path.join(datadirectory, 'test_model_states')
    os.mkdir(model_folder)

    # Call the load_model_state function
    loaded_model, last_state = load_model_state(model, datadirectory)

    # Check if the loaded model is the same as the input model
    assert loaded_model is model

    # Check if the last model state is 0
    assert last_state == 0
