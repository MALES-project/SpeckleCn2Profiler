import unittest
import shutil
import os
import torch
from speckcn2.mlmodels import load_model_state


class TestLoadModelState(unittest.TestCase):

    def setUp(self):
        self.model = torch.nn.Module()
        self.datadirectory = 'speckcn2/assets/test'

    def test_load_model_state_no_model_folder(self):
        # Remove the model folder if it exists
        model_folder = f'{self.datadirectory}/model_states'
        if os.path.isdir(model_folder):
            shutil.rmtree(model_folder)

        # Call the load_model_state function
        loaded_model, last_state = load_model_state(self.model,
                                                    self.datadirectory)

        # Check if the model folder is created
        self.assertTrue(os.path.isdir(model_folder))

        # Check if the loaded model is the same as the input model
        self.assertIs(loaded_model, self.model)

        # Check if the last model state is 0
        self.assertEqual(last_state, 0)

    def test_load_model_state_existing_model_states(self):
        # Create some dummy model state files
        model_folder = f'{self.datadirectory}/model_states'
        if not os.path.isdir(model_folder):
            os.mkdir(model_folder)
        torch.save(self.model.state_dict(), f'{model_folder}/model_1.pth')
        torch.save(self.model.state_dict(), f'{model_folder}/model_2.pth')

        # Call the load_model_state function
        loaded_model, last_state = load_model_state(self.model,
                                                    self.datadirectory)

        # Check if the loaded model is the same as the input model
        self.assertIs(loaded_model, self.model)

        # Check if the last model state is the highest numbered state file
        self.assertEqual(last_state, 2)

    def test_load_model_state_no_model_states(self):
        # Remove any existing model state files
        model_folder = f'{self.datadirectory}/model_states'
        if os.path.isdir(model_folder):
            for file_name in os.listdir(model_folder):
                os.remove(f'{model_folder}/{file_name}')

        # Call the load_model_state function
        loaded_model, last_state = load_model_state(self.model,
                                                    self.datadirectory)

        # Check if the loaded model is the same as the input model
        self.assertIs(loaded_model, self.model)

        # Check if the last model state is 0
        self.assertEqual(last_state, 0)


if __name__ == '__main__':
    unittest.main()
