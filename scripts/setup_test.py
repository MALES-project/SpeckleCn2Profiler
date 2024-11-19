from __future__ import annotations

import os
import shutil
import sys

# Define arrays of source and destination paths
SOURCE_DIRS = [
    'tests/test_data/speckles/imgs_to_Model_test_resnet',
    'tests/test_data/speckles/Model_test_resnet_score',
    'tests/test_data/speckles/imgs_to_Model_test_scnn',
    'tests/test_data/speckles/Model_test_scnn_score',
    'tests/test_data/speckles/Model_test_resnet_states',
    'tests/test_data/speckles/Model_test_scnn_states'
]
DEST_DIRS = [
    'tests/test_data/speckles/expected_results/.',
    'tests/test_data/speckles/expected_results/.',
    'tests/test_data/speckles/expected_results/.',
    'tests/test_data/speckles/expected_results/.',
    'tests/test_data/speckles/expected_model_weights/.',
    'tests/test_data/speckles/expected_model_weights/.'
]

# Check if the number of source and destination directories match
if len(SOURCE_DIRS) != len(DEST_DIRS):
    print(
        'Error: The number of source and destination directories do not match.'
    )
    sys.exit(1)

# Loop through each source and destination pair
for source_dir, dest_dir in zip(SOURCE_DIRS, DEST_DIRS):
    if not os.path.isdir(source_dir):
        print(
            f"[Warning]: Source directory '{source_dir}' does not exist. Skipping..."
        )
        continue

    # Create the destination directory if it does not exist
    if not os.path.isdir(dest_dir):
        print(f"Creating destination directory '{dest_dir}'...")
        os.makedirs(dest_dir)

    print(f"Moving '{source_dir}' to '{dest_dir}'...")
    shutil.move(source_dir, dest_dir)

    print(f"Setup complete: '{source_dir}' moved to '{dest_dir}'")
