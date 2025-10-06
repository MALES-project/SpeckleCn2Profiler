# Quickstart Guide

This guide will help you get started with `speckcn2` quickly. Follow these steps to install the package, verify your installation, and run your first example.

## 1. Install SpeckleCn2Profiler

First, install the package using pip. We recommend using a virtual environment:

```bash
# Create and activate a virtual environment
python -m venv speckcn2-env
source speckcn2-env/bin/activate  # On Windows: speckcn2-env\Scripts\activate

# Install the package
pip install speckcn2
```

!!! tip "Need more details?"
    For platform-specific instructions and troubleshooting, see the full [Installation Guide](installation.md).

## 2. Verify Installation

Verify that the package is installed correctly:

```bash
python -c "import speckcn2; print('Installation successful!')"
```

## 3. Run Tests

To ensure everything is working properly, run the test suite:

```bash
# Install pytest if not already installed
pip install pytest

# Run tests
pytest
```

!!! note "First-time test setup"
    Some tests may fail on the first run because test data needs to be set up. If this happens, run:
    ```bash
    python ./scripts/setup_test.py
    pytest
    ```

## 4. Try a Minimal Example

Here's a minimal working example to train a model on sample data. Save this as `quickstart_train.py`:

```python
import speckcn2 as sp2
import torch

def main():
    # Load configuration (you'll need to provide your own config file)
    config = sp2.load_config('path/to/config.yaml')

    # Prepare data
    print("Preparing data...")
    all_images, all_tags, all_ensemble_ids = sp2.prepare_data(config)

    # Normalize tags (helps the model work with reasonable numbers)
    nz = sp2.Normalizer(config)

    # Split data into training and testing sets
    train_set, test_set = sp2.train_test_split(
        all_images, all_tags, all_ensemble_ids, nz
    )

    # Setup device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load and configure the model
    model, last_model_state = sp2.setup_model(config)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = sp2.ComposableLoss(config, nz, device)
    optimizer = sp2.setup_optimizer(config, model)

    # Train the model
    print("Training model...")
    model, avg_loss = sp2.train(
        model, last_model_state, config,
        train_set, test_set, device,
        optimizer, criterion, criterion
    )

    print(f'Training complete! Final loss: {avg_loss:.5f}')

if __name__ == '__main__':
    main()
```

Run the script with:

```bash
python quickstart_train.py
```

## 5. Working with Example Data

The repository includes sample data and configuration files. To use them:

### Clone the Examples Repository

```bash
git clone https://github.com/MALES-project/examples_speckcn2.git
cd examples_speckcn2
```

### Run with Sample Configuration

Use one of the provided configuration files:

```bash
python example_train.py configurations/configuration_ResNet18.yaml
```

This will:

- Load the sample speckle pattern data
- Train a ResNet18 model to predict turbulence profiles
- Save the trained model and results

### Expected Output

You should see output similar to:

```
Using cuda.
Preparing data...
Loading images from ./data/
Training model...
Epoch 1/100, Loss: 0.12345
Epoch 2/100, Loss: 0.11234
...
Finished Training, Loss: 0.05432
```

## 6. Next Steps

Now that you have `speckcn2` running, here are some suggested next steps:

### Learn More About Configuration

The configuration file controls all aspects of training, from data preprocessing to model architecture. Learn how to customize it:

- **[Configuration Guide](examples/run.md#configuration-file-explanation)** - Detailed explanation of all configuration parameters

### Explore Different Models

`speckcn2` supports multiple model architectures:

- ResNet18, ResNet50, ResNet152
- Custom SCNN (Steerable CNN)
- Equivariant models for rotation-invariant predictions

See the [Models API](api/models.md) for more details.

### Understand Your Data

Learn about the input data format and how to prepare your own SCIDAR observations:

- **[Input Data Guide](examples/input_data.md)** - Data format specifications and preparation

### Run Postprocessing and Analysis

After training, analyze your model's performance:

```python
# Score the model on test data
test_tags, test_losses, test_measures, test_cn2_pred, test_cn2_true, \
    test_recovered_tag_pred, test_recovered_tag_true = sp2.score(
        model, test_set, device, criterion, nz
    )

# Plot results
sp2.plot_J_error_details(config, test_recovered_tag_true, test_recovered_tag_pred)
sp2.plot_loss(config, model, datadirectory)
```

See the [Postprocessing Guide](examples/run.md#plotting) for visualization examples.

### Predict Turbulence Parameters

Beyond Cn² profiles, `speckcn2` can predict important atmospheric parameters:

- **Fried parameter** (`r₀`) - measures atmospheric coherence length
- **Isoplanatic angle** (`θ₀`) - angular size of coherent atmospheric regions  
- **Rytov index** (`σ`) - scintillation strength indicator

Configure these in the `preproc` section of your configuration file.

## Need Help?

- **Full Documentation**: Browse the complete [documentation](index.md) for in-depth guides
- **API Reference**: Check the [Python API](api/api.md) for detailed function documentation
- **Examples**: Explore more [examples](examples/run.md) and use cases
- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/MALES-project/SpeckleCn2Profiler/issues)
- **Contributing**: Want to contribute? See our [Contributing Guidelines](CONTRIBUTING.md)

## Quick Reference

| Task | Command |
|------|---------|
| Install package | `pip install speckcn2` |
| Verify installation | `python -c "import speckcn2"` |
| Run tests | `pytest` |
| Train a model | `python train.py config.yaml` |
| Check documentation | Visit [https://males-project.github.io/SpeckleCn2Profiler/](https://males-project.github.io/SpeckleCn2Profiler/) |

---

**Ready to dive deeper?** Head to the [Examples section](examples/run.md) for complete walkthroughs and detailed explanations.
