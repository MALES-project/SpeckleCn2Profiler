# Example: How to run SpeckleCn2Profiler

Here you find examples on the usage of SpeckleCn2Profiler.
This page is divided in two sections, corresponding to the two files that the user has to control to use `speckcn2`:  

1. [run.py](#run-file-explanation)

2. [configuration.yaml](#configuration-file-explanation)

Once you have SpeckleCn2Profiler installed, you will run your workflow as:

```console
python run.py configuration.yaml
```

So let's show you how to set up the `run.py` and `configuration.yaml` files.

## Run File Explanation

The first step of the run is to load the configuration file and the required packages:

```python
import speckcn2 as sp2

config = sp2.load_config(conf_name)
```
then you usually want to load and preprocess the data:
```python

all_images, all_tags, all_ensemble_ids = sp2.prepare_data(config,
                                                          nimg_print=15)
nz = sp2.Normalizer(config)
train_set, test_set = sp2.train_test_split(all_images, all_tags,
                                           all_ensemble_ids, nz)
```
after that you have to define the loss function and the optimizer:
```python
criterion = sp2.ComposableLoss(config, nz, device)
criterion = criterion.to(device)
optimizer = sp2.setup_optimizer(config, model)
criterion_val = sp2.ComposableLoss(config, nz, device, validation=True)
```

Then you can load or create the model and train it:
```python
model, last_model_state = sp2.setup_model(config)
model, average_loss = sp2.train(model, last_model_state, config, train_set,
                                test_set, device, optimizer, criterion, criterion_val)
print(f'Finished Training, Loss: {average_loss:.5f}', flush=True)
```
Once the training is done, you can `score` the model evaluating its performance on the test set and measuring the desired observables.
If you want to do only inference and the model is already trained, you can skip the previous part.
An example of a postprocessing pipeline starts with scoring the model:
```python
test_tags, test_losses, test_measures, test_cn2_pred, test_cn2_true, test_recovered_tag_pred, test_recovered_tag_true = sp2.score(
    model, test_set, device, criterion, nz, nimg_plot=0)
```

### Plotting
After the model is trained, you can plot the results. Here are some examples:

```python
# Plot the distribution of the screen tags
sp2.plot_J_error_details(config, test_recovered_tag_true, test_recovered_tag_pred)
```
![jerr2](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/jerr2.png?raw=true)
```python
# Plot the distribution of the screen tags with bin resolved details
sp2.screen_errors(config, device, test_recovered_tag_pred, test_recovered_tag_true, nbins=20)
```
![jerr](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/jerr.png?raw=true)
```python
# Overview of the tags distribution and how they are predicted
sp2.tags_distribution(config,
                      train_set,
                      test_tags,
                      device,
                      rescale=True,
                      recover_tag=nz.recover_tag)
```
![thist](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/thist.png?raw=true)
```python

# Plot the histograms of the loss function
sp2.plot_histo_losses(config, test_losses, datadirectory)
```
![hls](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/hls.png?raw=true)
```python

# Plot the loss during training
sp2.plot_loss(config, model, datadirectory)
```
![ls](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/ls.png?raw=true)
```python

# Plot the execution time
sp2.plot_time(config, model, datadirectory)
```
![t](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/t.png?raw=true)
```python

# Plot histograms of the different parameters
sp2.plot_param_histo(config, test_losses, datadirectory, test_measures)
```
![hp](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/hp.png?raw=true)
```python

# Plot the parameters of the model vs the loss
sp2.plot_param_vs_loss(config, test_losses, datadirectory, test_measures)
```
![pl](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/pl.png?raw=true)
```python

# Test to see if averaging over speckle patterns improves the results
sp2.average_speckle_input(config, test_set, device, model, criterion, n_ensembles_to_plot=5)
```
![avs](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/avs.png?raw=true)
```python
# Test to see if averaging over speckle patterns improves the results
sp2.average_speckle_output(config, test_set, device, model, criterion, trimming=0.2, n_ensembles_to_plot=20)

```
![ave](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/ave.png?raw=true)

Refer to the [documentation](https://males-project.github.io/SpeckleCn2Profiler/) or one of the [examples](https://github.com/MALES-project/examples_speckcn2) if you want to understand and customize your workflow.


## Configuration File Explanation

Here we explain what it is expected in a typical `configuration.yaml` file.
Notice that many fields are optional and have default values, so you can start with a minimal configuration file and add more details as you need them. In the [example submodule](https://github.com/MALES-project/examples_speckcn2) you can find multiple examples and multiple configuration to take inspiration from.

A typical configuration file is divided in the following sections:

#### speckle
* **nscreens**: The number of screens used in the simulation.
* **hArray**: array corresponding to the altitudes of the screens.
* **split**: The distance from the next screen.
* **lambda**: The wavelength of the laser.
* **original_res**: The original resolution of the images.
* **datadirectory**: The directory where the data files are located.

#### preproc
* **polarize**: A boolean value indicating whether the images should be transformed into polar coordinate.
* **polresize**: The size to which the polarized images are resized.
* **equivariant**: A boolean value indicating whether the images should be made pseudo-equivariant, by setting the azimutal angle to the maximum pixel intensity.
* **randomrotate**: A boolean value indicating whether the images should be randomly rotated.
* **centercrop**: The size of the central crop of the images. Test this value to guarantee that the empty boundaries are removed.
* **resize**: The size to which the images are resized.
* **speckreps**: The number of times that we want to repeat each speckle pattern in order to augment the data. Use only in combination with random rotations.
* **ensemble**: The number of speckle patterns to use as ensemble. This is to train multi-shoot models.
* **ensemble_unif**: A boolean value indicating whether the ensemble is uniformly sampled.
* **normalization**: How to normalize the tags: `unif`, `lin`, `log` or `zscore`.
* **img_normalization**: if `true` normalize the pixel values of the images.
* **dataname**: The name of the file where the preprocessed images are saved.
* **XXX_details**: If `true` then plot the bin resolved details of `XXX` metrics

#### noise
* **D**: Fraction of the field.
* **t**: Width of the spider.
* **snr**: Signal to noise ratio.
* **dT**: Telescope diameter.
* **dO**: Fraction of obscuration.
* **rn**: Amplitude of the noise.
* **fw**: Full well capacity.
* **bit**: Bit level (sample depth).
* **discretize**: A boolean value indicating whether the images should be discretized.
* **rotation_sym**: Values in degrees of the rotation symmetry of the pattern. It is defined in the noise section since random rotations multiple of this value are applied to the images.

#### model
* **name**: String representing the name of the model. Used to store states and plots. It can be any name.
* **type**: The type of the model. We have implemented `resnet18`, `resnet50`, `resnet152` from the ResNet family, and `scnnC8`, `scnnC16`, `small_scnnC16`, which are equivariant CNN.
* **save_every**: The frequency (in epochs) at which the model is saved.
* **pretrained**: A boolean value indicating whether a pretrained model should be used. It is available only for the ResNet.

#### hyppar
* **maxepochs**: The maximum number of epochs for training the model.
* **batch_size**: The size of the batches used in training.
* **lr**: The learning rate for the optimizer.
* **lr_scheduler**: The learning rate scheduler used in training. We have implemented `StepLR`, `ReduceLROnPlateau` and `CosineAnnealingLR`.
* **loss**: The loss function used in training. We have implemented `MSELoss`, `BCELoss` and `Pearson`.
* **early_stopping**: The number of epochs of plateau to wait before stopping the training.
* **optimizer**: The optimizer used in training.

#### loss
* **XXX**: The weight of the loss `XXX` in the total loss. The total loss is the sum of all the losses weighted by their respective weights. A value of 0 means that the loss is not used. The list of available losses includes:
* This section of the YAML file defines the configuration for different loss functions used in a machine learning model. Each key represents a specific type of loss function, and the corresponding value indicates whether that loss function is enabled (1) or disabled (0). Here is a brief explanation of each loss function:
- **MAE (Mean Absolute Error)**: Measures the average magnitude of errors between predicted and actual values.
- **MSE (Mean Squared Error)**: Measures the average of the squares of the errors between predicted and actual values.
- **JMAE**: A variant of MAE that is computed over J, so the screen tag are reconstructed before the evaluation.
- **JMSE**: A variant of MSE that is computed over J, so the screen tag are reconstructed before the evaluation.
- **Cn2MAE**: Mean Absolute Error specific to Cn2 (a parameter related to atmospheric turbulence).
- **Cn2MSE**: Mean Squared Error specific to Cn2.
- **Pearson**: Measures the Pearson correlation coefficient between predicted and actual values.
- **Fried**: A loss function related to the Fried parameter, which is used in optical turbulence.
- **Isoplanatic**: A loss function related to the isoplanatic angle, another parameter in optical turbulence.
- **Rytov**: A loss function related to the Rytov variance, which is used in wave propagation.
- **Scintillation_w**: A loss function related to the weak scintillation index.
- **Scintillation_ms**: A loss function related to the medium-strong scintillation index.

#### val_loss
Same structure as the loss section, but for the validation loss measured at each step over the test set.

#### scnn
This block is used to define the architecture of the SCNN models. The keys are the names of the layers, and the values are the number of filters in each layer. The SCNN models are equivariant CNNs that are used to process the speckle images. A reference example is provided in the `configuration.yaml` file.

#### final_block
This block is used to define the architecture of the final block of any model to get from the image features to the tag predictions. This block has usually a fully connected structure, but here you can control the number of layers, the number of neurons in each layer, regularization, activation, ecc. A reference example is provided in the `configuration.yaml` file.
