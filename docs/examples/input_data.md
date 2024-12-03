# Example: Input data

The `speckcn2` tool expects a set of **2D black-and-white squared** images along with their corresponding prediction labels.
In its main application, these images are speckle patterns, and the values represent turbulence strength. However, the tool can be adapted for any image regression task, as long as the data is provided in the correct format.

Given the large amount of data, we use **hdf5** format for optimal storage.

Each image should be saved in a separate hdf5 file named `<NAME>-<ID>.h5`. The corresponding prediction value should be stored in a file named `<NAME>_label.h5`. Images with no corresponding label will be ignored.
Here, `<NAME>` is a common name for the dataset, so we can use `<NAME>='speckle_xxx'` for speckle patterns.
Instead, `<ID>` is an identifier that takes into account the fact that different input data can correspond to the same turbulence labels, so we have `speckle_A-1.h5` and `speckle_A-2.h5` but only a single `speckle_A_label.h5` that is the turbulence strength for both images.

In terms of dimensions, the images are stored as 2D arrays of size $N\times N$, while the labels are stored as 1D arrays of size $M$. This structure ensures that the data is organized and easily accessible for processing and analysis.

> **&#9432; config.yaml:**  In order to process the data in the correct format, the `config.yaml` file should contain the following information:
> ```yaml
> speckle:
>   nscreens: M # This is the number of labels that you want to predict for each image
>   original_res: N # This is the resolution of the images
> ```
