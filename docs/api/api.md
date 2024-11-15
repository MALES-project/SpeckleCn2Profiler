The main API of the `speckcn2` package is composed of the following modules:

- **speckcn2.preprocess**: contains the functions to preprocess the data before training the model.

- **speckcn2.models**: contains the classes that define the model architecture. In particular we support `ResNet` and `SCNN` based architectures.

- **speckcn2.mlops**: contains the functions to perform operations involving the models, where the most important are:
    - [speckcn2.mlops.train][speckcn2.mlops.train]
    - [speckcn2.mlops.score][speckcn2.mlops.score]

- **speckcn2.plot**: contains the functions to plot the results of the model.
