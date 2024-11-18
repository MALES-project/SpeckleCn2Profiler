---
title: 'Speckle Cn2 Profiler:
Improving Satellite Communications with SCIDAR and Machine Learning'
tags:
  - Python
  - machine learning
  - signal processing
  - turbulence
  - image processing
  - equivariance
authors:
  - name: Simone Ciarella
    orcid: 0000-0002-9247-139X
    corresponding: true
    affiliation: 1
  - name: Luisa Orozco
    orcid: 0000-0002-6668-8042
    affiliation: 1
  - name: Victor Azizi
    orcid: 0000-0003-3535-8320
    affiliation: 1
  - name: Marguerite Arvis
    orcid: 0009-0006-7409-3985
    affiliation: 2
  - name: Rudolf Saathof
    orcid: 0000-0003-0368-0139
    affiliation: 2

affiliations:
 - name: Netherlands eScience Center, Amsterdam, The Netherlands
   index: 1
   ror: 00hx57361
 - name: Delft University of Technology, Delft, NL
   index: 2
date: 13 December 2024
bibliography: paper.bib

---

# Summary

Optical satellite communications is a growing research field with bright commercial perspectives. One of the challenges for optical links through the atmosphere is turbulence, which is also apparent by the twinkling of stars. The reduction of the quality of signal communication can be calculated and then compensated, but the knowledge og the turbulence strength is required. To be able to quantify the effect of the turbulence, there are several alternative instrument, but each one with its own limitation. One possibility is to use  speckle-based observation, which are highly influenced by the turbulence profile. However the connection between speckle observation and turbulence is not clearly understood, so approximated numerical methods are required to reconstruct the turbulence profile.

![Example of speck2cn pipeline: speckle pattern as input to output a prediction of the turbulence profile (J). \label{fig:prediction}](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/singleprediciton.png?raw=true)

# Statement of need

`speck2cn` is a Python package to use machine learning for turbulence reconstruction.
Using PyTorch framework [@pytorch], it is possible to build, train and deploy deep learning models that are efficient and easy to use.
The API for `speck2cn` was
designed to provide a user-friendly interface to build, train and evaluate a machine learning model that predicts turbulence from speckle patterns.


`speck2cn` was created mainly for research in aerospace engineering, but it can also be useful in other fields. The package is designed to be simple to use and flexible enough to handle a variety of tasks. It works equally well with synthetic data from simulations and real data from experiments, making it versatile for different research needs.

Another important feature of `speck2cn` is how easy it is to extend. Researchers from other fields can add new functions or adapt the package to solve problems in their own areas. By combining techniques like equivariance and ensemble learning, it offers a strong and reliable tool for turning images into regression models, opening doors for many innovative applications.


# Key features
## Instrument specialization
To work with different instruments, `speck2cn` can be trained with various noise profiles. These profiles represent the noise from different instruments and can model different speckle detectors, whether real or simulated. By changing the `apply_noise` function, users can model any type of effect related to their research and instruments.

## Equivariant model
To take advantage of the symmetry in the input data, `speck2cn` uses a concept called equivariance [@cohen2016]. This means the model can learn the same features no matter how the input data is oriented. This is especially helpful for turbulence reconstruction, where the direction of the speckle pattern is not relevant.

`speck2cn` supports two types of equivariance: weak and strong. Weak equivariance is achieved by randomly rotating the input data, which can then be used with any model from torchvision [@torchvision], including fine-tuning ResNets [@resnet].

Strong equivariance is achieved using the equivariant sparse convolutional neural network (escnn) [@escnn1; @escnn2]. These networks are more powerful for this type of problem but are harder to train.



## Ensemble learning
`speck2cn` can also use ensemble learning by averaging the predictions from multiple input images. This means each model prediction requires a set of multiple input images. This is only useful if the input images change more quickly than the output. Since this is not the case for laser communications, this feature is optional and can be turned off.



## Software implementation
`speck2cn` is implemented in Python and uses PyTorch [@pytorch] for its machine learning tasks. This allows it to easily take advantage of GPU acceleration by running PyTorch models on the GPU, making computations faster and more efficient.

The package is easy to install via pip and PyPI, and it works on both Linux and MacOS. It has a simple and user-friendly API that lets users quickly build, train, and evaluate models. Whether you are a beginner or an experienced user, `speck2cn` is designed to be accessible and flexible.

For new users, the extensive documentation and examples provide a great starting point, helping them get up to speed quickly. Experienced users will appreciate the flexibility of the package, which allows for customization to meet specific research needs.

# Acknowledgements
The authors would like to acknowledge the Netherlands eScience Center for the funding
provided under grant number NLESC.XXXX
