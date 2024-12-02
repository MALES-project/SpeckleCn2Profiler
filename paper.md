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
    orcid: 0000-0002-9153-650X
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
 - name: Faculty of Aerospace Engineering, Delft University of Technology, Delft, The Netherlands
   index: 2
date: 13 December 2024
bibliography: paper.bib

---

# Summary

Optical satellite communications is a growing research field with bright commercial perspectives.
The idea is to use lasers to send signals from the ground to satellites, from satellites to satellites and then back to the ground, and the main advantage of using laser communications over radio waves is increased bandwidth, enabling the transfer of more data in less time.
However, one of the challenges for this protocol is the turbulence in the atmosphere that perturbs this transmission. The reduction of the quality of signal communication can be calculated and then compensated, but the knowledge of the turbulence strength is required. To be able to quantify the effect of the turbulence, there are several alternative instrument, but each one with its own limitation. One possibility is to use speckle-based observation, which basically consists in looking at the twinkling of the stars and use their pattern to infer the turbulence profile. This is a non-intrusive method that can be used in real-time, but it requires a deep understanding of the turbulence and the observed speckle patterns, which are highly influenced by the turbulence profile.
The connection between speckle observation and turbulence is not clearly understood, so an analytical theory does not exist.
Here we present `speckcn2`, a Python package that uses machine learning to provide a numerical reconstruction of the turbulence profile.

![Example of speckcn2 pipeline: speckle pattern as input to output a prediction of the turbulence profile (J). \label{fig:prediction}](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/singleprediciton.png?raw=true)

# Statement of need

The turbulence in the atmosphere is a well-known phenomenon that affects the quality of optical communication.
One way to compensate for this effect is to estimate the turbulence strength and then apply a correction to the signal, but there are not many instruments that can provide this information.
`speckcn2` is a Python package to use machine learning to estimate the turbulence and reconstruct its profile.
The idea of using deep learning to compensate for atmospheric turbulence is not new, and it has been already explored in the context of temporal mitigation using videos [@zhang2024spatio] and to compensate static image degradation [@9506614].
However these tools mainly provide a way to mitigate the effect of turbulence, while `speckcn2` aims to provide a numerical reconstruction of the turbulence profile that can be used to understand the turbulence and improve the communication system by inserting the turbulence profile in the communication model.

The approach of `speckcn2` is based on the observation of speckle patterns, which are the result of the interference of light waves that have been perturbed by the atmosphere.
Using PyTorch framework [@pytorch], it is possible to efficiently build, train and deploy deep learning models that predict the turbulence.
`speckcn2` was created mainly for research in aerospace engineering, that want to insert in their model a real-time estimation of the turbulence, but it can also be useful in other fields. The package is designed to be simple to use and flexible enough to handle a variety of tasks. It works equally well with synthetic data from simulations and real data from experiments, making it versatile for different research needs.
Another important feature of `speckcn2` is how easy it is to extend. Researchers from other fields can add new functions or adapt the package to solve problems in their own areas. By combining techniques like equivariance and ensemble learning, it offers a strong and reliable tool for turning images into regression models, opening doors for many innovative applications.


# Key features
## Instrument specialization
When estimating the turbulence, it is of fundamental importance to not mix the instrumental noise with the real effects that is being measured.
As such a fundamental aspect of `speckcn2` is the possibility to train the model with different noise profiles, that can be used to simulate the noise of different instruments.  
These profiles represent the noise from different instruments and can model different detectors, whether real or simulated. By changing the `apply_noise` function, users can model any type of effect related to their research and instruments. The current API provides a series of parameters that can be tuned to simulate the noise of different instruments, such as the signal-to-noise ratio, the detector gain, and the obscuration.

## Equivariant model
To take advantage of the symmetry in the input data, `speckcn2` uses a concept called equivariance [@cohen2016]. This means the model can learn the same features no matter how the input data is oriented. This is especially helpful for turbulence reconstruction, where the direction of the speckle pattern is not relevant.

`speckcn2` supports two types of equivariance: weak and strong. Weak equivariance is achieved by randomly rotating the input data, which can then be used with any model from torchvision [@torchvision], including fine-tuning ResNets [@resnet].

Strong equivariance is achieved using the equivariant sparse convolutional neural network (escnn) [@escnn1; @escnn2]. These networks are more powerful for this type of problem but are harder to train.



## Ensemble learning
`speckcn2` can also use ensemble learning by averaging the predictions from multiple input images. This means each model prediction requires a set of multiple input images. This is only useful if the input images change more quickly than the output. Since this is not the case for laser communications, this feature is optional and can be turned off.



## Software implementation
`speckcn2` is implemented in Python and uses PyTorch [@pytorch] for its machine learning tasks. This allows it to easily take advantage of GPU acceleration by running PyTorch models on the GPU, making computations faster and more efficient.

The package is easy to install via pip and PyPI, and it works on both Linux and MacOS. It has a simple and user-friendly API that lets users quickly build, train, and evaluate models. Whether you are a beginner or an experienced user, `speckcn2` is designed to be accessible and flexible.

For new users, the extensive documentation and examples provide a great starting point, helping them get up to speed quickly. Experienced users will appreciate the flexibility of the package, which allows for customization to meet specific research needs.

# Acknowledgements
The authors would like to acknowledge the Netherlands eScience Center for the funding provided under grant number NLESC.SSIML.2022c.021.

# References
