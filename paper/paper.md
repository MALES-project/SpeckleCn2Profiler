---
title: 'Speckle Cn2 Profiler:
Improving Satellite Communications with Machine Learning'
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

Optical satellite communications is a growing research field in which -using lasers- signals can be sent from the ground to satellites, from satellites to satellites and then back to the ground. The main advantage of using laser communication over radio waves is increased bandwidth that enables the transfer of more data in less time.
However, one of the challenges for this protocol is the turbulence in the atmosphere that perturbs such transmission.
The reduction of the quality of signal communication can be calculated and then compensated, but this requires a knowledge of the turbulence strength.
A common way to model the turbulence is to use the refractive index structure constant, $C_n^2$, which is a measure of the strength of the turbulence. Its profile can be used to estimate the effect of the turbulence on the signal and then apply a correction.
To measure $C_n^2$, there are several alternative instruments, each one with its own limitations. One possibility is to use speckle-based observation, which consists on looking at the twinkling of the stars and use their pattern to infer the turbulence profile. This is a non-intrusive method that can be used in real time, but it requires a deep understanding of the turbulence and the observed speckle patterns, which are highly influenced by the turbulence profile.
The connection between speckle observation, and turbulence ($C_n^2$) is not clearly understood, so an analytical theory does not exist.
Here we present `speckcn2`, a Python package that uses machine learning to provide a numerical reconstruction of the turbulence profile from a speckle pattern [@ciarella2024].


![Example of `speckcn2` pipeline: speckle pattern as input to output a prediction of the turbulence profile (J). \label{fig:prediction}](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/single_prediction.png?raw=true){ width=90% }


# Statement of need

While deep learning has been applied to mitigate atmospheric turbulence effects in imaging through temporal mitigation using videos [@zhang2024spatio] and static image degradation compensation [@9506614], these approaches focus on visual correction rather than quantitative turbulence characterization. In contrast, `speckcn2` addresses the critical need for numerical reconstruction of $C_n^2$ profiles, enabling both scientific understanding of atmospheric turbulence and practical integration into communication system models for performance optimization.

The primary target users of `speckcn2` are aerospace engineers working on optical satellite communication systems, atmospheric scientists studying turbulence phenomena, and astronomers developing adaptive optics systems. The package provides these communities with a tool to estimate turbulence profiles from speckle observations where analytical theories are lacking, using machine learning to bridge this gap.

Existing software packages address atmospheric turbulence from different perspectives. AOtools [@AOtools] provides general-purpose adaptive optics utilities including phase screen generation and turbulence parameter conversions, but lacks ML-based profile reconstruction from observations. FAST [@FAST] (Fourier domain Adaptive optics Simulation Tool) offers rapid Monte Carlo characterization of free space optical links with turbulence modeling, while OOPAO [@OOPAO] (Object-Oriented Python Adaptive Optics) provides end-to-end adaptive optics simulation. However, both focus on forward modeling—simulating turbulence effects given known profiles—rather than the inverse problem of reconstructing profiles from measurements. Traditional $C_n^2$ profiling instruments such as radiosondes, SCIDAR, or MASS require specialized hardware and intrusive deployment. `speckcn2` uniquely combines ML-based inverse modeling with speckle pattern analysis to provide explicit turbulence profile reconstruction in real-time, using only optical detection systems without specialized hardware. This fills a critical gap for applications requiring quantitative atmospheric characterization from passive observations.

Built on PyTorch [@pytorch], the package is designed to handle diverse image regression tasks with both synthetic and experimental data. By combining techniques like equivariance and ensemble learning, `speckcn2` provides a robust framework applicable beyond aerospace engineering to any field requiring image-to-profile regression. Complete documentation and API references are available at https://males-project.github.io/SpeckleCn2Profiler/, with installation instructions and examples to facilitate adoption across different research communities.

# Key features
## Instrument specialization
When estimating the turbulence features, it is of fundamental importance to not mix the instrumental noise with the real effects that are being measured.
A fundamental aspect of `speckcn2` is the possibility to train models with different noise profiles, representing the noise of different instruments and modeling different detectors, whether real or simulated. By adapting the `apply_noise` function, users can model any type of effect related to their research and instruments. The current API provides a series of parameters that can be tuned to simulate the noise of different instruments, such as the signal-to-noise ratio, the detector gain, and the obscuration.

## Equivariant model
To take advantage of the symmetry in the input data, `speckcn2` uses a concept called equivariance [@cohen2016]. This means that the model can learn the same features independently of the input data orientation. This is especially helpful for turbulence reconstruction, where the direction of the speckle pattern is not relevant.

`speckcn2` supports two types of equivariance: weak and strong. Weak equivariance is achieved by randomly rotating the input data, which can then be used with any model from torchvision [@torchvision], including fine-tuning ResNets [@resnet].

Strong equivariance is achieved by using the equivariant sparse convolutional neural network (escnn) [@escnn1; @escnn2]. These networks are more powerful for this type of problem but are harder to train.

## Ensemble learning
`speckcn2` can also use ensemble learning by averaging the predictions from multiple input images. This means that the prediction of each model requires a set of multiple input images. This is only useful if the input images change faster than the output. Since this is not the case for laser communications, this feature is optional and can be turned off.

# Applications: STORM for Laser Satellite Communications

A notable application of `speckcn2` is STORM (Speckle-based Turbulence Observation and Reconstruction via Machine learning) [@arvis2024storm], designed for line-of-sight turbulence profiling in laser satellite communications. STORM reconstructs 8-layer $C_n^2(h)$ profiles from single-shot speckle patterns, addressing limitations of traditional methods like SCIDAR or SLODAR that require binary sources or time-series measurements. This single-source-single-shot capability is crucial for tracking fast-moving LEO satellites where the air column changes rapidly between measurements. Simulation results show over 90% accuracy on the Fried parameter, isoplanatic angle, and Rytov index, enabling improved adaptive optics design and communication link optimization for optical space communications.

## Software implementation
`speckcn2` is implemented in Python and uses PyTorch [@pytorch] for its machine learning tasks. This allows the user to take advantage of GPU acceleration, making computations faster and more efficient.

The package is published to PyPI and is easy to install via pip, and it works on both Linux and MacOS. It has a simple and user-friendly API that lets users quickly build, train, and evaluate models. Whether you are a beginner or an experienced user, `speckcn2` is designed to be accessible and flexible.

For new users, the documentation and examples provide a great starting point, helping them get up to speed quickly. Experienced users will appreciate the flexibility of the package, which allows for customization to meet specific research needs.

# Acknowledgements
The authors would like to acknowledge the Netherlands eScience Center for the funding provided under grant number NLESC.SSIML.2022c.021.

# References
