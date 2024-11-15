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
    orcid: 0000-0002-9247-139X
    affiliation: 1
  - name: Victor Azizi
    orcid: 0000-0002-9247-139X
    affiliation: 1
  - name: Marguerite Arvis
    orcid: 0000-0002-9247-139X
    affiliation: 2
  - name: Rudolf Saathof
    orcid: 0000-0002-9247-139X
    affiliation: 2

affiliations:
 - name: Netherlands eScience Center, Amsterdam, The Netherlands
   index: 1
   ror: 00hx57361
 - name: TU Delft, The Netherlands
   index: 2
date: 13 December 2024
bibliography: paper.bib

---

# Summary

Optical satellite communications is a growing research field with bright commercial perspectives. One of the challenges for optical links through the atmosphere is turbulence, which is also apparent by the twinkling of stars. The reduction of the quality of signal communication can be calculated and then compensated, but the knowledge og the turbulence strength is required. To be able to quantify the effect of the turbulence, there are several alternative instrument, but each one with its own limitation. One possibility is to use  speckle-based observation, which are highly influenced by the turbulence profile. However the connection between speckle observation and turbulence is not clearly understood, so approximated numerical methods are required to reconstruct the turbulence profile.

![Example of speckcn2 pipeline: speckle pattern as input to output a prediction of the turbulence profile (J). \label{fig:prediction}](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/singleprediciton.png?raw=true)

# Statement of need

`speckcn2` is a Python package to use machine learning for turbulence reconstruction.
Using PyTorch framework [@pytorch], it is possible to build, train and deploy deep learning models that are efficient and easy to use.
The API for `speckcn2` was
designed to provide a user-friendly interface to build, train and evaluate a machine learning model that predicts turbulence from speckle patterns.
`Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package `[@astropy]` (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Key features
## Instrument specialization
It can change the noise profile due to the instrument, to model different type of speckle deterctors.

## Equivariant model
It can do equivariant learning either the weak way by doing rotations or the strong way with escnn

## Ensemble learning
it can do ensemble leatning by using multiple patterns at the same time

## Software implementation
The most computationally expensive part of open-DARTS is written in C++ with OpenMP
parallelization. open-DARTS can be installed as a Python module and it has a Python-based
interface, which makes it suitable for teaching and users unfamiliar with C++ language. There
are several benefits of this approach compared to a code fully written in C++.
• Easy installation via pip and PyPI.
• No need to install compilers.
• Flexible implementation of simulation framework, physical modelling and grids.
• Easy data visualization, including internal arrays and vtk.
• Use popular Python modules within open-DARTS and the user’s model for data processing
and input/output.
• Coupling with other Python-based numerical modelling software.

# Acknowledgements
The authors would like to acknowledge the Netherlands eScience Center for the funding
provided under grant number NLESC.XXXX
