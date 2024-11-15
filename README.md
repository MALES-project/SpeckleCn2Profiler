[![Python application test](https://github.com/MALES-project/SpeckleCn2Profiler/actions/workflows/test.yaml/badge.svg)](https://github.com/MALES-project/SpeckleCn2Profiler/actions/workflows/test.yaml)
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/SCiarella/ee30d5a40792fc1de92e9dcf0d0e092a/raw/covbadge.json)
[![Documentation Status](https://readthedocs.org/projects/gemdat/badge/?version=latest)](https://males-project.github.io/SpeckleCn2Profiler/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/speckcn2)](https://pypi.org/project/speckcn2/)
[![PyPI](https://img.shields.io/pypi/v/speckcn2)](https://pypi.org/project/speckcn2/)
[![FAIR checklist badge](https://fairsoftwarechecklist.net/badge.svg)](https://fairsoftwarechecklist.net/v0.2?f=31&a=30110&i=21202&r=132)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11447921.svg)](https://doi.org/10.5281/zenodo.11447921)
[![RSD](https://img.shields.io/badge/rsd-speckcn2-00a3e3.svg)](https://research-software-directory.org/software/speckcn2)

![MALES banner](https://raw.githubusercontent.com/MALES-project/SpeckleCn2Profiler/main/src/speckcn2/assets/logo_on_white.png#gh-light-mode-only)
![MALES banner](https://raw.githubusercontent.com/MALES-project/SpeckleCn2Profiler/main/src/speckcn2/assets/logo_on_black.png#gh-dark-mode-only)

<!---
![MALES banner](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/speckcn2/assets/logo_on_white.png#gh-light-mode-only)
![MALES banner](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/speckcn2/assets/logo_on_black.png#gh-dark-mode-only)
-->

# SpeckleCn2Profiler:
### Improving Satellite Communications with SCIDAR and Machine Learning

![Graphical abstract](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/cn2_profile.gif?raw=true)


## Overview

Optical satellite communications is a growing research field with bright commercial perspectives. One of the challenges for optical links through the atmosphere is turbulence, which is also apparent by the twinkling of stars. The reduction of the quality can be calculated, but it needs the turbulence strength over the path the optical beam is running. Estimation of the turbulence strength is done at astronomic sites, but not at rural or urban sites. To be able to do this, a simple instrument is required. We want to propose to use a single star Scintillation Detection and Ranging (SCIDAR), which is an instrument that can estimate the turbulence strength, based on the observation of a single star. In this setting, reliable signal processing of the received images of the star is most challenging. We propose to solve this by Machine Learning.

## Repository Contents

This repository contains the workflow to implement and train machine learning models for turbulence strength estimation from SCIDAR data. Extensive **[Documentation](https://males-project.github.io/SpeckleCn2Profiler/)** is available to explain the methodology, algorithms used, and guidelines for using the code.

## Getting Started

To get started with the project, follow these steps:

- **Install the package:**
   ```bash
   python -m pip install speck2cn
   ```

- **Or: Clone the repository:**
  ```bash
  git clone https://github.com/MALES-project/SpeckleCn2Profiler.git
  cd SpeckleCn2Profiler
  git submodule init
  git submodule update
  ```

## Usage

To use the package, you run the commands such as:

```console
python <mycode.py> <path_to_config.yml>
```

where `<mycode.py>` is the name of the script that trains/uses the `speckcn2` model and `<path_to_config.yml>` is the path to the configuration file.

[Here](https://males-project.github.io/SpeckleCn2Profiler/example) you can find a typical example run and an explanation of all the main configuration parameter. In the [example submodule](https://github.com/MALES-project/examples_speckcn2/) you can find multiple examples and multiple configuration to take inspiration from.

## What can we predict?

A machine learning model trained using `speckcn2` can predict:

##### 1. Instantaneous turbulence strength
![prediction](https://github.com/MALES-project/SpeckleCn2Profiler/blob/main/src/speckcn2/assets/singleprediciton.png?raw=true)
Given a speckle pattern, the model can predict the instantaneous turbulence strength and also provide an uncertainty estimate if more patterns are available.

##### 2. Parameters estimation
The model can also estimate important parameters that are useful for the analysis of the speckle pattern. At the moment we support:
* Fried parameter `r0`
* Isoplanatic angle `θ0`
* Rytov Index `σ`

We also provide histograms of the estimated parameters and the error of the estimation.


## Contribution Guidelines

We welcome contributions to improve and expand the capabilities of this project. If you have ideas, bug fixes, or enhancements, please submit a pull request.
Check out our [Contributing Guidelines](CONTRIBUTING.md#Getting-started-with-development) to get started with development.

## Generative-AI Disclaimer

Parts of the code have been generated and/or refined using GitHub Copilot. All AI-output has been verified for correctness, accuracy and completeness, revised where needed, and approved by the author(s).

## How to cite

Please consider citing this software that is published in Zenodo under the DOI [10.5281/zenodo.11447920](https://zenodo.org/records/11447920).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
