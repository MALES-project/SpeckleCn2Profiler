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

![Graphical abstract](/src/speckcn2/assets/cn2_profile.gif)

## Overview

Optical satellite communications is a growing research field with bright commercial perspectives. One of the challenges for optical links through the atmosphere is turbulence, which is also apparent by the twinkling of stars. The reduction of the quality can be calculated, but it needs the turbulence strength over the path the optical beam is running. Estimation of the turbulence strength is done at astronomic sites, but not at rural or urban sites. To be able to do this, a simple instrument is required. We want to propose to use a single star Scintillation Detection and Ranging (SCIDAR), which is an instrument that can estimate the turbulence strength, based on the observation of a single star. Here, reliable signal processing of the received images of the star is most challenging. We propose to solve this by Machine Learning.

## Project Goals

The primary objectives of this project are:

1. **Turbulence Strength Estimation:** Develop a robust algorithm using Machine Learning to estimate turbulence strength based on SCIDAR data.

2. **Signal Processing Enhancement:** Implement advanced signal processing techniques to improve the accuracy and reliability of turbulence strength calculations.

3. **Adaptability to Various Sites:** Ensure the proposed solution is versatile enough to be deployed in diverse environments, including rural and urban locations.

## Repository Contents

This repository contains:

- **Machine Learning Models:** Implementation of machine learning models tailored for turbulence strength estimation from SCIDAR data.

- **Signal Processing Algorithms:** Advanced signal processing algorithms aimed at enhancing the quality of received star images.

- **Dataset:** Sample datasets for training and testing the machine learning models.

- **[Documentation](https://males-project.github.io/SpeckleCn2Profiler/):** In-depth documentation explaining the methodology, algorithms used, and guidelines for using the code.

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

## Contribution Guidelines

We welcome contributions to improve and expand the capabilities of this project. If you have ideas, bug fixes, or enhancements, please submit a pull request.
Check out our [Contributing Guidelines](CONTRIBUTING.md#Getting-started-with-development) to get started with development.

## Generative-AI Disclaimer

Parts of the code have been generated and/or refined using GitHub Copilot. All AI-output has been verified for correctness, accuracy and completeness, revised where needed, and approved by the author(s).

## How to cite

Please consider citing this software that is published in Zenodo under the DOI [10.5281/zenodo.11447920](https://zenodo.org/records/11447920).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
