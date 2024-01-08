[![Python application test](https://github.com/MALES-project/SpeckleCn2Profiler/actions/workflows/test.yaml/badge.svg)](https://github.com/MALES-project/SpeckleCn2Profiler/actions/workflows/test.yaml)

![MALES banner](https://raw.githubusercontent.com/MALES-project/SpeckleCn2Profiler/main/speckcn2/assets/logo_on_white.png#gh-light-mode-only)
![MALES banner](https://raw.githubusercontent.com/MALES-project/SpeckleCn2Profiler/main/speckcn2/assets/logo_on_black.png#gh-dark-mode-only)

# SpeckleCn2Profiler: Improving Satellite Communications with SCIDAR and Machine Learning

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

- **Documentation:** In-depth documentation explaining the methodology, algorithms used, and guidelines for using the code.

## Getting Started

To get started with the project, follow these steps:

1. **Install the package:**
   ```bash
   pip install speckcn2
   ```

2. **Explore the Code:**
    Dive into the codebase to understand the implementation details and customize it according to your needs.

## Contribution Guidelines

We welcome contributions to improve and expand the capabilities of this project. If you have ideas, bug fixes, or enhancements, please submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
