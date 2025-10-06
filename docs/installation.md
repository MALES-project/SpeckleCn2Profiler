# Installation

This guide provides step-by-step instructions for installing `speckcn2` on different platforms and setups.

## Prerequisites

To correctly install `speckcn2`, you need **Python 3.9 or higher**. If you don't have it installed, you can download it from the [official Python website](https://www.python.org/downloads/).

You will also need header files required to compile Python extensions (contained in `python3-dev`):

=== "Ubuntu/Debian"
    ```bash
    sudo apt-get install python3-dev
    ```

=== "Fedora/RHEL"
    ```bash
    sudo dnf install python3-devel
    ```

=== "macOS"
    ```bash
    xcode-select --install
    ```

=== "Windows"
    Headers are included with the official Python installer - no additional steps needed.

## Installation Options

### Option 1: Install from PyPI (Recommended)

This is the simplest way to install `speckcn2` for regular usage:

```bash
python -m pip install speckcn2
```

#### Using Virtual Environment (Strongly Recommended)

We **strongly recommend** using a virtual environment to avoid dependency conflicts:

=== "Linux/macOS"
    ```bash
    # Create virtual environment
    python -m venv speckcn2-env

    # Activate virtual environment
    source speckcn2-env/bin/activate

    # Install the package
    python -m pip install speckcn2
    ```

=== "Windows"
    ```cmd
    # Create virtual environment
    python -m venv speckcn2-env

    # Activate virtual environment
    speckcn2-env\Scripts\activate

    # Install the package
    python -m pip install speckcn2
    ```

#### Verify Installation

After installation, verify that everything works correctly:

```bash
python -c "import speckcn2; print('Installation successful!')"
```

### Option 2: Development Installation

For advanced users and developers who want to modify the code or contribute to the project:

```bash
git clone https://github.com/MALES-project/SpeckleCn2Profiler.git
cd SpeckleCn2Profiler
git submodule init
git submodule update
pip install -e .
```

The `-e` flag installs the package in "editable" mode, meaning changes to the source code will be immediately reflected without reinstalling.

## Platform-Specific Instructions

### macOS with Apple Silicon (M1/M2)

Apple Silicon Macs require special handling due to dependency compatibility issues:

!!! warning "Python Version Limitation"
    Some dependencies (e.g., `scikit-learn`) may not support the latest Python version (3.12). We recommend using Python 3.10 or 3.11.

The `py3nj` dependency of `escnn` requires OpenMP, which needs to be installed via Homebrew with explicit compiler specification:

```bash
# Create conda environment with compatible Python version
conda create -n speckcn2 python=3.10
conda activate speckcn2

# Install py3nj with GNU compiler (instead of clang)
CC=gcc-13 pip install py3nj

# Install speckcn2
pip install -e .
```

!!! tip "Installing GCC"
    If you don't have `gcc-13` installed, you can install it via Homebrew:
    ```bash
    brew install gcc
    ```
