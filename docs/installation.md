# Installation
To get started with the project, follow these steps:

- **Prerequisites:**
  In order to correctly install `speckcn2` you need `python3.9` or higher. If you don't have it installed, you can download it from the [official website](https://www.python.org/downloads/). You will also need the Python header files that are required to compile extensions. On Ubuntu, you can install them with:
  ```bash
  apt-get install python3-dev
  ```

- **Install the package:**
   ```bash
   python -m pip install speckcn2
   ```

- **Or: Clone the repository:**
  ```bash
  git clone https://github.com/MALES-project/SpeckleCn2Profiler.git
  cd SpeckleCn2Profiler
  git submodule init
  git submodule update
  ```

## MacOS M1 arm64

Some dependencies (e.g. `scikit`) do not support the latest python version (3.12). Also `py3nj`, a dependency of `escnn`, requires openmp. We've installed this via homebrew and thus explicitly specifying the C compiler (gnu) prior to installation of this package does the trick.

```sh
conda create -n speckcn2 python=3.10
conda activate speckcn2
CC=gcc-13 pip3 install py3nj # install py3nj before with gcc instead of clang
pip install -e .
```
