# OpenFF Pablo

[//]: # (Badges)

| **Latest release** | [![Last release tag](https://img.shields.io/github/release-pre/openforcefield/openff-pablo.svg)](https://github.com/openforcefield/openff-pablo/releases/latest)  [![Documentation Status (Stable)](https://img.shields.io/readthedocs/openff-pablo/stable?logo=readthedocs&logoColor=white&label=docs%20-%20stable)](https://openff-pablo.readthedocs.io/en/stable/)                                                                                                                                                                                                                        |
| :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`main` Branch**         | [![GH Actions Status](https://img.shields.io/github/actions/workflow/status/openforcefield/openff-pablo/gh-ci.yaml?branch=main&logo=github&logoColor=white&label=CI%20-%20main)](https://github.com/openforcefield/openff-pablo/actions?query=branch%3Amain+workflow%3Agh-ci) [![Documentation Status (Latest)](https://img.shields.io/readthedocs/openff-pablo/latest?logo=readthedocs&logoColor=white&label=docs%20-%20latest)](https://openff-pablo.readthedocs.io/en/latest/) ![GitHub commits since latest release (by date) for a branch](https://img.shields.io/github/commits-since/openforcefield/openff-pablo/latest?include_prereleases&sort=semver)     |

New implementation of OpenFF's `Topology.from_pdb`

OpenFF Pablo is bound by a [Code of Conduct](https://github.com/openforcefield/openff-pablo/blob/main/CODE_OF_CONDUCT.md).

## Installation

This is a pre-release of Pablo and is not yet published in any package manager.
You can install it by managing your own Conda environment and installing it manually.

Here we describe dependency and environment management with Micromamba, but other Conda-compatible package managers such as Conda and Mamba work the same way - just change the name of the executable.

Install Pablo and its dependencies into the current environment:

```sh
# Install dependencies and pip into current environment
micromamba install -c conda-forge pip 'python>=3.12' 'openff-toolkit-base>=0.17.1' rustworkx rdkit openmm pyxdg gemmi
# Install Pablo's latest release into current environment via pip
pip install git+https://github.com/openforcefield/openff-pablo.git@v0.2.2
```

### Development build

Clone the repository:

```sh
git clone https://github.com/openforcefield/openff-pablo
```

Create a virtual environment:

```sh
micromamba create --name pablo-dev
```

Install the development and documentation dependencies:

```sh
micromamba env update -n pablo-dev --file openff-pablo/devtools/conda-envs/test_env.yaml
micromamba env update -n pablo-dev --file openff-pablo/devtools/conda-envs/docs_env.yaml
```

Install Pablo in editable mode:

```sh
micromamba run -n pablo-dev pip install -e openff-pablo
```

Then activate the environment to run commands in it:

```sh
micromamba activate pablo-dev
```

Or use `micromamba run -n pablo-dev`. If you want to update your dependencies, rebuild the environment from scratch.

## Copyright

The OpenFF Pablo source code is hosted at <https://github.com/openforcefield/openff-pablo>
and is available to all under the MIT license (see the file [LICENSE](https://github.com/openforcefield/openff-pablo/blob/main/LICENSE)).

Copyright (c) 2025, Open Force Field Initiative


## Acknowledgements

Project based on the
[OpenFF Cookiecutter](https://github.com/lilyminium/cookiecutter-openff) version 0.1.
