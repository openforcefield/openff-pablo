OpenFF Pablo
==============================
[//]: # (Badges)

| **Latest release** | [![Last release tag](https://img.shields.io/github/release-pre/openforcefield/openff-pablo.svg)](https://github.com/openforcefield/openff-pablo/releases) ![GitHub commits since latest release (by date) for a branch](https://img.shields.io/github/commits-since/openforcefield/openff-pablo/latest)  [![Documentation Status](https://readthedocs.org/projects/openff-pablo/badge/?version=latest)](https://openff-pablo.readthedocs.io/en/latest/?badge=latest)                                                                                                                                                                                                                        |
| :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**         | [![GH Actions Status](https://github.com/openforcefield/openff-pablo/actions/workflows/gh-ci.yaml/badge.svg)](https://github.com/openforcefield/openff-pablo/actions?query=branch%3Amain+workflow%3Agh-ci) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/openforcefield/openff-pablo/main.svg)](https://results.pre-commit.ci/latest/github/openforcefield/openff-pablo/main) |

New, independent implementation of OpenFF's `Topology.from_pdb`

OpenFF Pablo is bound by a [Code of Conduct](https://github.com/openforcefield/openff-pablo/blob/main/CODE_OF_CONDUCT.md).

### Installation

This is a pre-release of Pablo and is not yet published in any package manager.
You can install it by managing your own Conda environment and installing it manually.

Here we describe dependency and environment management with Micromamba, but other Conda-compatible package managers such as Conda and Mamba work the same way - just change the name of the executable.

Download the user environment YAML file and create a virtual environment from it:

```
curl https://raw.githubusercontent.com/openforcefield/openff-pablo/refs/heads/main/devtools/conda-envs/user_env.yaml | micromamba env create --name pablo -f /dev/stdin
```

#### Development build

Clone the repository:

```
git clone https://github.com/openforcefield/openff-pablo
```

Create a virtual environment:

```
micromamba create --name pablo-dev
```

Install the development and documentation dependencies:

```
micromamba env update -n pablo-dev --file openff-pablo/devtools/conda-envs/test_env.yaml
micromamba env update -n pablo-dev --file openff-pablo/docs/requirements.yaml
```

Install this package in editable mode:

```
micromamba run -n pablo-dev pip install -e openff-pablo
```

If you want to update your dependencies, rebuild the environment from scratch.

Then activate the environment to run commands in it:

```
micromamba activate pablo-dev
```

Or use `micromamba run -n pablo-dev`.

### Copyright

The OpenFF Pablo source code is hosted at https://github.com/openforcefield/openff-pablo
and is available under the MIT license (see the file [LICENSE](https://github.com/openforcefield/openff-pablo/blob/main/LICENSE)).

Copyright (c) 2025, Josh Mitchell


#### Acknowledgements

Project based on the
[OpenFF Cookiecutter](https://github.com/lilyminium/cookiecutter-openff) version 0.1.
