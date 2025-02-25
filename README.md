OpenFF Pablo
==============================
[//]: # (Badges)

| **Latest release** | [![Last release tag](https://img.shields.io/github/release-pre/openforcefield/openff-pablo.svg)](https://github.com/openforcefield/openff-pablo/releases) ![GitHub commits since latest release (by date) for a branch](https://img.shields.io/github/commits-since/openforcefield/openff-pablo/latest)  [![Documentation Status](https://readthedocs.org/projects/openff-pablo/badge/?version=latest)](https://openff-pablo.readthedocs.io/en/latest/?badge=latest)                                                                                                                                                                                                                        |
| :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**         | [![GH Actions Status](https://github.com/openforcefield/openff-pablo/actions/workflows/gh-ci.yaml/badge.svg)](https://github.com/openforcefield/openff-pablo/actions?query=branch%3Amain+workflow%3Agh-ci) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/openforcefield/openff-pablo/main.svg)](https://results.pre-commit.ci/latest/github/openforcefield/openff-pablo/main) |

New, independent implementation of `Topology.from_pdb`

OpenFF Pablo is bound by a [Code of Conduct](https://github.com/openforcefield/openff-pablo/blob/main/CODE_OF_CONDUCT.md).

### Installation

To build OpenFF Pablo from source,
we highly recommend using virtual environments.
If possible, we strongly recommend that you use
[Anaconda](https://docs.conda.io/en/latest/) as your package manager.
Below we provide instructions both for `conda` and
for `pip`.

#### With conda

Ensure that you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

Create a virtual environment and activate it:

```
conda create --name pablo
conda activate pablo
```

Install the development and documentation dependencies:

```
conda env update --name pablo --file devtools/conda-envs/test_env.yaml
conda env update --name pablo --file docs/requirements.yaml
```

Build this package from source:

```
pip install -e .
```

If you want to update your dependencies (which can be risky!), run:

```
conda update --all
```

And when you are finished, you can exit the virtual environment with:

```
conda deactivate
```

#### With pip

To build the package from source, run:

```
pip install -e .
```

If you want to create a development environment, install
the dependencies required for tests and docs with:

```
pip install -e ".[test,doc]"
```

### Copyright

The OpenFF Pablo source code is hosted at https://github.com/openforcefield/openff-pablo
and is available under the MIT license (see the file [LICENSE](https://github.com/openforcefield/openff-pablo/blob/main/LICENSE)).

Copyright (c) 2025, Josh Mitchell


#### Acknowledgements

Project based on the
[OpenFF Cookiecutter](https://github.com/lilyminium/cookiecutter-openff) version 0.1.
