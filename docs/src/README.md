# Compiling OpenFF Pablo's Documentation

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/).
To compile the docs, first install the documentation dependencies into a new environment called `pablo-docs`:

```bash
mamba env create -n pablo-docs -f ../devtools/conda-envs/docs_env.yaml
```

Once installed, you can use Sphinx to render the docs:

```bash
mamba activate pablo-docs
make html
```

Or from the root directory of the repository, use `sphinx-build` directly:

```bash
sphinx-build -j auto -b html docs docs/_build/html
```

The compiled docs will be in the `_build/html/` directory and can be viewed by opening `index.html`.

A configuration file for [Read The Docs](https://readthedocs.org/) (readthedocs.yaml) is included in the top level of the repository. To use Read the Docs to host your documentation, go to https://readthedocs.org/ and connect this repository. You may need to change your default branch to `main` under Advanced Settings for the project.

A configuration for automatically documenting OpenFF software with AutoSummary is included. The project is processed and documented as follows:

1. A module (or package) is documented if and only if it satisfies all the following criteria:
    1. Its parent module is documented, or it is the root module.
    2. It is public (ie, its name does not begin with an underscore).
    3. It is not listed in the `autosummary_context["exclude_modules"]` list in `conf.py`.
2. If a documented module has a `__all__` attribute that is a list of strings, its members will be documented if and only if their names appear in `__all__`.
3. If a documented module does not have `__all__`, all its members that satisfy all the following criteria will be documented:
    1. The member is public (ie, its name does not begin with an underscore)
    2. The member is defined in the module rather than imported
4. Any member of a documented class that satisfies all the following criteria will be documented:
    1. The member is public (ie, its name does not begin with an underscore)
    2. The member is defined in the class being documented (ie, it is not inherited)

This means it is usually not necessary to define `__all__`, except in cases when you want an imported object to be accessible in the API at a different point to where it is defined, or you want to document a private object. To document an inherited method without changing its behavior, define it in the child class without a docstring:

```python
class foo(bar):
    # Write the full signature to make sure it gets documented
    def method_on_bar(self):
        # Docstring will be inherited from bar.method_on_bar
        super().method_on_bar()
```
