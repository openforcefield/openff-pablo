# OpenFF Pablo

New, independent implementation of `Topology.from_pdb`

## Installing OpenFF Pablo

OpenFF recommends using Conda virtual environments for all scientific Python work. Pablo can be installed into a new Conda environment named `pablo` with the [`openff-pablo`] package:

```shell-session
$ mamba create -n pablo -c conda-forge openff-pablo
$ mamba activate pablo
```

If you do not have Conda or Mamba installed, see the [OpenFF installation documentation](inv:openff.docs#install).

[`openff-pablo`]: https://anaconda.org/conda-forge/openff-pablo

:::{toctree}
---
hidden: true
---

Overview <self>
tests.md
loading.md
:::

<!--
The autosummary directive renders to rST,
so we must use eval-rst here
-->
```{eval-rst}
.. raw:: html

    <div style="display: None">

.. autosummary::
   :recursive:
   :caption: API Reference
   :toctree: api/generated
   :nosignatures:

   openff.pablo

.. raw:: html

    </div>
```
