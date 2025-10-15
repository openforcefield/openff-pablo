:::{include} ../README.md
:::

:::{toctree}
---
hidden: true
caption: User's Guide
---

Overview <self>
glossary.md
:::

:::{toctree}
---
hidden: true
caption: The Pablo Book
glob: true
numbered: 3
titlesonly: true
---
book/*-*
:::

:::{toctree}
---
hidden: true
caption: Developer's Guide
---

tests.md
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
