"""
PDB loader that uses the CCD to read most PDB files without guessing bonds
"""

from importlib.metadata import version

from . import ccd, chem, exceptions, residue
from ._pdb import topology_from_pdb
from ._std_ccd_cache import STD_CCD_CACHE as _STD_CCD_CACHE
from .ccd import CcdCache
from .residue import ResidueDefinition

__all__ = [
    "topology_from_pdb",
    "STD_CCD_CACHE",
    "ResidueDefinition",
    "exceptions",
    "ccd",
    "residue",
    "chem",
]

__version__ = version("openff.pablo")

# This reassignment lets us workaround an Autodoc limitation:
# https://github.com/sphinx-doc/sphinx/issues/6495
STD_CCD_CACHE: CcdCache = _STD_CCD_CACHE
"""
The CCD, with commonly-required patches for biomolecular simulation.

``STD_CCD_CACHE`` is Pablo's default residue library.

This module attribute provides the CCD with a dict-like API. When a residue is
requested via the indexing syntax (for example, ``my_ccd_cache["ALA"]``), a
cache is checked first. If the residue is not present and the ``auto_download``
attribute is truthy, the residue is downloaded from the CCD. If the residue
cannot be retrieved from the cache or the CCD, a ``KeyError`` is raised.

Iterating over the cache, checking its length, using the ``in`` operator,
or otherwise treating ``STD_CCD_CACHE`` as a mapping other than with the
indexing syntax works only on the cache itself. As a result, accessing
a residue via indexing may return a value even if these other methods
suggest it won't.

Several common residues are packaged with Pablo and never require an internet
lookup. This includes the canonical amino and nucleic acids, water, several
ions, and a few others. CCD entries that are downloaded are stored in
``$XDG_CACHE_HOME/openff-pablo/ccd_cache`` and are preserved between Python
sessions. Pablo assumes that the files in the cache path were downloaded from
the CCD and may do unexpected things if they are edited by hand, though
deleting them is safe.

``STD_CCD_CACHE`` also includes a number of patches for CCD entries that are
known not to adhere to the high standards required for biomolecular simulation.
New patches can be added with the ``with_patch()`` method.

The ``with_()`` and ``with_replaced`` methods allow custom definitions to be
added to a new copy of ``STD_CCD_CACHE``. These custom definitions are not
patched. Since they are stored in the cache, custom definitions supercede any
that have not already been downloaded from the CCD.

Accessing the CCD requires internet access. Without internet access, only
entries packaged with Pablo or already in the cache can be loaded.

By default, ``STD_CCD_CACHE`` does not automatically access the internet.
Automatic CCD downloads of uncached residues can be enabled by setting
``STD_CCD_CACHE.auto_download = True``, and specific residues can be downloaded
on a case-by-case basis with :py:meth:`openff.pablo.ccd.CcdCache.get_from_ccd`.

For details about the available methods, see the class documentation at
:py:class:`openff.pablo.ccd.CcdCache`
"""
