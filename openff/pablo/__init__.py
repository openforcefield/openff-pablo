"""
PDB loader that uses the CCD to read most PDB files without guessing bonds
"""

from importlib.metadata import version

from . import ccd, chem, exceptions, residue
from ._pdb import topology_from_pdb
from .ccd import CCD_RESIDUE_DEFINITION_CACHE
from .residue import ResidueDefinition

__all__ = [
    "topology_from_pdb",
    "CCD_RESIDUE_DEFINITION_CACHE",
    "ResidueDefinition",
    "exceptions",
    "ccd",
    "residue",
    "chem",
]

__version__ = version("openff.pablo")
