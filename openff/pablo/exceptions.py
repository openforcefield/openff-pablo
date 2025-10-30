"""
Exceptions for the PDB loader.
"""

__all__ = [
    "PabloError",
    "ResidueValidationError",
    "PdbResidueMatchError",
    "UnknownOrAmbiguousSerialInConectError",
    "MissingAtomError",
]


from collections.abc import Collection


class PabloError(ValueError):
    """A generic Pablo error. Base class of all Pablo errors."""

    pass


class MissingAtomError(PabloError):
    """A required atom is missing from a ``ResidueDefinition``."""

    pass


class ResidueValidationError(PabloError):
    """A newly created ``ResidueDefinition`` is invalid."""

    pass


class PdbResidueMatchError(PabloError):
    """A residue could not be matched to a residue definition while loading a file."""

    pass


class UnknownOrAmbiguousSerialInConectError(PabloError):
    """While parsing a PDB file, a CONECT record has an incorrect atom serial."""

    def __init__(self, serial: str, possible_indices: Collection[int]):
        self.serial = serial
        self.possible_indices = possible_indices
        msg = f"Atom serial {serial} was found in a CONECT record, "
        if len(possible_indices) == 0:
            msg += "but no corresponding ATOM/HETATM record was found"
        else:
            msg += "but multiple corresponding ATOM/HETATM records were found "
            msg += f"(records {','.join(map(str, possible_indices))})"
        super().__init__(msg)
