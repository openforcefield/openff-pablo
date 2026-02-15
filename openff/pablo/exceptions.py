"""
Exceptions for the PDB loader.
"""

__all__ = [
    "PabloError",
    "ResidueValidationError",
    "PdbResidueMatchError",
    "UnknownOrAmbiguousSerialInConectError",
    "MissingAtomError",
    "GemmiError",
]


from collections.abc import Collection


class GemmiError(Exception):
    """Wrapper for exceptions raised by Gemmi"""

    def __init__(self, parent: Exception):
        super().__init__(f"{type(parent)}: {parent}")


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


class AmbiguousMatchError(PabloError):
    """Two matches disagree where they overlap.

    This error is raised if a residue definition can be mapped via
    `additional_definitions` to an otherwise unknown atom or bond in multiple
    chemically distinct ways.
    """

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__(
            f"resdef does not unambiguously map: {''.join(f'\n  {reason}' for reason in reasons)}",
        )
