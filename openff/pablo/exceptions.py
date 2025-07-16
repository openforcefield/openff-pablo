"""
Exceptions for the PDB loader.
"""

__all__ = [
    "NoMatchingResidueDefinitionError",
    "MultipleMatchingResidueDefinitionsError",
    "UnknownOrAmbiguousSerialInConectError",
]


from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from openff.toolkit import Molecule

from openff.pablo._utils import flatten

if TYPE_CHECKING:
    from openff.pablo._pdb_data import PdbData, ResidueMatch
    from openff.pablo.residue import ResidueDefinition


class NoMatchingResidueDefinitionError(ValueError):
    """Exception raised when a residue is missing from the database"""

    def __init__(
        self,
        res_atom_idcs: Sequence[int],
        data: "PdbData",
        unknown_molecules: Iterable[Molecule],
        additional_substructures: Iterable["ResidueDefinition"],
        residue_database: Mapping[
            str,
            Iterable["ResidueDefinition"],
        ],
        verbose_errors: bool = False,
    ):
        i = res_atom_idcs[0]
        res_name = data.res_name[i]

        line_nos = sorted(
            [data.line_no[j] for j in res_atom_idcs],
            key=lambda x: 0 if x is None else x,
        )
        first, last = line_nos[0], line_nos[-1]
        if (
            first is not None
            and last is not None
            and line_nos == list(range(first, last + 1))
        ):
            line_no = f"{first}-{last}"
        else:
            line_no = first

        msg = [
            (
                "No residue definitions covered all atoms in residue"
                + f"{data.chain_id[i]}:{res_name}#{data.res_seq[i]}"
                + (
                    f" ({data.src_filename}:l{line_no})"
                    if data.src_filename is not None
                    else f" (l{line_no})"
                )
            ),
        ]
        if verbose_errors:
            residue_definitions = list(residue_database[res_name])
            if len(residue_definitions) == 0:
                msg.append("  No residues in database for residue name {res_name}")
            found_names = {data.name[i] for i in res_atom_idcs}
            for resdef in residue_definitions:
                extra_names = found_names.difference(
                    flatten([atom.name, *atom.synonyms] for atom in resdef.atoms),
                )
                missing_names = {
                    "|".join([atom.name, *atom.synonyms])
                    for atom in resdef.atoms
                    if atom.name not in found_names
                    and all([synonym not in found_names for synonym in atom.synonyms])
                    and not atom.leaving
                }
                missing_leavers = {
                    "|".join([atom.name, *atom.synonyms])
                    for atom in resdef.atoms
                    if atom.name not in found_names
                    and all([synonym not in found_names for synonym in atom.synonyms])
                    and atom.leaving
                }
                msg.append(f"    In {resdef.description}:")
                msg.append(f"      {sorted(extra_names)} were found but not expected")
                msg.append(f"      {sorted(missing_names)} were expected but not found")
                msg.append(
                    f"      Leaving atoms {sorted(missing_leavers)} were also not found",
                )

        super().__init__("\n".join(msg))


class MultipleMatchingResidueDefinitionsError(ValueError):
    def __init__(
        self,
        matches: Sequence["ResidueMatch"],
        res_atom_idcs: tuple[int, ...],
        data: "PdbData",
        verbose_errors: bool,
    ):
        i = res_atom_idcs[0]

        msg = (
            f"{len(matches)} residue definitions matched residue "
            + f"{data.chain_id[i]}:{data.res_name[i]}#{data.res_seq[i]}"
        )

        if verbose_errors:
            msg += ":"
            msg = "\n    ".join(
                [
                    msg,
                    *map(
                        lambda m: m.residue_definition.description
                        + f" {m.expect_crosslink=} {m.expect_prior_bond=} {m.expect_posterior_bond=}",
                        matches,
                    ),
                ],
            )

        super().__init__(msg)


class UnknownOrAmbiguousSerialInConectError(ValueError):
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
