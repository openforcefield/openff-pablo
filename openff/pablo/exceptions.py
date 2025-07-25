"""
Exceptions for the PDB loader.
"""

__all__ = [
    "PdbResidueMatchError",
    "UnknownOrAmbiguousSerialInConectError",
]


from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING

from openff.pablo._matching import (
    MatchProtocol,
    MismatchProtocol,
    MoleculeMatch,
    NoResidueDefinitions,
    ResidueMatch,
    ResidueMismatch,
)
from openff.pablo.residue import ResidueDefinition

if TYPE_CHECKING:
    from openff.pablo._pdb_data import PdbData


def _format_src(data: "PdbData", res_atom_idcs: Sequence[int]) -> str:
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
        line_no = f"{first}"

    return (
        f"{data.src_filename}:l{line_no}"
        if data.src_filename is not None
        else f"l{line_no}"
    )


class PdbResidueMatchError(ValueError):
    def __init__(
        self,
        data: "PdbData",
        errors: list[list[MismatchProtocol] | list[MatchProtocol]],
    ):
        msg = [
            "some residues could not be identified",
            "A topology cannot be created without chemical information for every",
            "atom and bond. The following residues present in the PDB file could",
            "not be identified from the provided chemical library:",
        ]

        for residue_errors in errors:
            if len(residue_errors) == 0:
                continue

            prototype_residue_error = residue_errors[0]
            i = prototype_residue_error.prototype_index
            src = _format_src(data, prototype_residue_error.res_atom_idcs)
            resid = f"{data.chain_id[i]}:{data.res_name[i]}#{data.res_seq[i]}"

            if isinstance(prototype_residue_error, MatchProtocol):
                msg.append(
                    f"  {resid} ({src}): Multiple conflicting residue definition matches:",
                )
            elif (
                isinstance(prototype_residue_error, NoResidueDefinitions)
                and len(residue_errors) == 1
            ):
                msg.append(f"  {resid} ({src}): No residue definitions")
            else:
                msg.append(f"  {resid} ({src}): No matching residue definitions:")

            residue_mismatch_resdef_description_lens = [
                len(match.residue_definition.description)
                for match in residue_errors
                if isinstance(match.residue_definition, ResidueDefinition)
            ]
            residue_mismatch_resdef_description_max_length = (
                max(residue_mismatch_resdef_description_lens)
                if residue_mismatch_resdef_description_lens
                else 0
            )
            for err in sorted(residue_errors, key=lambda x: (type(x), x.sort_key())):
                if isinstance(err, NoResidueDefinitions):
                    continue
                elif isinstance(err, ResidueMatch):
                    msg.append(
                        f"    ├ {err.residue_definition.description}: {err.expects_crosslink=} {err.expects_prior_bond=} {err.expects_posterior_bond=}",
                    )
                elif isinstance(err, MoleculeMatch):
                    msg.append(
                        f"    ├ Unknown molecule {err.description}",
                    )
                elif isinstance(err, ResidueMismatch):
                    desc = err.residue_definition.description
                    padding = residue_mismatch_resdef_description_max_length - len(desc)
                    msg.append(
                        f"    ├{'─' * padding} {desc} "
                        + f"failed to match: {err.reason}",
                    )
                else:
                    msg.append(f"    ├ {err.description}")

            if msg[-1].startswith("    ├"):
                msg[-1] = "    ╰" + msg[-1][5:]

        return super().__init__("\n".join(msg))


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
