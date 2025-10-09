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
    NoResidueDefinitions,
    ResidueMatch,
    ResidueMismatch,
    SuccessfulMatch,
    only_matched,
)
from openff.pablo._utils import flatten
from openff.pablo.residue import ResidueDefinition

if TYPE_CHECKING:
    from openff.pablo._pdb_data import PdbData


def _format_linenos(data: "PdbData", res_atom_idcs: Sequence[int]) -> str:
    line_nos = sorted(
        [data.line_no[j] for j in res_atom_idcs if data.line_no[j] != -1],
    )
    if len(line_nos) == 0:
        return "l?"
    first, last = line_nos[0], line_nos[-1]
    if line_nos == list(range(first, last + 1)):
        return f"l{first}-{last}"
    else:
        return f"l{first}"


class PdbResidueMatchError(ValueError):
    def __init__(
        self,
        data: "PdbData",
        errors: list[list[MismatchProtocol] | list[SuccessfulMatch]],
    ):
        msg: list[str] = [
            "some residues could not be identified",
            "A topology cannot be created without chemical information for every",
            (
                f"atom and bond. The following residues present in PDB file\n{data.src_filename}"
                if data.src_filename is not None
                else "atom and bond. The following residues present in the PDB file"
            ),
            "could not be identified from the provided chemical library:",
        ]

        for residue_errors in errors:
            if len(residue_errors) == 0:
                continue

            prototype_residue_error = residue_errors[0]
            i = prototype_residue_error.prototype_index
            src = _format_linenos(data, prototype_residue_error.res_atom_idcs)
            resid = f"{data.chain_id[i]}:{data.res_name[i]}#{data.res_seq[i]}"

            expects_crosslinks = [
                err.expects_crosslink for err in only_matched(residue_errors)
            ]
            expects_posteriors = [
                err.expects_posterior_bond for err in only_matched(residue_errors)
            ]
            expects_priors = [
                err.expects_prior_bond for err in only_matched(residue_errors)
            ]

            residue_mismatch_resdef_description_lens = [
                len(match.residue_definition.description)
                for match in residue_errors
                if isinstance(match.residue_definition, ResidueDefinition)
            ]
            max_padding = (
                max(residue_mismatch_resdef_description_lens)
                if residue_mismatch_resdef_description_lens
                else 0
            )

            if isinstance(prototype_residue_error, MatchProtocol):
                msg.append(
                    f"  {resid} ({src}): Multiple conflicting residue definition matches:",
                )
                consensus_expects_start = (
                    f"    │{' ' * (max_padding - 10)} Every match expects "
                )
                if all(expects_crosslinks):
                    msg.append(consensus_expects_start + "crosslink")
                elif not any(expects_crosslinks):
                    msg.append(consensus_expects_start + "no crosslink")
                if all(expects_priors):
                    msg.append(consensus_expects_start + "prior bond")
                elif not any(expects_priors):
                    msg.append(consensus_expects_start + "no prior bond")
                if all(expects_posteriors):
                    msg.append(consensus_expects_start + "posterior bond")
                elif not any(expects_posteriors):
                    msg.append(consensus_expects_start + "no posterior bond")
            elif (
                isinstance(prototype_residue_error, NoResidueDefinitions)
                and len(residue_errors) == 1
            ):
                msg.append(f"  {resid} ({src}): No residue definitions")
            else:
                msg.append(f"  {resid} ({src}): No matching residue definitions:")

            for err in sorted(
                residue_errors,
                key=lambda x: (str(type(x)), x.sort_key()),
            ):
                if isinstance(err, NoResidueDefinitions):
                    continue
                elif isinstance(err, ResidueMatch):
                    desc = err.residue_definition.description
                    padding = max_padding - len(desc)
                    expects = ", ".join(
                        flatten(
                            [
                                ("crosslink",)
                                if err.expects_crosslink
                                and not all(expects_crosslinks)
                                and any(expects_crosslinks)
                                else (),
                                ("posterior bond",)
                                if err.expects_posterior_bond
                                and not all(expects_posteriors)
                                and any(expects_posteriors)
                                else (),
                                ("prior bond",)
                                if err.expects_prior_bond
                                and not all(expects_priors)
                                and any(expects_priors)
                                else (),
                            ],
                        ),
                    )
                    if expects == "":
                        expects = "no other linkages"
                    msg.append(
                        f"    ├{'─' * padding} {desc}: expects {expects}",
                    )
                elif isinstance(err, ResidueMismatch):
                    desc = err.residue_definition.description
                    padding = max_padding - len(desc)
                    msg.append(
                        f"    ├{'─' * padding} {desc} "
                        + f"failed to match: {err.reason}",
                    )
                else:
                    msg.append(f"    ├ {err.description}")

            if msg[-1].startswith("    ├"):
                msg[-1] = "    ╰" + msg[-1][5:]

            msg.append("")

        if msg[-1] == "":
            msg = msg[:-1]

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
