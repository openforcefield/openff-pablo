"""
Exceptions for the PDB loader.

This is a private module for internal use. It re-exports the exceptions in
:py:module:`openff.pablo.exceptions`, and also provides creation functions for
exceptions that can be consumed publically but are produced privately.
"""

__all__ = [
    "PabloError",
    "ResidueValidationError",
    "PdbResidueMatchError",
    "UnknownOrAmbiguousSerialInConectError",
    "MissingAtomError",
    "create_pdb_residue_match_error",
]


import itertools
from collections.abc import Iterable, Sequence
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
from openff.pablo.exceptions import (
    MissingAtomError,
    PabloError,
    PdbResidueMatchError,
    ResidueValidationError,
    UnknownOrAmbiguousSerialInConectError,
)
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


def create_pdb_residue_match_error(
    data: "PdbData",
    errors: list[list[MismatchProtocol] | list[SuccessfulMatch]],
    additional_definitions: Sequence[ResidueDefinition] = (),
    additional_matches: Sequence[SuccessfulMatch] | None = None,
    unmatched_pdb_idcs: Iterable[int] = (),
) -> PdbResidueMatchError:
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

        for err in sorted(residue_errors, key=lambda x: str(type(x))):
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
                    f"    ├{'─' * padding} {desc} " + f"failed to match: {err.reason}",
                )
            else:
                msg.append(f"    ├ {err.description}")

        if msg[-1].startswith("    ├"):
            msg[-1] = "    ╰" + msg[-1][5:]

        msg.append("")

    if msg[-1] == "":
        msg = msg[:-1]

    unmatched_atoms = [
        f"{data.chain_id[i]}:{data.res_name[i]}{data.res_seq[i]}.{data.name[i]} (l{data.line_no[i]})"
        for i in unmatched_pdb_idcs
    ]
    if len(additional_definitions) > 0:
        if additional_matches is None:
            msg.extend(
                [
                    "",
                    "additional_definitions were not consulted because they",
                    "cannot resolve the above ambiguity.",
                ],
            )
        elif len(additional_matches) == 0:
            msg.extend(
                [
                    "",
                    "Also, no graph-based matches could be found amongst",
                    "the additional_definitions",
                ],
            )
        elif len(unmatched_atoms) > 0:
            msg.extend(
                [
                    "",
                    "Also, the following additional_definitions could be",
                    "matched to unknown atoms, but they did not cover all",
                    "atoms that were left unknown and so some atoms were",
                    "left without chemical information:",
                ],
            )
            for match in additional_matches:
                msg.append(f"    {match.description}")

            if len(unmatched_atoms) < 100:
                atom_len = max(len(s) for s in unmatched_atoms)
                batch_size = (80) // (atom_len + 2)
                msg.extend(
                    [
                        "The following atoms were left unidentified:",
                        *(
                            "  "
                            + "  ".join(
                                f"{{:<{atom_len}}}".format(
                                    atom.replace(
                                        " ",
                                        " " * (atom_len - len(atom) + 1),
                                    ),
                                )
                                for atom in atoms
                            )  # nofmt
                            for atoms in itertools.batched(
                                unmatched_atoms,
                                batch_size,
                            )
                        ),
                    ],
                )
        else:
            msg.extend(
                [
                    "",
                    "Also, the following additional_definitions could be",
                    "matched to unknown atoms, but they did not cover all",
                    "bonds that were left unknown and so some bonds were",
                    "left without chemical information:",
                ],
            )
            for match in additional_matches:
                msg.append(f"    {match.description}")

    return PdbResidueMatchError("\n".join(msg))
