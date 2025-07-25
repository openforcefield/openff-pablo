import dataclasses
import functools
import itertools
import logging
import warnings
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from io import TextIOBase
from os import PathLike
from typing import IO, Any, DefaultDict, Protocol, Self

from openff.toolkit import Molecule
from openff.units import elements

from ._matching import (
    MismatchProtocol,
    MoleculeMatch,
    NoResidueDefinitions,
    PossibleResidueMatch,
    ResidueMatch,
    ResidueMismatch,
    SuccessfulMatch,
    only_matched,
)
from ._utils import (
    __UNSET__,
    charge_int_or_none,
    dec_hex,
    flatten,
    sort_tuple,
    unwrap,
    with_neighbours,
)
from .exceptions import (
    PdbResidueMatchError,
    UnknownOrAmbiguousSerialInConectError,
)
from .residue import BondDefinition, ResidueDefinition

__all__ = [
    "PdbData",
]


@dataclass
class PdbData:
    src_filename: str | None = None
    line_no: list[int | None] = field(default_factory=list)
    model: list[int | None] = field(default_factory=list)
    serial: list[str] = field(default_factory=list)
    name: list[str] = field(default_factory=list)
    alt_loc: list[str] = field(default_factory=list)
    res_name: list[str] = field(default_factory=list)
    chain_id: list[str] = field(default_factory=list)
    res_seq: list[str] = field(default_factory=list)
    i_code: list[str] = field(default_factory=list)
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)
    occupancy: list[float] = field(default_factory=list)
    temp_factor: list[float] = field(default_factory=list)
    element: list[str] = field(default_factory=list)
    charge: list[int | None] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    res_idx: list[int] | None = None
    serial_to_index: DefaultDict[str, list[int]] = field(
        default_factory=lambda: defaultdict(list),
    )
    conects: list[set[int]] = field(default_factory=list)
    """The ith set contains atom indices CONECTed to atom index i"""
    cryst1_a: float | None = None
    cryst1_b: float | None = None
    cryst1_c: float | None = None
    cryst1_alpha: float | None = None
    cryst1_beta: float | None = None
    cryst1_gamma: float | None = None
    strict: bool = False
    """True to read in strict, spec-compliant mode, False to use common extensions.

    Extensions include:
    - hexadecimal extensions for atom serials and residue sequence numbers that
    do not fit in the fixed column format
    - missing charge column indicates unknown charge, not 0
    - residue name may extend into column 21 iff the reside name has 4
    non-whitespace characters"""

    @classmethod
    def from_file(cls, path: str | PathLike[str]) -> Self:
        with open(path) as f:
            ret = cls.from_file_object(f)
        ret.src_filename = str(path)
        return ret

    @classmethod
    def from_file_object(cls, file: IO[str] | TextIOBase) -> Self:
        return cls.parse_pdb(file.readlines())

    def _append_coord_line(self, line: str):
        for field_ in dataclasses.fields(self):
            value = getattr(self, field_.name)
            if hasattr(value, "append"):
                value.append(__UNSET__)
                assert value[-1] is __UNSET__

        self.line_no[-1] = None
        self.model[-1] = None
        self.serial[-1] = line[6:11].strip()
        self.serial_to_index[self.serial[-1]].append(len(self.serial) - 1)
        self.name[-1] = line[12:16].strip()
        self.alt_loc[-1] = line[16].strip() or ""
        self.res_name[-1] = (
            line[17:20].strip()
            if self.strict or any(char.isspace() for char in line[17:21])
            else line[17:21]
        )
        self.chain_id[-1] = line[21].strip()
        self.res_seq[-1] = line[22:26].strip()
        self.i_code[-1] = line[26].strip() or " "
        self.x[-1] = float(line[30:38])
        self.y[-1] = float(line[38:46])
        self.z[-1] = float(line[46:54])
        self.occupancy[-1] = float(line[54:60])
        self.temp_factor[-1] = float(line[60:66])
        self.element[-1] = line[76:78].strip()
        self.charge[-1] = charge_int_or_none(line[78:80].strip(), strict=self.strict)
        self.terminated[-1] = False
        self.conects[-1] = set()

        # Ensure we've assigned a value to every field
        for field_ in dataclasses.fields(self):
            value = getattr(self, field_.name)
            if hasattr(value, "append"):
                assert value[-1] is not __UNSET__

    @classmethod
    def parse_pdb(cls, lines: Iterable[str], strict: bool = False) -> Self:
        model_n = None
        data = cls(strict=strict)
        for i, line in enumerate(lines):
            if line.startswith("MODEL "):
                model_n = int(line[10:14])
            if line.startswith("ENDMDL "):
                model_n = None
            if line.startswith("HETATM") or line.startswith("ATOM  "):
                data._append_coord_line(line)
                data.line_no[-1] = i + 1
                data.model[-1] = model_n
            if line.startswith("TER   "):
                terminated_resname = data.res_name[-1]
                terminated_chainid = data.chain_id[-1]
                terminated_resseq = data.res_seq[-1]
                terminated_icode = data.i_code[-1]
                for i in range(len(data.res_name) - 1, 0, -1):
                    if (
                        data.res_name[i] == terminated_resname
                        and data.chain_id[i] == terminated_chainid
                        and data.res_seq[i] == terminated_resseq
                        and data.i_code[i] == terminated_icode
                    ):
                        data.terminated[i] = True
                    else:
                        break
            if line.startswith("CRYST1"):
                data.cryst1_a = float(line[6:15])
                data.cryst1_b = float(line[15:24])
                data.cryst1_c = float(line[24:33])
                data.cryst1_alpha = float(line[33:40])
                data.cryst1_beta = float(line[40:47])
                data.cryst1_gamma = float(line[47:54])

        # Read all CONECT records
        data.conects = cls._process_conects(
            lines,
            data.serial_to_index,
            data.conects,
            data.model,
        )

        return data

    @staticmethod
    def _process_conects(
        lines: Iterable[str],
        serial_to_index: dict[str, list[int]],
        conects: list[set[int]],
        model: Sequence[int | None],
    ) -> list[set[int]]:
        for line in lines:
            if line.startswith("CONECT "):
                # a is the serial of the first atom in the conect, we need its indices
                a = line[6:11].strip()
                a_idcs = serial_to_index.get(a, [])

                # Conects are usually provided once for multi-model files
                # Raise an error if there are multiple indices for the serial
                # within a single model, as the bond is ambiguous
                a_models = {model[i] for i in a_idcs}
                if len(a_models) != len(a_idcs):
                    raise UnknownOrAmbiguousSerialInConectError(a, a_idcs)

                for a_idx, a_model in zip(a_idcs, a_models):
                    for start, stop in [(11, 16), (16, 21), (21, 26), (26, 31)]:
                        if len(line.strip()) <= start:
                            continue

                        b = line[start:stop].strip()

                        b_idcs = [
                            i for i in serial_to_index.get(b, []) if model[i] == a_model
                        ]
                        if len(b_idcs) != 1:
                            raise UnknownOrAmbiguousSerialInConectError(b, b_idcs)
                        b_idx = b_idcs[0]

                        conects[a_idx].add(b_idx)
                        conects[b_idx].add(a_idx)
        return conects

    @property
    def residue_indices(self) -> Iterator[tuple[int, ...]]:
        indices = []
        prev = None
        res_idx = 0
        self.res_idx = []
        for atom_idx, (alt_loc, *residue_info) in enumerate(
            zip(
                self.alt_loc,
                self.model,
                self.res_name,
                self.chain_id,
                self.res_seq,
                self.i_code,
            ),
        ):
            if alt_loc != "":
                # TODO: Support alt-locs as alternate conformers
                warnings.warn(
                    "Alt locs not supported; only empty or 'A' alt locs will be read",
                )
                if alt_loc != "A":
                    continue
            if prev is not None and residue_info[0] != prev[0]:
                # TODO: Support multi-model files
                warnings.warn(
                    "Multi-model files not supported; topology will reflect first model",
                )
                break
            if prev == residue_info or prev is None:
                indices.append(atom_idx)
            else:
                yield tuple(indices)
                res_idx += 1
                indices = [atom_idx]
            self.res_idx.append(res_idx)
            prev = residue_info

        yield tuple(indices)

    def subset_matches_residue(
        self,
        res_atom_idcs: Collection[int],
        residue_definition: ResidueDefinition,
    ) -> PossibleResidueMatch:
        # Raise an error if the match would be empty - this way the
        # return value's truthiness always reflects whether there was a match
        if len(res_atom_idcs) == 0:
            raise ValueError("cannot match empty res_atom_idcs")

        logging.debug(f"  Attempting match against {residue_definition.description}")

        # Skip definitions with too few atoms
        if len(residue_definition.atoms) < len(res_atom_idcs):
            reason = f"Too few atoms in residue definition ({len(residue_definition.atoms)} < {len(res_atom_idcs)})"
            logging.debug("    Match failed: " + reason)
            return ResidueMismatch(
                residue_definition=residue_definition,
                index_to_atomdef={i: None for i in res_atom_idcs},
                reason=reason,
            )

        # Skip non-(cross)linking definitions with the wrong number of atoms
        if (
            residue_definition.linking_bond is None
            and residue_definition.crosslink is None
            and len(
                residue_definition.atoms,
            )
            != len(res_atom_idcs)
        ):
            reason = "No links and wrong number of atoms"
            logging.debug("    Match failed: " + reason)
            return ResidueMismatch(
                residue_definition=residue_definition,
                index_to_atomdef={i: None for i in res_atom_idcs},
                reason=reason,
            )

        # Get the map from the canonical names to the indices
        try:
            index_to_atomdef = {
                i: residue_definition.name_to_atom[self.name[i]] for i in res_atom_idcs
            }
        except KeyError:
            reason = "Name missing from residue definition"
            logging.debug("    Match failed: " + reason)
            return ResidueMismatch(
                residue_definition=residue_definition,
                index_to_atomdef={i: None for i in res_atom_idcs},
                reason=reason,
            )

        match = ResidueMatch(
            index_to_atomdef=index_to_atomdef,
            residue_definition=residue_definition,
        )

        matched_atoms = {atom.name for atom in index_to_atomdef.values()}

        # Fail to match if any atoms in PDB file got matched to more than one name
        if len(matched_atoms) != len(res_atom_idcs):
            reason = "Atom definition matched multiple PDB coordinate records"
            logging.debug("    Match failed: " + reason)
            return match.reject(reason)

        # This assert should be guaranteed by the above
        assert set(index_to_atomdef.keys()) == set(res_atom_idcs)

        # Check that elements match, but tolerate missing columns and wrong case
        if any(
            self.element[i] != "" and self.element[i].lower() != atom.symbol.lower()
            for i, atom in index_to_atomdef.items()
        ):
            reason = "Element mismatch"
            logging.debug("    Match failed: " + reason)
            return match.reject(reason)

        # PDB files can not be trusted; ignore charges.
        # # Check that charges match, but tolerate missing columns
        # if any(
        #     self.charge[i] is not None and self.charge[i] != atom.charge
        #     for i, atom in index_to_atomdef.items()
        # ):
        #     logging.debug("    Match failed: Charge mismatch")
        #     return None

        missing_atoms = [
            atom for atom in residue_definition.atoms if atom.name not in matched_atoms
        ]

        # Match only if all the leaving atoms associated with each linking atom
        # is either entirely present or entirely absent
        missing_atom_names = {atom.name for atom in missing_atoms}
        if any(not atom.leaving for atom in missing_atoms):
            reason = "Missing atom is not leaving atom"
            logging.debug("    Match failed: " + reason)
            return match.reject(reason)
        elif (
            (
                missing_atom_names.issuperset(
                    residue_definition.prior_bond_leaving_atoms,
                )
                or missing_atom_names.isdisjoint(
                    residue_definition.prior_bond_leaving_atoms,
                )
            )
            and (
                missing_atom_names.issuperset(
                    residue_definition.posterior_bond_leaving_atoms,
                )
                or missing_atom_names.isdisjoint(
                    residue_definition.posterior_bond_leaving_atoms,
                )
            )
            and (
                missing_atom_names.issuperset(
                    residue_definition.crosslink_leaving_atoms,
                )
                or missing_atom_names.isdisjoint(
                    residue_definition.crosslink_leaving_atoms,
                )
            )
        ):
            logging.debug("    Match succeeded!")
            return match
        else:
            reason = "Missing atoms do not specify link"
            logging.debug("    Match failed: " + reason)
            return match.reject(reason)

    @cached_property
    def atom_idx_to_res_idx(self) -> dict[int, int]:
        value: dict[int, int] = {}
        for res_idx, atom_indices in enumerate(self.residue_indices):
            for atom_idx in atom_indices:
                value[atom_idx] = res_idx
        return value

    def get_name_based_matches(
        self,
        residue_database: Mapping[str, Iterable[ResidueDefinition]],
    ) -> Iterator[list[PossibleResidueMatch]]:
        for res_atom_idcs in self.residue_indices:
            prototype_index = res_atom_idcs[0]
            res_name = self.res_name[prototype_index]
            logging.debug(f"Beginning name-based match of {res_name} {res_atom_idcs}")
            if len(res_atom_idcs) <= 3:
                logging.debug(
                    f"  Atom names are ({', '.join(self.name[i] for i in res_atom_idcs)})",
                )

            matches = [
                self.subset_matches_residue(
                    res_atom_idcs,
                    residue_definition,
                )
                for residue_definition in residue_database.get(res_name, [])
            ]
            if len(matches) > 0:
                yield matches
            else:
                yield [
                    NoResidueDefinitions(
                        residue_definition=res_name,
                        index_to_atomdef={i: None for i in res_atom_idcs},
                    ),
                ]

    def filter_on_polymer_linkages(
        self,
        this_matches: Sequence[PossibleResidueMatch],
        prev_matches: Sequence[PossibleResidueMatch],
        next_matches: Sequence[PossibleResidueMatch],
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
    ) -> Iterator[PossibleResidueMatch]:
        if len(list(only_matched(this_matches))) != 0:
            logging.debug(
                f"Beginning link-based match of {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
            )
        else:
            yield from this_matches
            return
        # neighbour_supported_posterior_bonds and
        # neighbour_supported_prior_bonds are maps from possible linking
        # bonds in this residue to the atom index (in the neighbouring
        # residue) where the linking atom can be found
        neighbour_supported_posterior_bonds: dict[BondDefinition, int] = {
            next_match.residue_definition.linking_bond: next_match.canonical_atom_name_to_index[
                next_match.residue_definition.prior_bond_linking_atom
            ]
            for next_match in only_matched(next_matches)
            if isinstance(next_match, ResidueMatch)
            and next_match.residue_definition.linking_bond is not None
            and next_match.expects_prior_bond
        }
        neighbour_supported_prior_bonds: dict[BondDefinition, int] = {
            prev_match.residue_definition.linking_bond: prev_match.canonical_atom_name_to_index[
                prev_match.residue_definition.posterior_bond_linking_atom
            ]
            for prev_match in only_matched(prev_matches)
            if isinstance(prev_match, ResidueMatch)
            and prev_match.residue_definition.linking_bond is not None
            and prev_match.expects_posterior_bond
        }

        valid_next_matches = list(only_matched(next_matches))
        neighbours_support_molecule_end = (
            any(not next_match.expects_prior_bond for next_match in valid_next_matches)
            or len(valid_next_matches) == 0
        )
        valid_prev_matches = list(only_matched(prev_matches))
        neighbours_support_molecule_start = (
            any(
                not prev_match.expects_posterior_bond
                for prev_match in valid_prev_matches
            )
            or len(valid_prev_matches) == 0
        )
        prev_residue_terminated = (
            len(prev_matches) > 0 and self.terminated[prev_matches[0].prototype_index]
        )
        this_residue_terminated = self.terminated[this_matches[0].prototype_index]
        if len(list(only_matched(this_matches))) != 0:
            logging.debug(
                f"  {neighbour_supported_posterior_bonds=}",
            )
            logging.debug(
                f"  {neighbour_supported_prior_bonds=}",
            )
            logging.debug(
                f"  {neighbours_support_molecule_end=}",
            )
            logging.debug(
                f"  {neighbours_support_molecule_start=}",
            )
        for match in this_matches:
            if not isinstance(match, ResidueMatch):
                yield match
                continue
            logging.debug(
                f"  Attempting link-based match against {match.residue_definition.description}",
            )
            if match.expects_prior_bond:
                if (
                    match.residue_definition.linking_bond
                    not in neighbour_supported_prior_bonds
                ):
                    reason = "Prior bond expected but not supported by neighbours"
                    logging.debug(f"    Match failed: {reason}")
                    yield match.reject(reason)
                    continue
                elif prev_residue_terminated:
                    reason = "Prior bond expected but cannot form polymer bond across TER record"
                    logging.debug(f"    Match failed: {reason}")
                    yield match.reject(reason)
                    continue
                else:
                    match.set_prior_bond(neighbour_supported_prior_bonds)
            elif not neighbours_support_molecule_start:
                reason = "Prior bond not permitted but required by neighbours"
                logging.debug(f"    Match failed: {reason}")
                yield match.reject(reason)
                continue

            if match.expects_posterior_bond:
                if (
                    match.residue_definition.linking_bond
                    not in neighbour_supported_posterior_bonds
                ):
                    reason = "Posterior bond expected but not supported by neighbours"
                    logging.debug(f"    Match failed: {reason}")
                    yield match.reject(reason)
                    continue
                elif this_residue_terminated:
                    reason = "Posterior bond expected but cannot form polymer bond across TER record"
                    logging.debug(f"    Match failed: {reason}")
                    yield match.reject(reason)
                    continue
                else:
                    match.set_posterior_bond(
                        neighbour_supported_posterior_bonds,
                    )
            elif not neighbours_support_molecule_end:
                reason = "Posterior bond not expected but required by neighbours"
                logging.debug(f"    Match failed: {reason}")
                yield match.reject(reason)
                continue

            logging.debug("    Accepted")
            yield match

    def filter_on_crosslinks(
        self,
        this_matches: Sequence[PossibleResidueMatch],
        prev_matches: Sequence[PossibleResidueMatch],
        next_matches: Sequence[PossibleResidueMatch],
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
    ) -> Iterator[PossibleResidueMatch]:
        # Check for crosslinks
        # TODO: This could be simplified if we required crosslinking atoms not to have synonyms
        if len(list(only_matched(this_matches))) != 0:
            logging.debug(
                f"Assigning crosslinks for {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
            )
        else:
            yield from this_matches
            return
        for match in this_matches:
            if not isinstance(match, ResidueMatch):
                yield match
                continue
            logging.debug(
                f"  Attempting crosslink-based match against {match.residue_definition.description}",
            )
            if match.crosslink_idcs is not None:
                # This match's crosslink has already been assigned
                logging.debug(
                    "    Skipping (crosslink already assigned)",
                )
                yield match
                continue
            if not match.expects_crosslink:
                logging.debug(
                    "    Skipping (crosslink not expected)",
                )
                yield match
                continue
            this_crosslink_def = match.residue_definition.crosslink
            if this_crosslink_def is None:
                # No crosslink defined for this match
                logging.debug(
                    "    Skipping (crosslink not expected 2: electric boogaloo)",
                )
                yield match
                continue
            this_crosslink_atom_idx = match.canonical_atom_name_to_index[
                this_crosslink_def.atom1
            ]

            this_crosslink_conects = self.conects[this_crosslink_atom_idx]
            for other_crosslink_atom_idx in this_crosslink_conects:
                logging.debug(
                    f"    checking possible partner {other_crosslink_atom_idx}",
                )
                other_crosslink_res_idx = self.atom_idx_to_res_idx[
                    other_crosslink_atom_idx
                ]
                other_matches = list(all_matches[other_crosslink_res_idx])
                for other_match in other_matches:
                    if not isinstance(other_match, ResidueMatch):
                        continue
                    logging.debug(
                        f"      in {other_match.residue_definition.description}",
                    )
                    other_crosslink_def = other_match.residue_definition.crosslink
                    other_crosslink_atom_canonical_name = other_match.atom(
                        other_crosslink_atom_idx,
                    ).name
                    if (
                        other_match.expects_crosslink
                        and other_crosslink_def is not None
                        and other_crosslink_def.flipped() == this_crosslink_def
                        and other_crosslink_def.atom2
                        == other_crosslink_atom_canonical_name
                    ):
                        logging.debug(
                            "        crosslink found!",
                        )
                        # We've found a crosslink!
                        # TODO: What if there are multiple possible crosslinks?
                        #       ATM the last one is assigned, then rejected
                        #       because the other CONECT records are not
                        #       satisfied
                        match.set_crosslink(
                            this_crosslink_atom_idx,
                            other_crosslink_atom_idx,
                        )
                        other_match.set_crosslink(
                            other_crosslink_atom_idx,
                            this_crosslink_atom_idx,
                        )
            if match.expects_crosslink and match.crosslink_idcs is None:
                yield match.reject(
                    "crosslink expected but no matching crosslink partner could be found",
                )
            else:
                yield match

    def match_additional_substructures(
        self,
        this_matches: Sequence[PossibleResidueMatch],
        prev_matches: Sequence[PossibleResidueMatch],
        next_matches: Sequence[PossibleResidueMatch],
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
        additional_substructures: Iterable[ResidueDefinition],
    ) -> Iterator[PossibleResidueMatch]:
        yield from this_matches
        if not any(this_matches):
            logging.debug(
                f"Matching additional_substructures to {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
            )
            res_atom_idcs = this_matches[0].res_atom_idcs
            for residue_definition in additional_substructures:
                yield self.subset_matches_residue(
                    res_atom_idcs,
                    residue_definition,
                )
            logging.debug(
                "  Done",
            )

    def filter_on_conect_records(
        self,
        this_matches: Sequence[PossibleResidueMatch],
        prev_matches: Sequence[PossibleResidueMatch],
        next_matches: Sequence[PossibleResidueMatch],
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
    ) -> Iterator[PossibleResidueMatch]:
        logging.debug(
            f"Filtering matches on CONECT records for {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
        )
        for match in this_matches:
            if not isinstance(match, ResidueMatch):
                yield match
                continue

            logging.debug(
                f"  Checking match {match.description}",
            )

            expected_bonds: set[tuple[int, int]] = set()
            for bond in match.residue_definition.bonds:
                try:
                    atom1_idx = match.canonical_atom_name_to_index[bond.atom1]
                    atom2_idx = match.canonical_atom_name_to_index[bond.atom2]
                except KeyError:
                    # Bond is for a missing leaving atom
                    continue
                expected_bonds.add(sort_tuple((atom1_idx, atom2_idx)))
            if match.crosslink_idcs is not None:
                expected_bonds.add(sort_tuple(match.crosslink_idcs))
            if match.prior_bond_idcs is not None:
                expected_bonds.add(sort_tuple(match.prior_bond_idcs))
            if match.posterior_bond_idcs is not None:
                expected_bonds.add(sort_tuple(match.posterior_bond_idcs))

            found_conects = set(
                flatten(
                    (sort_tuple((i, j)) for j in self.conects[i])
                    for i in match.res_atom_idcs
                ),
            )
            if not found_conects.issubset(expected_bonds):
                logging.debug(
                    "    REJECTED: CONECT record but no bond",
                )
                yield match.reject(
                    "found CONECT record that could not be matched with a bond",
                )
            else:
                logging.debug(
                    "    ACCEPTED",
                )
                yield match

    def filter_on_consecutive_chain_linkages(
        self,
        this_matches: Sequence[PossibleResidueMatch],
        prev_matches: Sequence[PossibleResidueMatch],
        next_matches: Sequence[PossibleResidueMatch],
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
    ) -> Iterator[PossibleResidueMatch]:
        """If adjacent residues within a chain can be linked, reject matches that don't link them"""
        # TODO: Nail down the definition of "Adjacent"
        logging.debug(
            f"Filtering matches for polymer linkages in {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
        )

        this_ptype_idx = next(iter(this_matches)).prototype_index
        prev_ptype_idx = (
            prev_matches[0].prototype_index if len(prev_matches) > 0 else None
        )
        next_ptype_idx = (
            next_matches[0].prototype_index if len(next_matches) > 0 else None
        )

        this_chain = self.chain_id[this_ptype_idx]
        prev_chain = (
            self.chain_id[prev_ptype_idx] if prev_ptype_idx is not None else None
        )
        next_chain = (
            self.chain_id[next_ptype_idx] if next_ptype_idx is not None else None
        )

        this_terminated = self.terminated[this_ptype_idx]
        prev_terminated = (
            self.terminated[prev_ptype_idx] if prev_ptype_idx is not None else True
        )

        can_form_prior_bond = any(
            m.prior_bond_idcs is not None
            for m in only_matched(this_matches)
            if isinstance(m, ResidueMatch)
        )
        can_form_posterior_bond = any(
            m.posterior_bond_idcs is not None
            for m in only_matched(this_matches)
            if isinstance(m, ResidueMatch)
        )

        prev_is_adjacent = (
            prev_ptype_idx is not None
            and prev_chain is not None
            and prev_chain == this_chain
            and not prev_terminated
        )
        next_is_adjacent = (
            next_ptype_idx is not None
            and next_chain is not None
            and next_chain == this_chain
            and not this_terminated
        )

        logging.debug(
            f"  {can_form_prior_bond=}",
        )
        logging.debug(
            f"  {can_form_posterior_bond=}",
        )
        logging.debug(
            f"  {prev_is_adjacent=}",
        )
        logging.debug(
            f"  {next_is_adjacent=}",
        )

        for match in this_matches:
            if not isinstance(match, ResidueMatch):
                yield match
                continue

            logging.debug(
                f"  Checking {match.description}",
            )

            current_match_doesnt_form_prior_bond = match.prior_bond_idcs is None
            current_match_doesnt_form_posterior_bond = match.posterior_bond_idcs is None

            if (
                can_form_prior_bond
                and prev_is_adjacent
                and current_match_doesnt_form_prior_bond
            ):
                reason = "adjacent residues in unterminated chain can be linked"
                logging.debug(
                    f"    REJECTED: {reason}",
                )
                yield match.reject(reason)
                continue
            if (
                can_form_posterior_bond
                and next_is_adjacent
                and current_match_doesnt_form_posterior_bond
            ):
                reason = "adjacent residues in unterminated chain can be linked"
                logging.debug(
                    f"    REJECTED: {reason}",
                )
                yield match.reject(reason)
                continue
            if match:
                logging.debug(
                    "    ACCEPTED",
                )
            yield match

    def _match_unknown_molecules_to_indices(
        self,
        indices: Sequence[int],
        unknown_molecules: Iterable[Molecule],
    ) -> Molecule | None:
        conects: set[tuple[int, int]] = set()
        pdb_idx_to_mol_idx: dict[int, int] = {}
        pdbmol = Molecule()
        for pdb_index in indices:
            pdb_idx_to_mol_idx[pdb_index] = pdbmol.add_atom(
                atomic_number=elements.NUMBERS[self.element[pdb_index]],
                formal_charge=self.charge[pdb_index] or 0,
                is_aromatic=False,
                stereochemistry=None,
                name=self.name[pdb_index],
                metadata={
                    "leaving": False,
                    **self._generate_atom_metadata(pdb_index),
                },
            )
            for conect_idx in self.conects[pdb_index]:
                conects.add(sort_tuple((pdb_index, conect_idx)))

        for a, b in conects:
            try:
                pdbmol.add_bond(
                    atom1=pdb_idx_to_mol_idx[a],
                    atom2=pdb_idx_to_mol_idx[b],
                    bond_order=1,
                    is_aromatic=False,
                )
            except KeyError:
                # Bonds between this residue and another are not supported yet
                return None

        for molecule in unknown_molecules:
            (match_found, mapping) = Molecule.are_isomorphic(
                molecule,
                pdbmol,
                return_atom_map=True,
                aromatic_matching=False,
                formal_charge_matching=False,
                bond_order_matching=False,
                atom_stereochemistry_matching=False,
                bond_stereochemistry_matching=False,
                strip_pyrimidal_n_atom_stereo=True,
            )
            if match_found:
                assert mapping is not None
                molecule = molecule.remap(mapping)
                for atom, pdbatom in zip(molecule.atoms, pdbmol.atoms):
                    atom.metadata.update(pdbatom.metadata)
                    atom.name = pdbatom.name
                molecule.generate_conformers(n_conformers=0, clear_existing=True)
                molecule.properties["pdb_idx_to_mol_atom_idx"] = pdb_idx_to_mol_idx

                molecule_pdb_indices = [
                    atom.metadata["pdb_index"] for atom in molecule.atoms
                ]
                assert molecule_pdb_indices == sorted(molecule_pdb_indices)

                return molecule
        else:
            return None

    def match_unknown_molecules(
        self,
        this_matches: Sequence[PossibleResidueMatch],
        prev_matches: Sequence[PossibleResidueMatch],
        next_matches: Sequence[PossibleResidueMatch],
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
        unknown_molecules: Iterable[Molecule],
    ) -> Iterator[PossibleResidueMatch]:
        yield from this_matches

        if any(this_matches):
            return

        logging.debug(
            f"Matching unknown_molecules against {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
        )

        unk_mol_match = self._match_unknown_molecules_to_indices(
            indices=this_matches[0].res_atom_idcs,
            unknown_molecules=unknown_molecules,
        )
        if unk_mol_match is None:
            logging.debug(
                "  No match",
            )
        else:
            logging.debug(
                f"  Matched {unk_mol_match}",
            )
            yield from (
                MoleculeMatch(
                    residue_definition=unk_mol_match,
                    index_to_atomdef={
                        i: None
                        for i in unk_mol_match.properties["pdb_idx_to_mol_atom_idx"]
                    },
                ),
            )

    def match_residues(
        self,
        residue_database: Mapping[str, Iterable[ResidueDefinition]],
        additional_substructures: Iterable[ResidueDefinition],
        unknown_molecules: Iterable[Molecule],
    ) -> list[list[PossibleResidueMatch]]:
        matches = list(self.get_name_based_matches(residue_database))

        class Filter(Protocol):
            def __call__(
                self,
                this_matches: Sequence[PossibleResidueMatch],
                prev_matches: Sequence[PossibleResidueMatch],
                next_matches: Sequence[PossibleResidueMatch],
                all_matches: Sequence[Sequence[PossibleResidueMatch]],
            ) -> Iterator[PossibleResidueMatch]: ...

        match_filters: list[Filter] = [
            self.filter_on_polymer_linkages,
            self.filter_on_crosslinks,
            functools.partial(
                self.match_additional_substructures,
                additional_substructures=additional_substructures,
            ),
            self.filter_on_conect_records,
            self.filter_on_consecutive_chain_linkages,
            functools.partial(
                self.match_unknown_molecules,
                unknown_molecules=unknown_molecules,
            ),
        ]
        for match_filter in match_filters:
            matches = [
                list(
                    match_filter(
                        this_matches=this_matches,
                        prev_matches=prev_matches,
                        next_matches=next_matches,
                        all_matches=matches,
                    ),
                )
                for prev_matches, this_matches, next_matches in with_neighbours(
                    matches,
                    default=(),
                )
            ]
        logging.debug("------")
        return matches

    def get_residue_matches(
        self,
        residue_database: Mapping[str, Iterable[ResidueDefinition]],
        additional_substructures: Iterable[ResidueDefinition],
        unknown_molecules: Iterable[Molecule],
    ) -> list[SuccessfulMatch]:
        logging.debug(
            "Getting all residue matches",
        )
        # List of residue matches, one per residue. This list has one residue
        # match for each residue in the PDB file iff every residue could be
        # identified
        residues: list[SuccessfulMatch] = []
        errors: list[list[MismatchProtocol] | list[SuccessfulMatch]] = []
        all_residues_successful = True
        for possible_residue_matches in self.match_residues(
            residue_database,
            additional_substructures,
            unknown_molecules,
        ):
            logging.debug(
                f"  Checking errors for {self.res_name[possible_residue_matches[0].prototype_index]} {possible_residue_matches[0].res_atom_idcs}",
            )

            matches: list[SuccessfulMatch] = []
            mismatches: list[MismatchProtocol] = []
            for possible_match in possible_residue_matches:
                if isinstance(possible_match, SuccessfulMatch):
                    matches.append(possible_match)
                else:
                    mismatches.append(possible_match)

            if len(matches) == 0:
                logging.debug(
                    "    FAILURE: no successful matches; PDB loading has failed",
                )
                # No matches, PDB loading has failed
                all_residues_successful = False
                errors.append(mismatches)
            elif len(matches) == 1:
                logging.debug(
                    "    SUCCESS: 1 successful match",
                )
                # 1 match, this residue has succeeded
                residues.append(unwrap(matches))
            elif all(a.agrees_with(b) for a, b in itertools.pairwise(matches)):
                logging.debug(
                    "    SUCCESS: multiple matches with consistent chemistry",
                )
                # Multiple matches that specify identical chemistry, this residue has succeeded
                residues.append(next(iter(matches)))
            else:
                logging.debug(
                    "    FAILURE: multiple matches with inconsistent chemistry; PDB loading has failed",
                )
                # Multiple matches that specify different chemistry, PDB loading has failed
                all_residues_successful = False
                errors.append(matches)

        if all_residues_successful:
            logging.debug(
                "All residues successfully matched",
            )
            return residues
        else:
            raise PdbResidueMatchError(self, errors)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)[index]
            for field in dataclasses.fields(self)
        }

    def _generate_atom_metadata(
        self,
        pdb_index: int,
    ) -> dict[str, str | int]:
        if self.res_idx is None:
            list(self.residue_indices)
            assert self.res_idx is not None, (
                "Iterating over residue indices sets res_idx"
            )

        try:
            res_number = dec_hex(self.res_seq[pdb_index])
        except ValueError:
            res_number = self.res_idx[pdb_index]

        return {
            "residue_name": self.res_name[pdb_index],
            "residue_number": res_number,
            "res_seq": self.res_seq[pdb_index],
            "residue_index": self.res_idx[pdb_index],
            "insertion_code": self.i_code[pdb_index],
            "chain_id": self.chain_id[pdb_index],
            "pdb_index": pdb_index,
            "atom_serial": self.serial[pdb_index],
            "b_factor": str(self.temp_factor[pdb_index]),
            "occupancy": str(self.occupancy[pdb_index]),
            "alt_loc": str(self.alt_loc[pdb_index]),
        }
