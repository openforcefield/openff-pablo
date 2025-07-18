import dataclasses
import logging
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from io import TextIOBase
from os import PathLike
from typing import IO, Any, ClassVar, DefaultDict, Protocol, Self
from collections.abc import Collection

from ._utils import (
    __UNSET__,
    charge_int_or_none,
    dec_hex,
    flatten,
    sort_tuple,
    with_neighbours,
)
from .exceptions import (
    UnknownOrAmbiguousSerialInConectError,
)
from .residue import AtomDefinition, BondDefinition, ResidueDefinition

__all__ = [
    "ResidueMatch",
    "PdbData",
]


# TODO: Refactor
@dataclass(frozen=True)
class ResidueMatchProtocol(Protocol):
    residue_definition: ResidueDefinition | str
    index_to_atomdef: Mapping[int, AtomDefinition | None]
    is_match: ClassVar[bool]

    @cached_property
    def res_atom_idcs(self) -> set[int]:
        return set(self.index_to_atomdef)

    def __bool__(self) -> bool:
        return self.is_match

    @property
    def description(self) -> str:
        if isinstance(self.residue_definition, str):
            return self.residue_definition
        else:
            return self.residue_definition.description


@dataclass(frozen=True)
class NoResidueDefinitions(ResidueMatchProtocol):
    index_to_atomdef: Mapping[int, None]
    residue_definition: str
    is_match = False

    @property
    def description(self) -> str:
        return f"No residue definitions for {self.residue_definition}@{list(self.res_atom_idcs)[:1]}"


@dataclass(frozen=True)
class ResidueMismatch(ResidueMatchProtocol):
    residue_definition: ResidueDefinition
    index_to_atomdef: Mapping[int, AtomDefinition | None]
    reasons: list[str]
    is_match = False

    @property
    def description(self) -> str:
        return f"{self.residue_definition.description} failed to match because {self.reasons}"


@dataclass(frozen=True)
class ResidueMatch(ResidueMatchProtocol):
    residue_definition: ResidueDefinition
    index_to_atomdef: dict[int, AtomDefinition]
    prior_bond_idcs: tuple[int, int] | None = None
    posterior_bond_idcs: tuple[int, int] | None = None
    crosslink_idcs: tuple[int, int] | None = None
    """PDB indices of each bonded atom"""
    is_match = True

    def atom(self, identifier: int | str) -> AtomDefinition:
        """Get an atom definition by name or PDB index"""
        if isinstance(identifier, int):
            return self.index_to_atomdef[identifier]
        elif isinstance(identifier, str):
            return self.residue_definition.name_to_atom[identifier]
        else:
            raise TypeError(f"unknown identifier type {type(identifier)}")

    def set_crosslink(self, atom1_idx: int, atom2_idx: int) -> None:
        """Set a match for crosslinking"""
        if (
            self.residue_definition.crosslink is None
            or atom1_idx not in self.res_atom_idcs
            or self.atom(atom1_idx).name != self.residue_definition.crosslink.atom1
            or atom2_idx in self.res_atom_idcs
        ):
            raise ValueError("bad crosslink index(es)")
        object.__setattr__(self, "crosslink_idcs", (atom1_idx, atom2_idx))

    def set_prior_bond(self, d: dict[BondDefinition, int]) -> None:
        if self.residue_definition.linking_bond is None:
            raise ValueError("cannot set prior bond without linking_bond")

        prior_bond_idcs: tuple[int, int] = (
            d[self.residue_definition.linking_bond],
            self.canonical_atom_name_to_index[
                self.residue_definition.prior_bond_linking_atom
            ],
        )
        object.__setattr__(
            self,
            "prior_bond_idcs",
            prior_bond_idcs,
        )

    def set_posterior_bond(self, d: dict[BondDefinition, int]) -> None:
        if self.residue_definition.linking_bond is None:
            raise ValueError("cannot set posterior bond without linking_bond")

        posterior_bond_idcs: tuple[int, int] = (
            d[self.residue_definition.linking_bond],
            self.canonical_atom_name_to_index[
                self.residue_definition.posterior_bond_linking_atom
            ],
        )
        object.__setattr__(
            self,
            "posterior_bond_idcs",
            posterior_bond_idcs,
        )

    @cached_property
    def prototype_index(self) -> int:
        return next(iter(self.index_to_atomdef))

    @cached_property
    def missing_atoms(self) -> set[str]:
        """Atoms present in the residue definition that were not matched"""
        return {
            atom.name
            for atom in self.residue_definition.atoms
            if atom.name not in self.canonical_atom_name_to_index
        }

    @cached_property
    def missing_leaving_atoms(self) -> set[str]:
        return {
            atom_name
            for atom_name in self.missing_atoms
            if self.atom(atom_name).leaving
        }

    @cached_property
    def canonical_atom_name_to_index(self) -> dict[str, int]:
        return {atom.name: i for i, atom in self.index_to_atomdef.items()}

    @cached_property
    def expects_prior_bond(self) -> bool:
        if self.residue_definition.linking_bond is None:
            return False

        linking_atom = self.residue_definition.prior_bond_linking_atom
        expected_leaving_atoms = self.residue_definition.prior_bond_leaving_atoms

        return (
            linking_atom in self.canonical_atom_name_to_index
            and len(expected_leaving_atoms) > 0
            and expected_leaving_atoms.issubset(self.missing_leaving_atoms)
        )

    @cached_property
    def expects_posterior_bond(self) -> bool:
        if self.residue_definition.linking_bond is None:
            return False

        linking_atom = self.residue_definition.posterior_bond_linking_atom
        expected_leaving_atoms = self.residue_definition.posterior_bond_leaving_atoms

        return (
            linking_atom in self.canonical_atom_name_to_index
            and len(expected_leaving_atoms) > 0
            and expected_leaving_atoms.issubset(self.missing_leaving_atoms)
        )

    @cached_property
    def expects_crosslink(self) -> bool:
        if self.residue_definition.crosslink is None:
            return False

        linking_atom = self.residue_definition.crosslink.atom1
        expected_leaving_atoms = self.residue_definition.crosslink_leaving_atoms

        return (
            linking_atom in self.canonical_atom_name_to_index
            and len(expected_leaving_atoms) > 0
            and expected_leaving_atoms.issubset(self.missing_leaving_atoms)
        )

    def agrees_with(self, other: Self) -> bool:
        """True if both matches would assign the same chemistry, False otherwise"""
        if set(self.index_to_atomdef.keys()) != set(other.index_to_atomdef.keys()):
            return False

        name_map: dict[str, str] = {}
        for i, self_atom in self.index_to_atomdef.items():
            other_atom = other.index_to_atomdef[i]
            if not (
                self_atom.aromatic == other_atom.aromatic
                and self_atom.charge == other_atom.charge
                and self_atom.symbol == other_atom.symbol
                and self_atom.stereo == other_atom.stereo
            ):
                return False
            name_map[self_atom.name] = other_atom.name

        self_bonds = {
            (
                *sorted([name_map[bond.atom1], name_map[bond.atom2]]),
                bond.aromatic,
                bond.order,
                bond.stereo,
            )
            for bond in self.residue_definition.bonds
            if bond.atom1 in self.canonical_atom_name_to_index
            and bond.atom2 in self.canonical_atom_name_to_index
        }
        other_bonds = {
            (
                *sorted([bond.atom1, bond.atom2]),
                bond.aromatic,
                bond.order,
                bond.stereo,
            )
            for bond in other.residue_definition.bonds
            if bond.atom1 in self.canonical_atom_name_to_index
            and bond.atom2 in self.canonical_atom_name_to_index
        }
        if self_bonds != other_bonds:
            return False

        if self.expects_crosslink and (
            self.crosslink_idcs != other.crosslink_idcs
            or self.residue_definition.crosslink != other.residue_definition.crosslink
        ):
            return False

        if (self.expects_prior_bond or self.expects_posterior_bond) and (
            self.residue_definition.linking_bond
            != other.residue_definition.linking_bond
        ):
            return False

        return (
            self.expects_crosslink == other.expects_crosslink
            and self.expects_prior_bond == other.expects_prior_bond
            and self.expects_posterior_bond == other.expects_posterior_bond
        )


@dataclass
class PossibleResidueMatch:
    match: ResidueMismatch | ResidueMatch | NoResidueDefinitions

    @classmethod
    def matched(
        cls,
        index_to_atomdef: dict[int, AtomDefinition],
        residue_definition: ResidueDefinition,
    ) -> Self:
        return cls(
            match=ResidueMatch(
                residue_definition=residue_definition,
                index_to_atomdef=index_to_atomdef,
                crosslink_idcs=None,
            ),
        )

    @classmethod
    def mismatched(
        cls,
        index_to_atomdef: dict[int, AtomDefinition | None],
        residue_definition: ResidueDefinition,
        reason: str,
    ) -> Self:
        return cls(
            match=ResidueMismatch(
                residue_definition=residue_definition,
                index_to_atomdef=index_to_atomdef,
                reasons=[reason],
            ),
        )

    def reject(self, reason: str) -> Self:
        if isinstance(self.match, ResidueMatch):
            self.match = ResidueMismatch(
                residue_definition=self.match.residue_definition,
                index_to_atomdef=self.match.index_to_atomdef,
                reasons=[*self._reasons(), reason],
            )
        return self

    @property
    def res_atom_idcs(self) -> set[int]:
        return self.match.res_atom_idcs

    @property
    def prototype_index(self) -> int:
        return next(iter(self.match.index_to_atomdef))

    def _reasons(self) -> list[str]:
        return self.match.reasons if isinstance(self.match, ResidueMismatch) else []

    def __bool__(self) -> bool:
        return bool(self.match)


def only_matched(iterable: Iterable[PossibleResidueMatch]) -> Iterable[ResidueMatch]:
    return (p.match for p in iterable if isinstance(p.match, ResidueMatch))


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
                indices = [atom_idx]
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
            return PossibleResidueMatch.mismatched(
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
            return PossibleResidueMatch.mismatched(
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
            return PossibleResidueMatch.mismatched(
                residue_definition=residue_definition,
                index_to_atomdef={i: None for i in res_atom_idcs},
                reason=reason,
            )

        match = PossibleResidueMatch.matched(
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

    def get_residue_matches(
        self,
        residue_database: Mapping[str, Iterable[ResidueDefinition]],
        additional_substructures: Iterable[ResidueDefinition],
    ) -> Iterator[list[ResidueMatch]]:
        possible_matches = self.match_residues(
            residue_database,
            additional_substructures,
        )
        return (list(only_matched(matches)) for matches in possible_matches)

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
                    PossibleResidueMatch(
                        match=NoResidueDefinitions(
                            residue_definition=res_name,
                            index_to_atomdef={i: None for i in res_atom_idcs},
                        ),
                    ),
                ]

    def filter_on_polymer_linkages(
        self,
        matches: Sequence[Sequence[PossibleResidueMatch]],
    ) -> None:
        for prev_matches, this_matches, next_matches in with_neighbours(
            matches,
            default=(),
        ):
            if len(list(only_matched(this_matches))) != 0:
                logging.debug(
                    f"Beginning link-based match of {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
                )
            # neighbour_supported_posterior_bonds and
            # neighbour_supported_prior_bonds are maps from possible linking
            # bonds in this residue to the atom index (in the neighbouring
            # residue) where the linking atom can be found
            neighbour_supported_posterior_bonds: dict[BondDefinition, int] = {
                next_match.residue_definition.linking_bond: next_match.canonical_atom_name_to_index[
                    next_match.residue_definition.prior_bond_linking_atom
                ]
                for next_match in only_matched(next_matches)
                if next_match.residue_definition.linking_bond is not None
                and next_match.expects_prior_bond
            }
            neighbour_supported_prior_bonds: dict[BondDefinition, int] = {
                prev_match.residue_definition.linking_bond: prev_match.canonical_atom_name_to_index[
                    prev_match.residue_definition.posterior_bond_linking_atom
                ]
                for prev_match in only_matched(prev_matches)
                if prev_match.residue_definition.linking_bond is not None
                and prev_match.expects_posterior_bond
            }

            valid_next_matches = list(only_matched(next_matches))
            neighbours_support_molecule_end = (
                any(
                    not next_match.expects_prior_bond
                    for next_match in valid_next_matches
                )
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
                if not isinstance(match.match, ResidueMatch):
                    continue
                logging.debug(
                    f"  Attempting link-based match against {match.match.residue_definition.description}",
                )
                if match.match.expects_prior_bond:
                    if (
                        match.match.residue_definition.linking_bond
                        not in neighbour_supported_prior_bonds
                    ):
                        reason = "Prior bond expected but not supported by neighbours"
                        logging.debug(f"    Match failed: {reason}")
                        match.reject(reason)
                        continue
                    else:
                        match.match.set_prior_bond(neighbour_supported_prior_bonds)
                elif not neighbours_support_molecule_start:
                    reason = "Prior bond not permitted but required by neighbours"
                    logging.debug(f"    Match failed: {reason}")
                    match.reject(reason)
                    continue

                if match.match.expects_posterior_bond:
                    if (
                        match.match.residue_definition.linking_bond
                        not in neighbour_supported_posterior_bonds
                    ):
                        reason = (
                            "Posterior bond expected but not supported by neighbours"
                        )
                        logging.debug(f"    Match failed: {reason}")
                        match.reject(reason)
                        continue
                    else:
                        match.match.set_posterior_bond(
                            neighbour_supported_posterior_bonds,
                        )
                elif not neighbours_support_molecule_end:
                    reason = "Posterior bond not expected but required by neighbours"
                    logging.debug(f"    Match failed: {reason}")
                    match.reject(reason)
                    continue

                logging.debug("    Accepted")

    def filter_on_crosslinks(
        self,
        matches: Sequence[Sequence[PossibleResidueMatch]],
    ) -> None:
        # Check for crosslinks
        # TODO: This could be simplified if we required crosslinking atoms not to have synonyms
        for residue_matches in matches:
            if len(list(only_matched(residue_matches))) != 0:
                logging.debug(
                    f"Assigning crosslinks for {self.res_name[residue_matches[0].prototype_index]} {residue_matches[0].res_atom_idcs}",
                )
            for match in residue_matches:
                if not isinstance(match.match, ResidueMatch):
                    continue
                logging.debug(
                    f"  Attempting crosslink-based match against {match.match.residue_definition.description}",
                )
                if match.match.crosslink_idcs is not None:
                    # This match's crosslink has already been assigned
                    logging.debug(
                        "    Skipping (crosslink already assigned)",
                    )
                    continue
                if not match.match.expects_crosslink:
                    logging.debug(
                        "    Skipping (crosslink not expected)",
                    )
                    continue
                this_crosslink_def = match.match.residue_definition.crosslink
                if this_crosslink_def is None:
                    # No crosslink defined for this match
                    logging.debug(
                        "    Skipping (crosslink not expected 2: electric boogaloo)",
                    )
                    continue
                this_crosslink_atom_idx = match.match.canonical_atom_name_to_index[
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
                    other_matches = list(matches[other_crosslink_res_idx])
                    for other_match in other_matches:
                        if not isinstance(other_match.match, ResidueMatch):
                            continue
                        logging.debug(
                            f"      in {other_match.match.residue_definition.description}",
                        )
                        other_crosslink_def = (
                            other_match.match.residue_definition.crosslink
                        )
                        other_crosslink_atom_canonical_name = other_match.match.atom(
                            other_crosslink_atom_idx,
                        ).name
                        if (
                            other_match.match.expects_crosslink
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
                            match.match.set_crosslink(
                                this_crosslink_atom_idx,
                                other_crosslink_atom_idx,
                            )
                            other_match.match.set_crosslink(
                                other_crosslink_atom_idx,
                                this_crosslink_atom_idx,
                            )
                if match.match.expects_crosslink and match.match.crosslink_idcs is None:
                    match.reject(
                        "crosslink expected but no matching crosslink partner could be found",
                    )

    def match_unknown_molecules(
        self,
        matches: Iterable[list[PossibleResidueMatch]],
        additional_substructures: Iterable[ResidueDefinition],
    ) -> Iterator[list[PossibleResidueMatch]]:
        for residue_matches in matches:
            if not any(residue_matches):
                res_atom_idcs = residue_matches[0].res_atom_idcs
                additional_matches: list[PossibleResidueMatch] = []
                for residue_definition in additional_substructures:
                    additional_matches.append(
                        self.subset_matches_residue(
                            res_atom_idcs,
                            residue_definition,
                        ),
                    )
                yield residue_matches + additional_matches
            else:
                yield residue_matches

    def filter_on_conect_records(
        self,
        matches: Iterable[list[PossibleResidueMatch]],
    ) -> None:
        for residue_matches in matches:
            for match in residue_matches:
                if not isinstance(match.match, ResidueMatch):
                    continue

                expected_bonds: set[tuple[int, int]] = set()
                for bond in match.match.residue_definition.bonds:
                    try:
                        atom1_idx = match.match.canonical_atom_name_to_index[bond.atom1]
                        atom2_idx = match.match.canonical_atom_name_to_index[bond.atom2]
                    except KeyError:
                        # Bond is for a missing leaving atom
                        continue
                    expected_bonds.add(sort_tuple((atom1_idx, atom2_idx)))
                if match.match.crosslink_idcs is not None:
                    expected_bonds.add(sort_tuple(match.match.crosslink_idcs))
                if match.match.prior_bond_idcs is not None:
                    expected_bonds.add(sort_tuple(match.match.prior_bond_idcs))
                if match.match.posterior_bond_idcs is not None:
                    expected_bonds.add(sort_tuple(match.match.posterior_bond_idcs))

                found_conects = set(
                    flatten(
                        (sort_tuple((i, j)) for j in self.conects[i])
                        for i in match.res_atom_idcs
                    ),
                )
                if not found_conects.issubset(expected_bonds):
                    print("rejected")
                    match.reject(
                        "found CONECT record that could not be matched with a bond",
                    )

    def filter_on_consecutive_chain_linkages(
        self,
        matches: Iterable[list[PossibleResidueMatch]],
    ) -> None:
        """If adjacent residues within a chain can be linked, reject matches that don't link them"""
        # TODO: Nail down the definition of "Adjacent"
        for prev_matches, this_matches, next_matches in with_neighbours(matches):
            this_ptype_idx = next(iter(this_matches)).prototype_index
            prev_ptype_idx = (
                next(iter(prev_matches)).prototype_index
                if prev_matches is not None
                else None
            )
            next_ptype_idx = (
                next(iter(next_matches)).prototype_index
                if next_matches is not None
                else None
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
                m.prior_bond_idcs is not None for m in only_matched(this_matches)
            )
            can_form_posterior_bond = any(
                m.posterior_bond_idcs is not None for m in only_matched(this_matches)
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

            for match in this_matches:
                if not isinstance(match.match, ResidueMatch):
                    continue

                current_match_doesnt_form_prior_bond = (
                    match.match.prior_bond_idcs is None
                )
                current_match_doesnt_form_posterior_bond = (
                    match.match.posterior_bond_idcs is None
                )

                if (
                    can_form_prior_bond
                    and prev_is_adjacent
                    and current_match_doesnt_form_prior_bond
                ):
                    match.reject(
                        "adjacent residues in unterminated chain can be linked",
                    )
                if (
                    can_form_posterior_bond
                    and next_is_adjacent
                    and current_match_doesnt_form_posterior_bond
                ):
                    match.reject(
                        "adjacent residues in unterminated chain can be linked",
                    )

    def match_residues(
        self,
        residue_database: Mapping[str, Iterable[ResidueDefinition]],
        additional_substructures: Iterable[ResidueDefinition],
    ) -> list[list[PossibleResidueMatch]]:
        matches = list(self.get_name_based_matches(residue_database))
        # self.rescue_partial_matches_with_conect_element_charge(matches)
        self.filter_on_polymer_linkages(matches)
        self.filter_on_crosslinks(matches)
        matches = list(
            self.match_unknown_molecules(
                matches,
                additional_substructures,
            ),
        )
        self.filter_on_conect_records(matches)
        self.filter_on_consecutive_chain_linkages(matches)
        return matches

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)[index]
            for field in dataclasses.fields(self)
        }

    def _generate_atom_metadata(
        self,
        pdb_index: int,
    ) -> dict[str, str | int]:
        try:
            res_seq = dec_hex(self.res_seq[pdb_index])
        except ValueError:
            res_seq = self.res_seq[pdb_index]

        return {
            "residue_name": self.res_name[pdb_index],
            "residue_number": self.res_seq[pdb_index],
            "res_seq": res_seq,
            "insertion_code": self.i_code[pdb_index],
            "chain_id": self.chain_id[pdb_index],
            "pdb_index": pdb_index,
            "atom_serial": self.serial[pdb_index],
            "b_factor": str(self.temp_factor[pdb_index]),
            "occupancy": str(self.occupancy[pdb_index]),
            "alt_loc": str(self.alt_loc[pdb_index]),
        }
