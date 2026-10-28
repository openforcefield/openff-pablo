import logging
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import ClassVar, Literal, Protocol, Self, TypeAlias

from openff.pablo._utils import sort_tuple
from openff.pablo.exceptions import AmbiguousMatchError, PabloError

from .residue import AtomDefinition, BondDefinition, _Linkages, ResidueDefinition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PossibleMatchProtocol(Protocol):
    residue_definition: ResidueDefinition | str
    index_to_atomdef: Mapping[int, AtomDefinition | None]
    is_match: ClassVar[bool]

    @cached_property
    def res_atom_idcs(self) -> tuple[int, ...]:
        return tuple(sorted(self.index_to_atomdef))

    @cached_property
    def identified_atom_idcs(self) -> frozenset[int]:
        """PDB indices of atoms whose chemistry this match identifies"""
        return frozenset(
            i
            for i, atom in self.index_to_atomdef.items()
            if atom is not None and atom.symbol != ""
        )

    def __bool__(self) -> bool:
        return self.is_match

    @property
    def description(self) -> str:
        if isinstance(self.residue_definition, str):
            return self.residue_definition
        else:
            return self.residue_definition.description

    @property
    def prototype_index(self) -> int:
        return max(self.index_to_atomdef)


@dataclass(frozen=True)
class MatchProtocol(PossibleMatchProtocol):
    @cached_property
    def expects_prior_bond(self) -> bool:
        return False

    @cached_property
    def expects_posterior_bond(self) -> bool:
        return False

    @cached_property
    def expects_crosslink(self) -> bool:
        return False

    def agrees_with(self, other: "MatchProtocol") -> bool:
        return other == self


@dataclass(frozen=True)
class MismatchProtocol(PossibleMatchProtocol):
    pass


@dataclass(frozen=True)
class NoResidueDefinitions(MismatchProtocol):
    index_to_atomdef: Mapping[int, None]
    residue_definition: str
    is_match = False

    @property
    def description(self) -> str:
        return f"No residue definitions for {self.residue_definition}@{list(self.res_atom_idcs)[:1]}"


@dataclass(frozen=True)
class ResidueMismatch(MismatchProtocol):
    residue_definition: ResidueDefinition
    index_to_atomdef: Mapping[int, AtomDefinition | None]
    reason: str
    linkages: frozenset[_Linkages] = frozenset()
    """The combinations of external bonds this mismatch rejects

    Empty if matching failed before any combination was considered."""
    is_match = False

    @property
    def description(self) -> str:
        return f"{self.residue_definition.description} failed to match: {self.reason}"


@dataclass(frozen=True)
class ResidueMatch(MatchProtocol):
    residue_definition: ResidueDefinition
    index_to_atomdef: dict[int, AtomDefinition]
    vsite_idcs: tuple[int, ...]
    linkages: _Linkages
    """The external bonds this match forms, accounting for its missing atoms"""
    prior_bond_idcs: tuple[int, int] | None = None
    posterior_bond_idcs: tuple[int, int] | None = None
    crosslink_idcs: tuple[int, int] | None = None
    """PDB indices of each bonded atom"""
    is_match = True

    def __post_init__(self):
        missing_core_atoms = sorted(
            name for name in self.missing_atoms if not self.atom(name).leaving
        )
        if missing_core_atoms:
            raise PabloError(f"match is missing non-leaving atoms {missing_core_atoms}")
        if self.linkages not in self.residue_definition._linkage_combinations:
            raise PabloError(
                f"{self.linkages} is not a valid combination of linkages for"
                + f" {self.residue_definition.description}",
            )
        if self.linkages.leavers != frozenset(self.missing_leaving_atoms):
            raise PabloError(
                f"{self.linkages} does not account for the missing leaving atoms"
                + f" {sorted(self.missing_leaving_atoms)}",
            )
        for bond_idcs, formed, bond_name in [
            (self.prior_bond_idcs, self.linkages.prior, "prior bond"),
            (self.posterior_bond_idcs, self.linkages.posterior, "posterior bond"),
            (self.crosslink_idcs, self.linkages.crosslink, "crosslink"),
        ]:
            if bond_idcs is not None and not formed:
                raise PabloError(f"{bond_name} set on a match without that linkage")

    @classmethod
    def create_split(
        cls,
        residue_definition: ResidueDefinition,
        index_to_atomdef: dict[int, AtomDefinition],
        vsite_idcs: tuple[int, ...] = (),
    ) -> Sequence[Self | ResidueMismatch]:
        """
        Match ``index_to_atomdef`` to ``residue_definition``.

        Returns one match for each combination of linkages that the missing
        atoms allow, and a single mismatch rejecting every combination they
        don't. If a non-leaving atom is missing, no combination is considered
        and a single mismatch is returned with no linkages.
        """
        matched_names = {atom.name for atom in index_to_atomdef.values()}
        missing_atoms = [
            atom for atom in residue_definition.atoms if atom.name not in matched_names
        ]
        missing_core_atoms = sorted(
            atom.name for atom in missing_atoms if not atom.leaving
        )
        if missing_core_atoms:
            return [
                cls._mismatch(
                    residue_definition,
                    index_to_atomdef,
                    "Expected atoms are missing: " + ", ".join(missing_core_atoms),
                ),
            ]

        missing_leaving_atoms = frozenset(atom.name for atom in missing_atoms)
        results: list[Self | ResidueMismatch] = [
            cls(
                residue_definition=residue_definition,
                # TODO: Make this a frozendict once Python 3.14 support is
                #       dropped; it is shared by every match from this split
                index_to_atomdef=index_to_atomdef,
                vsite_idcs=vsite_idcs,
                linkages=linkages,
            )
            for linkages in residue_definition._linkage_combinations
            if linkages.leavers == missing_leaving_atoms
        ]

        rejected = frozenset(
            linkages
            for linkages in residue_definition._linkage_combinations
            if linkages.leavers != missing_leaving_atoms
        )
        if len(rejected) == 0:
            return results

        if len(results) == 0:
            reason = (
                f"Missing leaving atoms ({', '.join(sorted(missing_leaving_atoms))})"
                + " are not explained by any combination of bonds to other residues"
            )
        elif len(missing_leaving_atoms) == 0:
            reason = (
                "No leaving atoms are missing, but these combinations of bonds to"
                + " other residues remove some"
            )
        else:
            reason = (
                f"Missing leaving atoms ({', '.join(sorted(missing_leaving_atoms))})"
                + " are not explained by these combinations of bonds to other residues"
            )
        results.append(
            cls._mismatch(residue_definition, index_to_atomdef, reason, rejected),
        )
        return results

    @classmethod
    def _mismatch(
        cls,
        residue_definition: ResidueDefinition,
        index_to_atomdef: Mapping[int, AtomDefinition | None],
        reason: str,
        linkages: frozenset[_Linkages] = frozenset(),
    ) -> ResidueMismatch:
        return ResidueMismatch(
            residue_definition=residue_definition,
            index_to_atomdef=index_to_atomdef,
            reason=reason,
            linkages=linkages,
        )

    def reject(self, reason: str) -> ResidueMismatch:
        return self._mismatch(
            self.residue_definition,
            self.index_to_atomdef,
            reason,
            frozenset({self.linkages}),
        )

    def atom(self, identifier: int | str) -> AtomDefinition:
        """Get an atom definition by name or PDB index"""
        if isinstance(identifier, int):
            return self.index_to_atomdef[identifier]
        elif isinstance(identifier, str):
            return self.residue_definition.name_to_atom[identifier]
        else:
            raise TypeError(f"unknown identifier type {type(identifier)}")

    def bond_indices(self) -> Generator[tuple[int, int]]:
        if self.prior_bond_idcs is not None:
            yield self.prior_bond_idcs
        if self.posterior_bond_idcs is not None:
            yield self.posterior_bond_idcs
        if self.crosslink_idcs is not None:
            yield self.crosslink_idcs
        for bonddef in self.residue_definition.bonds:
            atom1_idx = self.canonical_atom_name_to_index.get(bonddef.atom1)
            atom2_idx = self.canonical_atom_name_to_index.get(bonddef.atom2)
            if atom1_idx is not None and atom2_idx is not None:
                yield (atom1_idx, atom2_idx)

    def set_crosslink(self, atom1_idx: int, atom2_idx: int) -> None:
        """Set a match for crosslinking"""
        if (
            self.residue_definition.crosslink is None
            or atom1_idx not in self.res_atom_idcs
            or self.atom(atom1_idx).name != self.residue_definition.crosslink.atom1
            or atom2_idx in self.res_atom_idcs
        ):
            raise PabloError("bad crosslink index(es)")
        if not self.linkages.crosslink:
            raise PabloError("cannot set crosslink on a match without a crosslink")
        object.__setattr__(self, "crosslink_idcs", (atom1_idx, atom2_idx))

    def set_prior_bond(self, atom1_idx: int, atom2_idx: int) -> None:
        if self.residue_definition.linking_bond is None:
            raise PabloError("cannot set prior bond without linking_bond")
        if not self.linkages.prior:
            raise PabloError("cannot set prior bond on a match without a prior bond")
        if atom1_idx in self.res_atom_idcs:
            raise PabloError("atom1 in prior bond should be in another residue")
        if atom2_idx not in self.res_atom_idcs:
            raise PabloError("atom2 in prior bond should be in this residue")

        object.__setattr__(
            self,
            "prior_bond_idcs",
            (atom1_idx, atom2_idx),
        )

    def set_posterior_bond(self, atom1_idx: int, atom2_idx: int) -> None:
        if self.residue_definition.linking_bond is None:
            raise PabloError("cannot set posterior bond without linking_bond")
        if not self.linkages.posterior:
            raise PabloError(
                "cannot set posterior bond on a match without a posterior bond",
            )
        if atom1_idx not in self.res_atom_idcs:
            raise PabloError("atom1 in posterior bond should be in this residue")
        if atom2_idx in self.res_atom_idcs:
            raise PabloError("atom2 in posterior bond should be in another residue")

        object.__setattr__(
            self,
            "posterior_bond_idcs",
            (atom1_idx, atom2_idx),
        )

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

    def must_infer_prior_bond(self) -> bool:
        if self.prior_bond_idcs is not None:
            return False
        return self.expects_prior_bond

    @cached_property
    def expects_prior_bond(self) -> bool:
        return self.linkages.prior

    def must_infer_posterior_bond(self) -> bool:
        if self.posterior_bond_idcs is not None:
            return False
        return self.expects_posterior_bond

    @cached_property
    def expects_posterior_bond(self) -> bool:
        return self.linkages.posterior

    @cached_property
    def expects_crosslink(self) -> bool:
        return self.linkages.crosslink

    def get_sorted_bond_map(self) -> dict[tuple[int, int], BondDefinition]:
        d: dict[tuple[int, int], BondDefinition] = {
            sort_tuple(
                (
                    self.canonical_atom_name_to_index[bond.atom1],
                    self.canonical_atom_name_to_index[bond.atom2],
                ),
            ): bond
            for bond in self.residue_definition.bonds
            if (
                bond.atom1 in self.canonical_atom_name_to_index
                and bond.atom2 in self.canonical_atom_name_to_index
            )
        }

        for bond_idcs, bond_definition in [
            (self.crosslink_idcs, self.residue_definition.crosslink),
            (self.prior_bond_idcs, self.residue_definition.linking_bond),
            (self.posterior_bond_idcs, self.residue_definition.linking_bond),
        ]:
            if bond_idcs is not None:
                assert bond_definition is not None
                i, j = bond_idcs
                if i < j:
                    d[i, j] = bond_definition
                else:
                    d[j, i] = bond_definition.flipped()

        return d

    def agrees_with(self, other: MatchProtocol) -> bool:
        """``True`` if both matches would assign the same chemistry, ``False`` otherwise.

        Matches are considered to assign the same chemistry if their
        connectivity graphs (ignoring bond order) and net formal charges are
        identical. Matches that identify different atoms are never considered
        to agree."""
        assert not isinstance(other, AdditionalDefMatch), (
            "matches to additional definitions are only compared with each other"
        )
        if self.identified_atom_idcs != other.identified_atom_idcs:
            logger.debug(
                "    DISAGREE: Covers different indices"
                + f" ({set(self.index_to_atomdef.keys())}"
                + f" vs {set(other.index_to_atomdef.keys())})",
            )
            return False

        if not isinstance(other, ResidueMatch):
            logger.debug("  deferring to super class")
            return super().agrees_with(other)

        # External connectivity graph should be identical
        if (
            self.residue_definition.crosslink is not None
            and self.expects_crosslink
            and (
                self.crosslink_idcs != other.crosslink_idcs
                or (
                    self.residue_definition.crosslink
                    != other.residue_definition.crosslink
                )
            )
        ):
            logger.debug(
                "  DISAGREES: external bonds differ",
            )
            return False

        if (self.expects_prior_bond or self.expects_posterior_bond) and (
            self.residue_definition.linking_bond
            != other.residue_definition.linking_bond
        ):
            logger.debug("  DISAGREES: linking bonds differ")
            return False

        expect_same_bonds = (
            self.expects_crosslink == other.expects_crosslink
            and self.expects_prior_bond == other.expects_prior_bond
            and self.expects_posterior_bond == other.expects_posterior_bond
        )
        if not expect_same_bonds:
            logger.debug("  DISAGREES: Expects different linking bonds")
            return False

        try:
            self.check_agrees_with_where_overlaps(other)
        except AmbiguousMatchError as e:
            logger.debug(
                f"  DISAGREES: {' and '.join(e.reasons)}",
            )
            return False
        return True

    def check_agrees_with_where_overlaps(self, other: "ResidueMatch"):
        overlap = {
            i: (self.index_to_atomdef[i], other.index_to_atomdef[i])
            for i in self.identified_atom_idcs & other.identified_atom_idcs
        }

        # All atoms should have the same element
        for i, (self_atom, other_atom) in overlap.items():
            if self_atom.symbol != other_atom.symbol:
                raise AmbiguousMatchError(
                    [
                        f"elements differ at index {i}:"
                        + f" {self_atom.symbol} vs {other_atom.symbol}",
                    ],
                )

        # Net charge should be identical
        self_net_charge, other_net_charge = reduce(
            lambda netcharges, atoms: (
                netcharges[0] + atoms[0].charge,
                netcharges[1] + atoms[1].charge,
            ),
            overlap.values(),
            (0, 0),
        )
        if self_net_charge != other_net_charge:
            raise AmbiguousMatchError(
                [
                    "Total formal charge differs:"
                    + f" {self_net_charge} vs {other_net_charge}",
                ],
            )

        # Internal connectivity graph should be identical
        self_bonds: set[tuple[int, int]] = {
            sort_tuple(
                (
                    self.canonical_atom_name_to_index[bond.atom1],
                    self.canonical_atom_name_to_index[bond.atom2],
                ),
            )
            for bond in self.residue_definition.bonds
            if bond.atom1 in self.canonical_atom_name_to_index
            and bond.atom2 in self.canonical_atom_name_to_index
        }
        other_bonds: set[tuple[int, int]] = {
            sort_tuple(
                (
                    other.canonical_atom_name_to_index[bond.atom1],
                    other.canonical_atom_name_to_index[bond.atom2],
                ),
            )
            for bond in other.residue_definition.bonds
            if bond.atom1 in other.canonical_atom_name_to_index
            and bond.atom2 in other.canonical_atom_name_to_index
        }
        self_bonds_overlap = {
            (idx_a, idx_b)
            for idx_a, idx_b in self_bonds
            if idx_a in overlap and idx_b in overlap
        }
        other_bonds_overlap = {
            (idx_a, idx_b)
            for idx_a, idx_b in other_bonds
            if idx_a in overlap and idx_b in overlap
        }
        if self_bonds_overlap != other_bonds_overlap:
            raise AmbiguousMatchError(["Bonds differ"])


class ResidueConectMismatch(ResidueMismatch):
    @property
    def description(self) -> str:
        return f"{self.residue_definition.description} (CONECT-based) failed to match: {self.reason}"


class ResidueConectMatch(ResidueMatch):
    @classmethod
    def _mismatch(
        cls,
        residue_definition: ResidueDefinition,
        index_to_atomdef: Mapping[int, AtomDefinition | None],
        reason: str,
        linkages: frozenset[_Linkages] = frozenset(),
    ) -> ResidueConectMismatch:
        return ResidueConectMismatch(
            residue_definition=residue_definition,
            index_to_atomdef=index_to_atomdef,
            reason=reason,
            linkages=linkages,
        )


@dataclass(frozen=True)
class AdditionalDefMatch(ResidueMatch):
    @classmethod
    def from_mapping(
        cls,
        atom_mapping: Mapping[int, AtomDefinition],
        resdef: ResidueDefinition,
        linkages: _Linkages,
    ) -> Self:
        """Compute the bond mapping for a particular atom mapping"""
        n_unique_mapped_names = len(
            {atomdef.name for atomdef in atom_mapping.values()},
        )
        assert n_unique_mapped_names == len(atom_mapping)

        neighbouring_atoms: dict[str, int] = {}
        matched_atoms: dict[str, int] = {}
        for i, atom in atom_mapping.items():
            if atom.symbol == "":
                neighbouring_atoms[atom.name] = i
            else:
                matched_atoms[atom.name] = i

        prior_bond = (
            (
                neighbouring_atoms[resdef._prior_bond_partner_atom + "_PRIOR"],
                matched_atoms[resdef._prior_bond_linking_atom],
            )
            if linkages.prior
            else None
        )
        posterior_bond = (
            (
                matched_atoms[resdef._posterior_bond_linking_atom],
                neighbouring_atoms[resdef._posterior_bond_partner_atom + "_POSTERIOR"],
            )
            if linkages.posterior
            else None
        )
        crosslink = (
            (
                matched_atoms[resdef._crosslink_linking_atom],
                neighbouring_atoms[resdef._crosslink_partner_atom + "_CROSSLINK"],
            )
            if linkages.crosslink
            else None
        )

        return cls(
            residue_definition=resdef,
            index_to_atomdef={k: v for k, v in atom_mapping.items()},
            vsite_idcs=(),
            linkages=linkages,
            prior_bond_idcs=prior_bond,
            posterior_bond_idcs=posterior_bond,
            crosslink_idcs=crosslink,
        )

    @cached_property
    def missing_atoms(self) -> set[str]:
        """Atoms present in the residue definition that were not matched

        Stand-ins for atoms outside the residue do not count as matches."""
        identified_names = {
            atom.name for atom in self.index_to_atomdef.values() if atom.symbol != ""
        }
        return {
            atom.name
            for atom in self.residue_definition.atoms
            if atom.name not in identified_names
        }

    def agrees_with(self, other: MatchProtocol) -> bool:
        """``True`` if both matches would assign the same chemistry, ``False`` otherwise.

        Unlike :py:meth:`ResidueMatch.agrees_with`, external bonds are compared
        by the atoms they join rather than by which linkage forms them."""
        assert isinstance(other, AdditionalDefMatch), (
            "matches to additional definitions are only compared with each other"
        )
        if self.identified_atom_idcs != other.identified_atom_idcs:
            return False
        if self._external_bonds() != other._external_bonds():
            logger.debug("  DISAGREES: external bonds differ")
            return False
        try:
            self.check_agrees_with_where_overlaps(other)
        except AmbiguousMatchError as e:
            logger.debug(f"  DISAGREES: {' and '.join(e.reasons)}")
            return False
        return True

    def _external_bonds(
        self,
    ) -> set[tuple[tuple[int, int], int, bool, Literal["E", "Z"] | None]]:
        """Sorted PDB indices, order, aromaticity and stereo of external bonds"""
        return {
            (sort_tuple(bond_idcs), bond.order, bond.aromatic, bond.stereo)
            for bond_idcs, bond in [
                (self.prior_bond_idcs, self.residue_definition.linking_bond),
                (self.posterior_bond_idcs, self.residue_definition.linking_bond),
                (self.crosslink_idcs, self.residue_definition.crosslink),
            ]
            if bond_idcs is not None and bond is not None
        }


SuccessfulMatch: TypeAlias = ResidueMatch
PossibleResidueMatch: TypeAlias = SuccessfulMatch | MismatchProtocol


def only_matched(
    iterable: Iterable[PossibleResidueMatch],
) -> Iterable[SuccessfulMatch]:
    return (p for p in iterable if isinstance(p, SuccessfulMatch))


def unwrap_successful(iterable: Iterable[PossibleResidueMatch]) -> SuccessfulMatch:
    iterator = iter(only_matched(iterable))

    try:
        value = next(iterator)
    except StopIteration:
        raise PabloError("container has no successful elements")

    for redundancy in iterator:
        if not value.agrees_with(redundancy):
            raise PabloError("container has multiple disagreeing successful elements")

    return value
