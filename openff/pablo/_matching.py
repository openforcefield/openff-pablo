import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import ClassVar, Protocol, TypeAlias

from openff.pablo._utils import sort_tuple
from openff.pablo.exceptions import AmbiguousMatchError, PabloError

from .residue import AtomDefinition, BondDefinition, ResidueDefinition


@dataclass(frozen=True)
class PossibleMatchProtocol(Protocol):
    residue_definition: ResidueDefinition | str
    index_to_atomdef: Mapping[int, AtomDefinition | None]
    is_match: ClassVar[bool]

    @cached_property
    def res_atom_idcs(self) -> tuple[int, ...]:
        return tuple(sorted(self.index_to_atomdef))

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
    is_match = False

    @property
    def description(self) -> str:
        return f"{self.residue_definition.description} failed to match: {self.reason}"


@dataclass(frozen=True)
class ResidueMatch(MatchProtocol):
    residue_definition: ResidueDefinition
    index_to_atomdef: dict[int, AtomDefinition]
    vsite_idcs: tuple[int, ...]
    prior_bond_idcs: tuple[int, int] | None = None
    posterior_bond_idcs: tuple[int, int] | None = None
    crosslink_idcs: tuple[int, int] | None = None
    """PDB indices of each bonded atom"""
    is_match = True

    def reject(self, reason: str) -> ResidueMismatch:
        return ResidueMismatch(
            residue_definition=self.residue_definition,
            index_to_atomdef=self.index_to_atomdef,
            reason=reason,
        )

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
            raise PabloError("bad crosslink index(es)")
        object.__setattr__(self, "crosslink_idcs", (atom1_idx, atom2_idx))

    def set_prior_bond(self, atom1_idx: int, atom2_idx: int) -> None:
        if self.residue_definition.linking_bond is None:
            raise PabloError("cannot set prior bond without linking_bond")
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

    @cached_property
    def expects_prior_bond(self) -> bool:
        if self.residue_definition.linking_bond is None:
            return False

        linking_atom = self.residue_definition._prior_bond_linking_atom
        expected_leaving_atoms = self.residue_definition._prior_bond_leaving_atoms

        return (
            linking_atom in self.canonical_atom_name_to_index
            and len(expected_leaving_atoms) > 0
            and expected_leaving_atoms.issubset(self.missing_leaving_atoms)
        )

    @cached_property
    def expects_posterior_bond(self) -> bool:
        if self.residue_definition.linking_bond is None:
            return False

        linking_atom = self.residue_definition._posterior_bond_linking_atom
        expected_leaving_atoms = self.residue_definition._posterior_bond_leaving_atoms

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
        expected_leaving_atoms = self.residue_definition._crosslink_leaving_atoms

        return (
            linking_atom in self.canonical_atom_name_to_index
            and len(expected_leaving_atoms) > 0
            and expected_leaving_atoms.issubset(self.missing_leaving_atoms)
        )

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
        identical."""
        if set(self.index_to_atomdef.keys()) != set(other.index_to_atomdef.keys()):
            logging.debug(
                "  DISAGREE: Covers different indices"
                + f" ({set(self.index_to_atomdef.keys())}"
                + f" vs {set(other.index_to_atomdef.keys())})",
            )
            return False

        if not isinstance(other, ResidueMatch):
            logging.debug("  deferring to super class")
            return super().agrees_with(other)

        # External connectivity graph should be identical
        if (
            self.residue_definition.crosslink is not None
            and self.expects_crosslink
            and (
                self.crosslink_idcs != other.crosslink_idcs
                or (
                    self.residue_definition.crosslink.flipped()
                    != other.residue_definition.crosslink
                )
            )
        ):
            logging.debug("  DISAGREES: external bonds differ")
            return False

        if (self.expects_prior_bond or self.expects_posterior_bond) and (
            self.residue_definition.linking_bond
            != other.residue_definition.linking_bond
        ):
            logging.debug("  DISAGREES: linking bonds differ")
            return False

        expect_same_bonds = (
            self.expects_crosslink == other.expects_crosslink
            and self.expects_prior_bond == other.expects_prior_bond
            and self.expects_posterior_bond == other.expects_posterior_bond
        )
        if not expect_same_bonds:
            logging.debug("  DISAGREES: Expects different linking bonds")
            return False

        try:
            self.check_agrees_with_where_overlaps(other)
        except AmbiguousMatchError as e:
            logging.debug(
                f"  DISAGREES: {''.join(f'\n  {reason}' for reason in e.reasons)}",
            )
            return False
        return True

    def check_agrees_with_where_overlaps(self, other: "ResidueMatch"):
        overlap = {
            i: (self_atom, other.index_to_atomdef[i])
            for i, self_atom in self.index_to_atomdef.items()
            if i in other.index_to_atomdef
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
    def reject(self, reason: str) -> ResidueConectMismatch:
        return ResidueConectMismatch(
            residue_definition=self.residue_definition,
            index_to_atomdef=self.index_to_atomdef,
            reason=reason,
        )


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
