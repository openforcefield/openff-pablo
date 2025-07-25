from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Protocol, Self, TypeAlias

from openff.toolkit import Molecule

from .residue import AtomDefinition, BondDefinition, ResidueDefinition


@dataclass(frozen=True)
class PossibleMatchProtocol(Protocol):
    residue_definition: ResidueDefinition | str | Molecule
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
        elif isinstance(self.residue_definition, Molecule):
            return self.residue_definition.name
        else:
            return self.residue_definition.description

    @property
    def prototype_index(self) -> int:
        return max(self.index_to_atomdef)

    def sort_key(self) -> tuple[str, ...]:
        return (self.description,)


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

    def sort_key(self) -> tuple[str, ...]:
        return (
            self.reason,
            f"{len(self.residue_definition.description):0>10}",
            self.residue_definition.description,
        )


@dataclass(frozen=True)
class ResidueMatch(MatchProtocol):
    residue_definition: ResidueDefinition
    index_to_atomdef: dict[int, AtomDefinition]
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

    def agrees_with(self, other: MatchProtocol) -> bool:
        """True if both matches would assign the same chemistry, False otherwise"""
        if set(self.index_to_atomdef.keys()) != set(other.index_to_atomdef.keys()):
            return False

        if not isinstance(other, ResidueMatch):
            return super().agrees_with(other)

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


@dataclass(frozen=True)
class MoleculeMatch(MatchProtocol):
    residue_definition: Molecule
    index_to_atomdef: dict[int, None]
    is_match = True

    def agrees_with(self, other: MatchProtocol) -> bool:
        return (  # no-fmt
            other.residue_definition is self.residue_definition
            and set(self.index_to_atomdef.keys()) == set(other.index_to_atomdef.keys())
        )

    @property
    def match(self) -> Self:
        return self


SuccessfulMatch: TypeAlias = ResidueMatch | MoleculeMatch
PossibleResidueMatch: TypeAlias = SuccessfulMatch | MismatchProtocol


def only_matched(
    iterable: Iterable[PossibleResidueMatch],
) -> Iterable[MatchProtocol]:
    return (p for p in iterable if isinstance(p, MatchProtocol))
