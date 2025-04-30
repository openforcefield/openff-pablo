"""
Classes for defining custom residues.
"""

import dataclasses
from collections.abc import Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Self

from openff.toolkit import Molecule
from openff.units import elements, unit

from openff.pablo._utils import __UNSET__, unwrap

__all__ = [
    "AtomDefinition",
    "BondDefinition",
]

_residue_definition_skip_validation = False


@contextmanager
def _skip_residue_definition_validation():  # type: ignore[deadcode]
    global _residue_definition_skip_validation
    _residue_definition_skip_validation = True
    yield
    _residue_definition_skip_validation = False


@dataclass(frozen=True)
class AtomDefinition:
    """
    Description of an atom in a residue from the Chemical Component Dictionary (CCD).
    """

    name: str
    """The canonical name of this atom"""
    synonyms: tuple[str, ...]
    """Other names this atom can have"""
    symbol: str
    """The elemental symbol for this atom"""
    leaving: bool
    """
    Whether this atom is absent when a bond is formed between two residues.

    Atoms marked as leaving are expected to be missing from a PDB file when a
    bond is formed between this residue and another to satisfy the valence
    change for the new bond. See also
    [`openff.pablo.residue.ResidueDefinition.linking_bond`][] and
    [`openff.pablo.residue.ResidueDefinition.crosslink`][].
    """
    charge: int
    """The formal charge of this atom"""
    aromatic: bool
    """Whether this atom is aromatic"""
    stereo: Literal["S", "R"] | None
    """The chirality of this atom"""

    @classmethod
    def with_defaults(
        cls,
        name: str,
        symbol: str,
        *,
        synonyms: Iterable[str] = (),
        leaving: bool = False,
        charge: int = 0,
        aromatic: bool = False,
        stereo: Literal["S", "R"] | None = None,
    ) -> Self:
        return cls(
            name=name,
            symbol=symbol,
            synonyms=tuple(synonyms),
            leaving=leaving,
            charge=charge,
            aromatic=aromatic,
            stereo=stereo,
        )

    def replace(
        self,
        *,
        name: str | __UNSET__ = __UNSET__(),
        symbol: str | __UNSET__ = __UNSET__(),
        synonyms: Iterable[str] | __UNSET__ = __UNSET__(),
        leaving: bool | __UNSET__ = __UNSET__(),
        charge: int | __UNSET__ = __UNSET__(),
        aromatic: bool | __UNSET__ = __UNSET__(),
        stereo: Literal["S", "R"] | None | __UNSET__ = __UNSET__(),
    ) -> Self:
        return self.__class__(
            name=self.name if isinstance(name, __UNSET__) else name,
            symbol=self.symbol if isinstance(symbol, __UNSET__) else symbol,
            synonyms=(
                self.synonyms if isinstance(synonyms, __UNSET__) else tuple(synonyms)
            ),
            leaving=self.leaving if isinstance(leaving, __UNSET__) else leaving,
            charge=self.charge if isinstance(charge, __UNSET__) else charge,
            aromatic=self.aromatic if isinstance(aromatic, __UNSET__) else aromatic,
            stereo=self.stereo if isinstance(stereo, __UNSET__) else stereo,
        )


@dataclass(frozen=True)
class BondDefinition:
    """
    Description of a bond in a residue from the Chemical Component Dictionary (CCD).
    """

    atom1: str
    """The canonical name of the first atom in this bond"""
    atom2: str
    """The canonical name of the second atom in this bond"""
    order: int
    """The bond order of this bond (1 for single bond, 2 for double bond, etc.)"""
    aromatic: bool
    """`True` if this bond is aromatic, `False` otherwise"""
    stereo: Literal["E", "Z"] | None
    """The stereochemistry of this bond."""

    def flipped(self) -> Self:
        """The same bond, but with the atoms in the opposite order"""
        return dataclasses.replace(self, atom1=self.atom2, atom2=self.atom1)

    @classmethod
    def with_defaults(
        cls,
        atom1: str,
        atom2: str,
        *,
        order: int = 1,
        aromatic: bool = False,
        stereo: Literal["E", "Z"] | None = None,
    ) -> Self:
        return cls(
            atom1=atom1,
            atom2=atom2,
            order=order,
            aromatic=aromatic,
            stereo=stereo,
        )

    def replace(
        self,
        *,
        atom1: str | __UNSET__ = __UNSET__(),
        atom2: str | __UNSET__ = __UNSET__(),
        order: int | __UNSET__ = __UNSET__(),
        aromatic: bool | __UNSET__ = __UNSET__(),
        stereo: Literal["E", "Z"] | None | __UNSET__ = __UNSET__(),
    ) -> Self:
        return self.__class__(
            atom1=self.atom1 if isinstance(atom1, __UNSET__) else atom1,
            atom2=self.atom2 if isinstance(atom2, __UNSET__) else atom2,
            order=self.order if isinstance(order, __UNSET__) else order,
            aromatic=(self.aromatic if isinstance(aromatic, __UNSET__) else aromatic),
            stereo=self.stereo if isinstance(stereo, __UNSET__) else stereo,
        )


@dataclass(frozen=True)
class ResidueDefinition:
    """
    Description of a residue from the Chemical Component Dictionary (CCD).
    """

    residue_name: str
    """The 3-letter residue code used in PDB files"""
    description: str
    """A longer description of the residue"""
    linking_bond: BondDefinition | None
    """Description of how this residue may bond to its neighbours in a polymer

    If the residue is only found as a monomer, `None`. Otherwise, a
    [`BondDefinition`][openff.pablo.residue.BondDefinition]. The `atom1` and
    `atom2` attributes give the canonical atom names of the atoms that form the
    linking bond. `atom1` is the name of the atom in the residue preceding the
    bond, and `atom2` is in the residue after the bond. Any atoms that the bond
    between residues replaces should be marked as `atom.leaving=True`.

    A linking bond will be formed to join two residues if and only if all of the
    below are true:

    - The residues' atom records are sequential in the PDB file
    - The residues have the same chain ID
    - There is no TER record between the residues in the PDB file
    - The two residues have identical `linking_bond` attributes
    - All leaving atoms associated with `linking_bond.atom1` are absent in
      the first encountered residue, and all leaving atoms associated with
      `linking_bond.atom2` are absent in the latter residue. Leaving atoms
      are those that have the `AtomDefinition.leaving` attribute set to
      `True`. A leaving atom is associated with atom `a` if it is bonded to
      `a`, or it is bonded to a leaving atom associated with `a`.
    - There is at least one leaving atom associated with each linking atom

    The charge of linking atoms is not modified; any change in valence is
    accounted for via the removal of leaving atoms.
    """
    crosslink: BondDefinition | None
    """Optional description of a crosslink between this residue and another.

    A crosslink is a bond between this residue and another. Crosslinks differ
    from linking bonds primarily because they may occur between residues that
    are do not appear sequentially in the PDB file. Crosslinks cannot occur
    within a residue; use a bond or residue variant instead.

    If the residue does not form cross links, `None`. Otherwise, a
    [`BondDefinition`][openff.pablo.residue.BondDefinition]. The `atom1` and
    `atom2` attributes give the canonical atom names of the atoms that form the
    crosslink. `atom1` is the name of the atom in this residue, and `atom2` is
    in the other residue. Any atoms that the bond between residues replaces
    should be marked as `atom.leaving=True`.

    A crosslink will be formed between two residues if and only if all of the
    below are true:

    - The two residues' `crosslink` attributes are identical except that their
      atom names are reversed
    - All leaving atoms associated with each residues' `crosslink.atom1`
      attribute are absent in that residue. Leaving atoms are those that have
      the `AtomDefinition.leaving` attribute set to `True`. A leaving atom
      is associated with atom `a` if it is bonded to `a`, or it is bonded to a
      leaving atom associated with `a`.
    - There is at least one leaving atom associated with each cross-linking atom

    The charge of linking atoms is not modified; any change in valence is
    accounted for via the removal of leaving atoms.
    """
    atoms: tuple[AtomDefinition, ...]
    """The atom definitions that make up this residue.

    Atoms may be marked as `leaving` only if they are associated with a
    linking or crosslinking bond; see the API documentation for those fields.
    All atoms must have unique canonical names."""
    bonds: tuple[BondDefinition, ...]
    """The bond definitions that make up this residue"""

    def replace(
        self,
        *,
        residue_name: str | __UNSET__ = __UNSET__(),
        description: str | __UNSET__ = __UNSET__(),
        linking_bond: BondDefinition | None | __UNSET__ = __UNSET__(),
        crosslink: BondDefinition | None | __UNSET__ = __UNSET__(),
        atoms: Iterable[AtomDefinition] | __UNSET__ = __UNSET__(),
        bonds: Iterable[BondDefinition] | __UNSET__ = __UNSET__(),
    ) -> Self:
        return self.__class__(
            residue_name=(
                self.residue_name
                if isinstance(residue_name, __UNSET__)
                else residue_name
            ),
            description=(
                self.description if isinstance(description, __UNSET__) else description
            ),
            linking_bond=(
                self.linking_bond
                if isinstance(linking_bond, __UNSET__)
                else linking_bond
            ),
            crosslink=self.crosslink if isinstance(crosslink, __UNSET__) else crosslink,
            atoms=self.atoms if isinstance(atoms, __UNSET__) else tuple(atoms),
            bonds=self.bonds if isinstance(bonds, __UNSET__) else tuple(bonds),
        )

    def __post_init__(self):
        if _residue_definition_skip_validation:
            return

        self._validate()

    def _validate(self):
        try:
            if (
                self.linking_bond is None
                and self.crosslink is None
                and True in {atom.leaving for atom in self.atoms}
            ):
                raise ValueError(
                    f"{self.residue_name}: Leaving atoms were specified, but there is no linking bond or crosslink",
                    self,
                )
            if len({atom.name for atom in self.atoms}) != len(self.atoms):
                raise ValueError(
                    f"{self.residue_name}: All atoms must have unique canonical names",
                )

            all_leaving_atoms = {atom.name for atom in self.atoms if atom.leaving}
            assigned_leaving_atoms = (
                self.prior_bond_leaving_atoms
                | self.posterior_bond_leaving_atoms
                | self.crosslink_leaving_atoms
            )
            unassigned_leaving_atoms = all_leaving_atoms.difference(
                assigned_leaving_atoms,
            )
            if len(unassigned_leaving_atoms) != 0:
                raise ValueError(
                    f"{self.residue_name}: Leaving atoms could not be assigned to a bond: {unassigned_leaving_atoms}",
                )
        except KeyError as e:
            raise e

    @classmethod
    def from_molecule(
        cls,
        molecule: Molecule,
        residue_name: str | None = None,
        linking_bond: BondDefinition | None = None,
        crosslink: BondDefinition | None = None,
        description: str | None = None,
    ) -> Self:
        """
        Create a `ResidueDefinition` from an OpenFF `Molecule`.

        Parameters
        ----------
        molecule
            The `Molecule` object. Canonical names are taken from the atom
            names in this object. Leaving atoms are identified from the atom
            metadata; atom's whose metadata includes a truthy value for the key
            `"leaving_atom"` are marked as leaving atoms. Synonyms are never
            set.
        residue_name
            The 3-letter code used to identify the residue in a PDB file. If
            `None`, takes name from atom's `"residue_name"` metadata entry,
            or raises `ValueError` if they do not all agree. See also
            [`openff.pablo.ResidueDefinition.residue_name`][].
        linking_bond
            Residue linking bond. May be taken from the molecule's
            `"linking_bond"` property if ``None``. See
            [`openff.pablo.ResidueDefinition.linking_bond`][].
        crosslink
            Residue crosslink. May be taken from the molecule's
            `"crosslink"` property if `None`. See
            [`openff.pablo.ResidueDefinition.crosslink`][].
        description
            An optional string describing the residue. Taken from the
            molecule's `"description"` property if `None`. See
            [`openff.pablo.ResidueDefinition.description`][].
        """
        if residue_name is None:
            atom_residue_names = {
                atom.metadata.get("residue_name", None) for atom in molecule.atoms
            }
            if len(atom_residue_names) == 1:
                atom_residue_name = unwrap(atom_residue_names)
                if isinstance(atom_residue_name, str):
                    residue_name = atom_residue_name
                else:
                    raise ValueError(
                        "residue_name None but atoms' residue names are not strings",
                    )
            else:
                raise ValueError(
                    "residue_name None and atoms do not agree on residue name",
                )

        if crosslink is None:
            molecule_crosslink = molecule.properties.get("crosslink")
            if isinstance(molecule_crosslink, BondDefinition):
                crosslink = molecule_crosslink

        if linking_bond is None:
            molecule_linking_bond = molecule.properties.get("crosslink")
            if isinstance(molecule_linking_bond, BondDefinition):
                linking_bond = molecule_linking_bond

        if description is None:
            molecule_description = molecule.properties.get("description")
            if isinstance(molecule_description, str):
                description = molecule_description
            else:
                description = ""

        atoms: list[AtomDefinition] = []
        for atom in molecule.atoms:
            synonyms_str = str(atom.metadata.get("synonyms", ""))
            atoms.append(
                AtomDefinition(
                    name=atom.name,
                    synonyms=tuple(synonyms_str.split()),
                    symbol=atom.symbol,
                    leaving=bool(atom.metadata.get("leaving_atom")),
                    charge=atom.formal_charge.m_as(unit.elementary_charge),  # type: ignore
                    stereo=atom.stereochemistry,
                    aromatic=atom.is_aromatic,
                ),
            )
        bonds: list[BondDefinition] = []
        for bond in molecule.bonds:
            bonds.append(
                BondDefinition(
                    atom1=bond.atom1.name,
                    atom2=bond.atom2.name,
                    order=bond.bond_order,
                    aromatic=bond.is_aromatic,
                    stereo=bond.stereochemistry,
                ),
            )

        return cls(
            residue_name=residue_name,
            description=description,
            linking_bond=linking_bond,
            crosslink=crosslink,
            atoms=tuple(atoms),
            bonds=tuple(bonds),
        )

    @classmethod
    def from_capped_molecule(
        cls,
        molecule: Molecule,
        residue_name: str,
        leaving_atom_indices: Collection[int],
        linking_bond: BondDefinition,
        crosslink: BondDefinition | None = None,
        description: str | None = None,
    ) -> Self:
        """
        Create a linking `ResidueDefinition` from an OpenFF `Molecule`

        Parameters
        ----------
        residue_name
            The 3-letter code used to identify the residue in a PDB file. See
            [`openff.pablo.ResidueDefinition.residue_name`][]
        molecule
            The `Molecule` object. Canonical names are taken from the atom
            names in this object. Synonyms are never set.
        leaving_atom_indices
            Indices of atoms within the `molecule` argument that should be
            marked as leaving atoms. See
            [`openff.pablo.residue.AtomDefinition.leaving`][]
        linking_bond
            The bond linking this residue to its neighbours in a polymer. See
            [`openff.pablo.ResidueDefinition.linking_bond`][]
        crosslink
            See [`openff.pablo.ResidueDefinition.crosslink`][]
        description
            An optional string describing the residue. See
            [`openff.pablo.ResidueDefinition.description`][]
        """
        molecule = deepcopy(molecule)
        for i in leaving_atom_indices:
            molecule.atom(i).metadata["leaving_atom"] = True
        return cls.from_molecule(
            residue_name=residue_name,
            molecule=molecule,
            linking_bond=linking_bond,
            crosslink=crosslink,
            description=description,
        )

    @classmethod
    def from_smiles(
        cls,
        mapped_smiles: str,
        atom_names: Mapping[int, str],
        residue_name: str,
        leaving_atoms: Collection[int] = (),
        linking_bond: BondDefinition | None = None,
        crosslink: BondDefinition | None = None,
        description: str | None = None,
    ) -> Self:
        """
        Create a ``ResidueDefinition`` from a mapped SMILES string.

        Parameters
        ----------
        residue_name
            The 3-letter code used to identify the residue in a PDB file. See
            [`openff.pablo.ResidueDefinition.residue_name`][]
        mapped_smiles
            The SMILES string. All atoms must be explicitly included with
            contiguous mapping numbers starting at 1.
        atom_names
            Mapping from SMILES string mapping numbers to the canonical atom
            name. Note that this refers to numbers in the actual SMILES string,
            and so keys should be contiguous integers starting at 1. Atom names
            must be unique.
        leaving_atoms
            SMILES string mapping numbers for atoms that should be marked as
            leaving atoms.
        linking_bond
            The bond linking this residue to its neighbours in a polymer. See
            [`openff.pablo.ResidueDefinition.linking_bond`][]
        crosslink
            See [`openff.pablo.ResidueDefinition.crosslink`][]
        description
            An optional string describing the residue. See
            [`openff.pablo.ResidueDefinition.description`][]
        """
        molecule = Molecule.from_mapped_smiles(
            mapped_smiles,
            allow_undefined_stereo=True,
        )
        leaving_atom_indices = {i - 1 for i in leaving_atoms}
        index_to_atom_name = {i - 1: name for i, name in atom_names.items()}

        if len(index_to_atom_name) != molecule.n_atoms:
            raise ValueError("Should be an atom name for each atom in SMILES string")
        if set(index_to_atom_name.keys()) != set(range(molecule.n_atoms)):
            raise ValueError(
                "Keys of atom_names should be contiguous integers starting at 1",
            )

        for i, atom in enumerate(molecule.atoms):
            if i in leaving_atom_indices:
                atom.metadata["leaving_atom"] = True
            atom.name = index_to_atom_name[i]

        return cls.from_molecule(
            residue_name=residue_name,
            molecule=molecule,
            linking_bond=linking_bond,
            description=mapped_smiles if description is None else description,
            crosslink=crosslink,
        )

    def to_openff_molecule(self) -> Molecule:
        molecule = Molecule()
        atoms: dict[str, int] = {}
        for atom in self.atoms:
            atoms[atom.name] = molecule.add_atom(
                atomic_number=elements.NUMBERS[atom.symbol],
                formal_charge=atom.charge,
                is_aromatic=atom.aromatic,
                stereochemistry=atom.stereo,
                name=atom.name,
                metadata={
                    "residue_name": self.residue_name,
                    "leaving_atom": atom.leaving,
                    "substructure_atom": not atom.leaving,
                    "synonyms": " ".join(atom.synonyms),
                },
            )

        for bond in self.bonds:
            molecule.add_bond(
                atom1=atoms[bond.atom1],
                atom2=atoms[bond.atom2],
                bond_order=bond.order,
                is_aromatic=bond.aromatic,
                stereochemistry=bond.stereo,
            )

        molecule.properties.update(
            {
                "linking_bond": self.linking_bond,
                "crosslink": self.crosslink,
                "description": self.description,
            },
        )

        return molecule

    @cached_property
    def name_to_atom(self) -> dict[str, AtomDefinition]:
        """Map from each atoms' name and synonyms to the atom definition."""
        mapping = {atom.name: atom for atom in self.atoms}
        canonical_names = set(mapping)
        for atom in self.atoms:
            for synonym in atom.synonyms:
                if synonym in mapping and mapping[synonym] != atom:
                    raise ValueError(
                        f"synonym {synonym} degenerately defined for canonical"
                        + f" names {mapping[synonym]} and {atom.name} in"
                        + f" residue {self.residue_name}",
                    )
                if synonym in canonical_names:
                    raise ValueError(
                        f"synonym {synonym} of atom {atom.name} clashes with"
                        + f" another canonical name in residue {self.residue_name}",
                    )
                mapping[synonym] = atom
        return mapping

    def atoms_bonded_to(self, atom_name: str) -> Iterator[str]:
        for bond in self.bonds:
            if bond.atom1 == atom_name:
                yield bond.atom2
            if bond.atom2 == atom_name:
                yield bond.atom1

    def _leaving_fragment_of(self, linking_atom: str) -> Iterator[str]:
        atoms_to_check = list(self.atoms_bonded_to(linking_atom))
        checked_atoms: set[str] = set()
        while atoms_to_check:
            atom_name = atoms_to_check.pop()
            if self.name_to_atom[atom_name].leaving:
                yield atom_name
                atoms_to_check.extend(
                    filter(
                        lambda x: x not in checked_atoms,
                        self.atoms_bonded_to(atom_name),
                    ),
                )
            checked_atoms.add(atom_name)

    @cached_property
    def posterior_bond_leaving_atoms(self) -> set[str]:
        return (
            set()
            if self.linking_bond is None
            else set(self._leaving_fragment_of(self.posterior_bond_linking_atom))
        )

    @cached_property
    def prior_bond_leaving_atoms(self) -> set[str]:
        return (
            set()
            if self.linking_bond is None
            else set(self._leaving_fragment_of(self.prior_bond_linking_atom))
        )

    @cached_property
    def crosslink_leaving_atoms(self) -> set[str]:
        return (
            set()
            if self.crosslink is None
            else set(self._leaving_fragment_of(self.crosslink.atom1))
        )

    @property
    def prior_bond_linking_atom(self) -> str:
        if self.linking_bond is None:
            raise ValueError("not a linking residue")
        return self.linking_bond.atom2

    @property
    def posterior_bond_linking_atom(self) -> str:
        if self.linking_bond is None:
            raise ValueError("not a linking residue")
        return self.linking_bond.atom1
