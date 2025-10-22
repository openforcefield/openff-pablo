from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Concatenate,
    Literal,
    Self,
)

import numpy
from numpy.typing import ArrayLike
from openff.toolkit import Molecule
from openff.toolkit.utils.exceptions import UndefinedStereochemistryError
from openff.units import unit
from pint import Quantity
from rdkit.Chem import (
    AddHs,
    AssignStereochemistryFrom3D,
    Atom,
    Bond,
    BondStereo,
    BondType,
    Conformer,
    EditableMol,
    GetMolFrags,
    GetPeriodicTable,
    Mol,
    MolFromSmiles,
    MolToSmiles,
    PeriodicTable,
    SanitizeFlags,
    SanitizeMol,
)

from openff.pablo.exceptions import PabloError

ORDER_TO_BOND_TYPE: dict[int, BondType] = {
    1: BondType.SINGLE,
    2: BondType.DOUBLE,
    3: BondType.TRIPLE,
    4: BondType.QUADRUPLE,
    5: BondType.QUINTUPLE,
    6: BondType.HEXTUPLE,
}

PERIODIC_TABLE: PeriodicTable = GetPeriodicTable()


@dataclass(frozen=True)
class RdMol:
    _mol: Mol = field(default_factory=Mol)

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        *,
        sanitize: bool = True,
        replacements: Mapping[str, str] = {},
    ) -> Self:
        rdmol = MolFromSmiles(
            smiles,
            sanitize=sanitize,
            replacements=dict(replacements),
        )
        return cls(AddHs(rdmol))

    def edit(self) -> "EditableRdMol":
        return EditableRdMol.from_frozen(self)

    def split_molecule_fragments(self) -> Iterator[Self]:
        # Insanely, GetMolFrags(mol, asMols=True) is so much slower than
        # GetMolFrags(mol) that we're better off writing our own _extract_fragment
        fragments = GetMolFrags(self._mol)
        # Sort each fragment, then sort the fragments, ensuring the output is as
        # much in the same order as possible.
        # RDKit seems to do this on its own, but sorting a sorted list is cheap
        # and guards against this behavior changing.
        # This ensures the PDB file is in order, though maybe it doesn't belong here
        sorted_fragments: list[list[int]] = sorted(
            sorted(atom_idcs) for atom_idcs in fragments
        )
        assert sum(len(atom_idcs) for atom_idcs in sorted_fragments) == self.n_atoms
        # Now set the conformers
        positions = self.conformers()
        for fragment_idcs in sorted_fragments:
            fragment = self._extract_fragment(fragment_idcs)
            fragment = (
                fragment.edit().add_conformers_and(positions[:, fragment_idcs]).freeze()
            )
            yield self.__class__(fragment._mol)

    def _extract_fragment(self, idcs: Sequence[int]) -> Self:
        fragment = EditableRdMol()
        bonds: dict[tuple[int, int], RdBond] = {}
        idcs_set = set(idcs)
        assert len(idcs_set) == len(idcs)
        idx_map: dict[int, int] = {}
        for i in idcs:
            atom = self.atom(i)
            for bond in atom.bonds:
                if (
                    bond.begin_atom.index not in idcs_set
                    or bond.end_atom.index not in idcs_set
                ):
                    raise PabloError("idcs must be a complete fragment")
                bonds[bond.begin_atom.index, bond.end_atom.index] = bond
            assert i not in idx_map
            idx_map[i] = fragment.add_atom(atom)
        for bond in bonds.values():
            fragment.add_bond(
                idx_map[bond.begin_atom.index],
                idx_map[bond.end_atom.index],
                bond.bond_type,
            )
        rdmol = fragment.freeze()
        assert rdmol.n_atoms == len(idcs)
        return self.__class__(rdmol._mol)

    def to_openff_molecule(self) -> Molecule:
        props = [atom.properties for atom in self.atoms]
        names = [atom.name for atom in self.atoms]
        offmol = Molecule.from_rdkit(self._mol, hydrogens_are_explicit=True)
        for atom, atom_props, atom_name in zip(offmol.atoms, props, names, strict=True):
            atom.name = atom_name
            for k, v in atom_props.items():
                atom.metadata[k] = v if isinstance(v, int) else str(v)
        return offmol

    def conformer(self, idx: int) -> Quantity:
        return numpy.asarray(self._mol.GetConformer(idx).GetPositions()) * unit.angstrom

    def conformers(self) -> Quantity:
        positions = numpy.asarray(
            [conformer.GetPositions() for conformer in self._mol.GetConformers()],
        )
        if len(positions) == 0:
            positions.shape = (0, self.n_atoms, 3)

        return positions * unit.angstrom

    @property
    def atoms(self) -> Iterator["RdAtom"]:
        yield from (RdAtom(atom) for atom in self._mol.GetAtoms())

    @property
    def n_atoms(self) -> int:
        return self._mol.GetNumAtoms(onlyExplicit=False)

    @property
    def bonds(self) -> Iterator["RdBond"]:
        yield from (RdBond(bond) for bond in self._mol.GetBonds())

    def atom(self, idx: int) -> "RdAtom":
        return RdAtom(self._mol.GetAtomWithIdx(idx))

    def to_smiles(self) -> str:
        return MolToSmiles(self._mol)


# TODO: Convert to use RWMol
@dataclass
class EditableRdMol:
    _mol: EditableMol = field(default_factory=lambda: EditableMol(Mol()))

    def __post_init__(self):
        self._begin_batch_edit()

    def _begin_batch_edit(self):
        self._mol.BeginBatchEdit()
        pass

    def _commit_batch_edit(self):
        self._mol.CommitBatchEdit()
        pass

    def _modify_via_mol[T, **P](
        self,
        func: Callable[Concatenate[Mol, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Call a function that usually works on a ``Mol`` as though it worked
        on ``EditableMol```"""
        self._commit_batch_edit()
        rdmol = self._mol.GetMol()
        ret = func(rdmol, *args, **kwargs)
        self._mol = EditableMol(rdmol)
        self._begin_batch_edit()
        return ret

    @classmethod
    def from_frozen(cls, rdmol: RdMol) -> Self:
        return cls(EditableMol(rdmol._mol))

    def freeze(self) -> RdMol:
        self._commit_batch_edit()
        return RdMol(Mol(self._mol.GetMol()))

    def add_atom(self, atom: "RdAtom") -> int:
        return self._mol.AddAtom(atom._atom)

    def add_atom_and(self, atom: "RdAtom") -> Self:
        self.add_atom(atom)
        return self

    def add_bond(self, atom1: int, atom2: int, order: int | BondType) -> int:
        if not isinstance(order, BondType):
            order = ORDER_TO_BOND_TYPE[order]

        return self._mol.AddBond(atom1, atom2, order)

    def add_bond_and(self, atom1: int, atom2: int, order: int) -> Self:
        self.add_bond(atom1, atom2, order)
        return self

    def add_atom_with(
        self,
        element: str | int,
        formal_charge: int | None = None,
        is_aromatic: bool | None = None,
        properties: Mapping[str, int | str | float] | None = None,
    ) -> int:
        if isinstance(element, int):
            atom = RdAtom.from_atomic_number(element)
        else:
            atom = RdAtom.from_symbol(element)

        if formal_charge is not None:
            atom = atom.with_formal_charge(formal_charge)

        if is_aromatic is not None:
            atom = atom.with_is_aromatic(is_aromatic)

        if properties is not None:
            atom = atom.with_properties(properties)

        return self.add_atom(atom)

    def add_atom_with_and(
        self,
        element: str | int,
        formal_charge: int | None = None,
        is_aromatic: bool | None = None,
        properties: Mapping[str, int | str | float] | None = None,
    ) -> Self:
        self.add_atom_with(
            element=element,
            formal_charge=formal_charge,
            is_aromatic=is_aromatic,
            properties=properties,
        )
        return self

    def assign_stereochemistry_from_3d(self, conformer_idx: int | None = None) -> None:
        if conformer_idx is None:
            if self._n_conformers == 1:
                conformer_idx = 0
            elif self._n_conformers == 0:
                raise PabloError(
                    "assign_stereochemistry_from_3d requires at least one conformer",
                )
            else:
                raise PabloError(
                    "Conformer index must be provided when there are more than one conformer",
                )

        self._modify_via_mol(
            AssignStereochemistryFrom3D,
            confId=conformer_idx,
            replaceExistingTags=True,
        )

    def assign_stereochemistry_from_3d_and(
        self,
        conformer_idx: int | None = None,
    ) -> Self:
        self.assign_stereochemistry_from_3d(conformer_idx)

        return self

    @property
    def _n_conformers(self) -> int:
        return self._modify_via_mol(Mol.GetNumConformers)

    def add_conformer(self, positions: Quantity | ArrayLike) -> int:
        if isinstance(positions, Quantity):
            positions = positions.m_as("angstrom")
        positions_array = numpy.asarray(positions)

        conformer = Conformer(len(positions_array))
        conformer.SetPositions(numpy.asarray(positions_array))

        return self._modify_via_mol(Mol.AddConformer, conformer)

    def add_conformer_and(self, positions: Quantity | ArrayLike) -> Self:
        self.add_conformer(positions)
        return self

    def add_conformers(self, positions: Quantity | ArrayLike):
        if isinstance(positions, Quantity):
            positions = positions.m_as("angstrom")
        positions_array = numpy.asarray(positions)

        for conformer_positions in positions_array:
            self.add_conformer(conformer_positions)

    def add_conformers_and(self, positions: Quantity | ArrayLike) -> Self:
        self.add_conformers(positions)
        return self

    def sanitize(
        self,
        adjust_hs: bool = True,
        clean_up: bool = True,
        clean_up_atropoisomers: bool = True,
        clean_up_chirality: bool = True,
        clean_up_organometallics: bool = True,
        find_radicals: bool = True,
        kekulize: bool = True,
        sanitize_properties: bool = True,
        set_aromaticity: bool = True,
        set_conjugation: bool = True,
        set_hybridization: bool = True,
        symmetrize_rings: bool = True,
    ) -> None:
        flags: SanitizeFlags = (
            SanitizeFlags.SANITIZE_ADJUSTHS * bool(adjust_hs)
            | SanitizeFlags.SANITIZE_CLEANUP * bool(clean_up)
            | SanitizeFlags.SANITIZE_CLEANUPATROPISOMERS * bool(clean_up_atropoisomers)
            | SanitizeFlags.SANITIZE_CLEANUPCHIRALITY * bool(clean_up_chirality)
            | SanitizeFlags.SANITIZE_CLEANUP_ORGANOMETALLICS
            * bool(clean_up_organometallics)
            | SanitizeFlags.SANITIZE_FINDRADICALS * bool(find_radicals)
            | SanitizeFlags.SANITIZE_KEKULIZE * bool(kekulize)
            | SanitizeFlags.SANITIZE_PROPERTIES * bool(sanitize_properties)
            | SanitizeFlags.SANITIZE_SETAROMATICITY * bool(set_aromaticity)
            | SanitizeFlags.SANITIZE_SETCONJUGATION * bool(set_conjugation)
            | SanitizeFlags.SANITIZE_SETHYBRIDIZATION * bool(set_hybridization)
            | SanitizeFlags.SANITIZE_SYMMRINGS * bool(symmetrize_rings)
        )
        self._modify_via_mol(SanitizeMol, sanitizeOps=flags)

    def sanitize_and(self) -> Self:
        self.sanitize()
        return self


@dataclass(frozen=True)
class RdAtom:
    _atom: Atom

    @classmethod
    def from_atomic_number(
        cls,
        atomic_number: int,
        *,
        allow_implicit_hydrogens: bool = False,
    ) -> Self:
        atom = Atom(atomic_number)
        atom.SetNoImplicit(not allow_implicit_hydrogens)
        return cls(atom)

    @classmethod
    def from_symbol(cls, symbol: str) -> Self:
        return cls.from_atomic_number(PERIODIC_TABLE.GetAtomicNumber(symbol))

    @property
    def atomic_number(self) -> int:
        return self._atom.GetAtomicNum()

    def with_formal_charge(self, formal_charge: int) -> Self:
        new = self._copy()
        new._atom.SetFormalCharge(formal_charge)
        return new

    @property
    def formal_charge(self) -> int:
        return self._atom.GetFormalCharge()

    def with_is_aromatic(self, is_aromatic: bool) -> Self:
        new = self._copy()
        new._atom.SetIsAromatic(is_aromatic)
        return new

    @property
    def is_aromatic(self) -> bool:
        return self._atom.GetIsAromatic()

    def _copy(self) -> Self:
        return self.__class__(Atom(self._atom))

    def with_properties(self, properties: Mapping[str, int | str | float]) -> Self:
        new = self._copy()
        for key, value in properties.items():
            if isinstance(value, str):
                new._atom.SetProp(key, value)
            elif isinstance(value, int):
                new._atom.SetUnsignedProp(key, value)
            elif isinstance(value, float):
                new._atom.SetDoubleProp(key, value)
            else:
                raise TypeError("atom properties must be str or int")
        return new

    @property
    def properties(self) -> Mapping[str, int | str | float]:
        return MappingProxyType(
            {
                # Work around https://github.com/rdkit/rdkit/issues/8890
                k: (v if v != "" else self._atom.GetProp(k))
                for k, v in self._atom.GetPropsAsDict(
                    includePrivate=False,
                    includeComputed=False,
                ).items()
            },
        )

    def stereochemistry(self) -> Literal[None, "R", "S"]:
        if self._atom.HasProp("_CIPCode"):
            stereo_code = self._atom.GetProp("_CIPCode")
            if stereo_code == "R":
                return "R"
            elif stereo_code == "S":
                return "S"
            else:
                raise UndefinedStereochemistryError(
                    "In from_pdb: Expected atom stereochemistry of R or S. "
                    f"Got {stereo_code} instead.",
                )
        else:
            return None

    @property
    def name(self) -> str:
        return self._atom.GetProp("_name")

    @property
    def n_radical_electrons(self) -> int:
        return self._atom.GetNumRadicalElectrons()

    @property
    def n_bonds(self) -> int:
        return len(self._atom.GetBonds())

    @property
    def bonds(self) -> Iterator["RdBond"]:
        for bond in self._atom.GetBonds():
            yield RdBond(bond)

    @property
    def index(self) -> int:
        return self._atom.GetIdx()


@dataclass(frozen=True)
class RdBond:
    _bond: Bond

    @property
    def begin_atom(self) -> RdAtom:
        return RdAtom(self._bond.GetBeginAtom())

    @property
    def end_atom(self) -> RdAtom:
        return RdAtom(self._bond.GetEndAtom())

    @property
    def order(self) -> float:
        return self._bond.GetBondTypeAsDouble()

    @property
    def bond_type(self) -> BondType:
        return self._bond.GetBondType()

    @property
    def stereochemistry(self) -> Literal[None, "Z", "E"]:
        tag = self._bond.GetStereo()
        match tag:
            case BondStereo.STEREOZ:
                return "Z"
            case BondStereo.STEREOE:
                return "E"
            case BondStereo.STEREONONE:
                return None
            case _:
                raise PabloError(
                    f"Expected RDKit bond stereochemistry of E or Z, got {tag} instead",
                )
