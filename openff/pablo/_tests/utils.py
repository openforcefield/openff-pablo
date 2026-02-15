import importlib.resources
from functools import lru_cache
from pathlib import Path

from openff.toolkit import Topology

from openff.pablo._utils import sort_tuple


@lru_cache
def get_test_data_path(path: Path | str) -> Path:
    """Get the filename of a resource"""
    # TODO: Do this non-deprecated-ly
    assert __package__ is not None
    with importlib.resources.as_file(importlib.resources.files(__package__)) as root:
        pass
    assert root.is_dir()
    return root / "data" / path


def topology_identical_to_jsontop(
    pablo_top: Topology,
    jsontop_top: Topology,
):
    assert pablo_top.n_molecules == jsontop_top.n_molecules
    for pablo_mol, jsontop_mol in zip(pablo_top.molecules, jsontop_top.molecules):
        assert pablo_mol.n_atoms == jsontop_mol.n_atoms
        for pablo_atom, jsontop_atom in zip(pablo_mol.atoms, jsontop_mol.atoms):
            assert pablo_atom.symbol == jsontop_atom.symbol
            assert pablo_atom.formal_charge == jsontop_atom.formal_charge
            assert pablo_atom.name == jsontop_atom.name
            # assert pablo_atom.metadata == jsontop_atom.metadata

        pablo_bonds = {
            (sort_tuple((bond.atom1_index, bond.atom2_index)), bond.bond_order)
            for bond in pablo_mol.bonds
        }
        jsontop_bonds = {
            (sort_tuple((bond.atom1_index, bond.atom2_index)), bond.bond_order)
            for bond in jsontop_mol.bonds
        }
        if pablo_bonds == jsontop_bonds:
            # molecules are identical (even the same kekulization)
            continue
        else:
            assert pablo_mol.is_isomorphic_with(jsontop_mol)
