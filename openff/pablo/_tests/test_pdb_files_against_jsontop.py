from pathlib import Path

import pytest
from openff.toolkit import Molecule, Topology

from openff.pablo._pdb import topology_from_pdb
from openff.pablo._tests.utils import get_test_data_path
from openff.pablo._utils import sort_tuple
from openff.pablo.residue import ResidueDefinition

FAST_PDBS: list[tuple[str, str, list[Molecule], list[ResidueDefinition]]] = [
    (
        "prepared_pdbs/2MUM_neutralized.pdb",
        "prepared_pdbs/2MUM_neutralized.json",
        [],
        [],
    ),
    (
        "prepared_pdbs/2MUM_dryrun.pdb",
        "prepared_pdbs/2MUM_dryrun.json",
        [],
        [],
    ),
    (
        "prepared_pdbs/2MUM_dryrun.pdb",
        "prepared_pdbs/2MUM_neutralized.json",
        [],
        [],
    ),
    (
        "prepared_pdbs/2MUM_neutralized.pdb",
        "prepared_pdbs/2MUM_dryrun.json",
        [],
        [],
    ),
    (
        "prepared_pdbs/2MUM_blowup.pdb",
        "prepared_pdbs/2MUM_neutralized.json",
        [],
        [],
    ),
    (
        "prepared_pdbs/3h34_prepared.pdb",
        "prepared_pdbs/3h34_prepared.json",
        [],
        [],
    ),
    (
        "1A4T.pdb",
        "1A4T.json",
        [],
        [],
    ),
    (
        "prepared_pdbs/1a4t_samechain.pdb",
        "1A4T.json",
        [],
        [],
    ),
]
SLOW_PDBS: list[tuple[str, str, list[Molecule], list[ResidueDefinition]]] = [
    (
        "prepared_pdbs/5eil_fixed.pdb",
        "prepared_pdbs/5eil_fixed.json",
        [],
        [],
    ),
]


@pytest.mark.parametrize(
    "fn_stem",
    [
        "2MUM_letters_in_resseq",
        "2MUM_letters_in_serial",
        "2MUM_reuse_resseq",  # Fails because adjacent PROs can't be distinguished
        "2MUM_reuse_serial",
        "2MUM_icode",
        "2MUM_discontiguous_serial",
        "2MUM_discontiguous_resseq",
        "2MUM_composed_function",
    ],
)
def test_extended_atom_residue_numbering(
    fn_stem: str,
):
    path_stem = Path("prepared_pdbs") / fn_stem
    topology_identical_to_jsontop(
        path_stem.with_suffix(".pdb"),
        path_stem.with_suffix(".json"),
        [],
        [],
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("pdbfile", "jsontopfile", "unknown_molecules", "additional_substructures"),
    SLOW_PDBS,
)
def test_topology_identical_to_jsontop_slow(
    pdbfile: str,
    jsontopfile: str,
    unknown_molecules: list[Molecule],
    additional_substructures: list[ResidueDefinition],
):
    topology_identical_to_jsontop(
        pdbfile,
        jsontopfile,
        unknown_molecules,
        additional_substructures,
    )


@pytest.mark.parametrize(
    ("pdbfile", "jsontopfile", "unknown_molecules", "additional_substructures"),
    FAST_PDBS,
)
def test_topology_identical_to_jsontop_fast(
    pdbfile: str,
    jsontopfile: str,
    unknown_molecules: list[Molecule],
    additional_substructures: list[ResidueDefinition],
):
    topology_identical_to_jsontop(
        pdbfile,
        jsontopfile,
        unknown_molecules,
        additional_substructures,
    )


def topology_identical_to_jsontop(
    pdbfile: str | Path,
    jsontopfile: str | Path,
    unknown_molecules: list[Molecule],
    additional_substructures: list[ResidueDefinition],
):
    pablo_top = topology_from_pdb(
        get_test_data_path(pdbfile),
        unknown_molecules=unknown_molecules,
        additional_substructures=additional_substructures,
    )
    jsontop_top = Topology.from_json(get_test_data_path(jsontopfile).read_text())

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
        assert pablo_bonds == jsontop_bonds
