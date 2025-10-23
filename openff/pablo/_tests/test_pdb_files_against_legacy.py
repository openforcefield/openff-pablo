import pytest
from openff.toolkit import Molecule, Topology
from openff.toolkit.utils.exceptions import UnassignedChemistryInPDBError

from openff.pablo._pdb import topology_from_pdb
from openff.pablo._tests.utils import get_test_data_path
from openff.pablo._utils import sort_tuple
from openff.pablo.residue import ResidueDefinition

FAST_PDBS: list[tuple[str, list[Molecule], list[ResidueDefinition]]] = [
    (
        "prepared_pdbs/193l_prepared.pdb",  # Filename
        [],  # unknown_molecules
        [],  # additional_substructures
    ),
    (
        "3cu9_vicinal_disulfide.pdb",
        [],
        [],
    ),
    (
        "e2_7nel.pdb",
        [
            Molecule.from_smiles(
                r"C[C@]12CC[C@H]3[C@@H](CCc4cc(O)ccc34)[C@@H]1CC[C@@H]2O",
            ),
        ],
        [],
    ),
    (
        "prepared_pdbs/ions.pdb",
        [],
        [],
    ),
    (
        "prepared_pdbs/1hje_diffchain.pdb",
        [],
        [],
    ),
    (
        "prepared_pdbs/1hje_samechain.pdb",
        [],
        [],
    ),
    (
        "1UAO.pdb",
        [],
        [],
    ),
    (
        "2MUM.pdb",
        [],
        [],
    ),
]
SLOW_PDBS: list[tuple[str, list[Molecule], list[ResidueDefinition]]] = [
    (
        "5ap1_prepared.pdb",
        [
            Molecule.from_smiles(
                "O=C([O-])Cn1cc(cn1)c2ccc(cc2OCC#N)Nc3ccc(c(n3)NC4CCCCC4)C#N",
            ),
        ],
        [],
    ),
    (
        "prepared_pdbs/2hi7_prepared.pdb",
        [],
        [],
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("pdbfile", "unknown_molecules", "additional_substructures"),
    SLOW_PDBS,
)
def test_connectivity_and_atom_order_and_net_residue_charge_and_metadata_matches_legacy_slow(
    pdbfile: str,
    unknown_molecules: list[Molecule],
    additional_substructures: list[ResidueDefinition],
):
    connectivity_and_atom_order_and_net_residue_charge_and_metadata_matches_legacy(
        pdbfile,
        unknown_molecules,
        additional_substructures,
    )


@pytest.mark.parametrize(
    ("pdbfile", "unknown_molecules", "additional_substructures"),
    FAST_PDBS,
)
def test_connectivity_and_atom_order_and_net_residue_charge_and_metadata_matches_legacy_fast(
    pdbfile: str,
    unknown_molecules: list[Molecule],
    additional_substructures: list[ResidueDefinition],
):
    connectivity_and_atom_order_and_net_residue_charge_and_metadata_matches_legacy(
        pdbfile,
        unknown_molecules,
        additional_substructures,
    )


def connectivity_and_atom_order_and_net_residue_charge_and_metadata_matches_legacy(
    pdbfile: str,
    unknown_molecules: list[Molecule],
    additional_substructures: list[ResidueDefinition],
):
    filename = get_test_data_path(pdbfile)

    try:
        legacy_top: Topology = Topology.from_pdb(
            filename,
            unique_molecules=unknown_molecules,
            _additional_substructures=[
                resdef.to_openff_molecule() for resdef in additional_substructures
            ],
        )
    except UnassignedChemistryInPDBError as e:
        pytest.skip(f"PDB file {pdbfile} cannot be loaded by legacy loader")
        raise e

    pablo_top = topology_from_pdb(
        filename,
        additional_definitions=[
            *(ResidueDefinition.anon_from_molecule(mol) for mol in unknown_molecules),
            *additional_substructures,
        ],
    )

    assert pablo_top.n_molecules == legacy_top.n_molecules
    for pablo_mol, legacy_mol in zip(pablo_top.molecules, legacy_top.molecules):
        assert pablo_mol.n_atoms == legacy_mol.n_atoms
        assert pablo_mol.hill_formula == legacy_mol.hill_formula
        for pablo_atom, legacy_atom in zip(pablo_mol.atoms, legacy_mol.atoms):
            assert (
                pablo_atom.name == legacy_atom.name
                or pablo_atom.metadata.get("canonical_name", None) == legacy_atom.name
            )
            assert pablo_atom.symbol == legacy_atom.symbol
            # Legacy loader converts "A000" to "10000". New loader is more
            # hygienic about this, so it needs to be stringified
            assert (
                str(pablo_atom.metadata["residue_number"])
                == legacy_atom.metadata["residue_number"]
            )
            for key in [
                "residue_name",
                "chain_id",
                "insertion_code",
            ]:
                assert pablo_atom.metadata[key] == legacy_atom.metadata[key], (
                    f"{key} in {pablo_atom.metadata}"
                )

        pablo_bonds = {
            sort_tuple((bond.atom1_index, bond.atom2_index)) for bond in pablo_mol.bonds
        }
        legacy_bonds = {
            sort_tuple((bond.atom1_index, bond.atom2_index))
            for bond in legacy_mol.bonds
        }
        assert pablo_bonds == legacy_bonds

    for pablo_res, legacy_res in zip(pablo_top.residues, legacy_top.residues):
        pablo_res_charge, legacy_res_charge = 0, 0
        for pablo_atom, legacy_atom in zip(pablo_res.atoms, legacy_res.atoms):
            pablo_res_charge += pablo_atom.formal_charge  # type:ignore
            legacy_res_charge += legacy_atom.formal_charge  # type:ignore
        assert pablo_res_charge == legacy_res_charge
