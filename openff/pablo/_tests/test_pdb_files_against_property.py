import pytest
from openff.toolkit import Molecule, Topology, unit

from openff.pablo._pdb import topology_from_pdb
from openff.pablo._tests.utils import get_test_data_path
from openff.pablo._utils import sort_tuple
from openff.pablo.ccd import CCD_RESIDUE_DEFINITION_CACHE
from openff.pablo.chem import PEPTIDE_BOND
from openff.pablo.residue import ResidueDefinition


@pytest.mark.slow
def test_2zuq_cross_chain_disulfide_discontinuous():
    pdbfn = get_test_data_path("prepared_pdbs/2zuq_prepared.pdb")
    with pytest.warns():
        topology = topology_from_pdb(pdbfn)

    # Correct number of molecules
    assert topology.n_molecules == 3
    # Correct number of chains loaded in correct order
    assert [elem.identifier[0] for elem in topology.chains] == ["A", "C", "B", "D"]
    # Chains belong to correct molecules
    assert all(
        elem.scheme.parent is topology.molecule(mol_idx)
        for elem, mol_idx in zip(topology.chains, [0, 0, 1, 2])
    )
    # All molecules represent a single molecule
    assert not topology.molecule(0)._has_multiple_molecules()  # type: ignore
    assert not topology.molecule(1)._has_multiple_molecules()  # type: ignore
    assert not topology.molecule(2)._has_multiple_molecules()  # type: ignore
    # First molecule's pdb_index values are contiguous except for one discontinuity
    index_offset = 0
    for i, atom in enumerate(topology.molecule(0).atoms):
        pdb_index: int = atom.metadata["pdb_index"]  # type: ignore
        if pdb_index != i and index_offset == 0:
            index_offset = pdb_index - i

        assert pdb_index == i + index_offset
    # Other molecule's pdb_index values are contiguous
    assert all(
        i == atom.metadata["pdb_index"]
        for i, atom in enumerate(
            topology.molecule(1).atoms,
            start=topology.molecule(1).atom(0).metadata["pdb_index"],  # type: ignore
        )
    )
    assert all(
        i == atom.metadata["pdb_index"]
        for i, atom in enumerate(
            topology.molecule(2).atoms,
            start=topology.molecule(2).atom(0).metadata["pdb_index"],  # type: ignore
        )
    )


def test_2mum_neutralized_has_all_neutral_aas(all_aa_resnames: set[str]):
    pdbfn = get_test_data_path("prepared_pdbs/2MUM_neutralized.pdb")
    topology = topology_from_pdb(pdbfn)
    assert {residue.identifier[3] for residue in topology.residues} == all_aa_resnames
    print(
        *[
            (
                atom.name,
                atom.metadata["residue_name"],
                atom.metadata["res_seq"],
                atom.formal_charge.m,
            )
            for atom in topology.atoms
            if atom.formal_charge.m != 0
        ],
        sep="\n",
    )
    assert {
        atom.formal_charge.m_as("elementary_charge")
        for atom in topology.atoms  # type: ignore
    } == {0}


@pytest.mark.slow
def test_1p3q_loads_chains_without_ter():
    pdbfn = get_test_data_path("prepared_pdbs/1p3q_noter.pdb")
    topology = topology_from_pdb(pdbfn)

    # Correct number of molecules
    assert topology.n_molecules == 5
    # Correct number of chains loaded in correct order
    assert [elem.identifier[0] for elem in topology.chains] == ["A", "A", "B", "C", "D"]
    # Chains belong to correct molecules
    assert all(
        elem.scheme.parent is topology.molecule(mol_idx)
        for elem, mol_idx in zip(topology.chains, [0, 1, 2, 3, 4])
    )
    # All molecules represent a single molecule
    assert not topology.molecule(0)._has_multiple_molecules()  # type: ignore
    assert not topology.molecule(1)._has_multiple_molecules()  # type: ignore
    assert not topology.molecule(2)._has_multiple_molecules()  # type: ignore
    assert not topology.molecule(3)._has_multiple_molecules()  # type: ignore
    assert not topology.molecule(4)._has_multiple_molecules()  # type: ignore
    # All molecule's pdb_index values are contiguous
    for molecule in topology.molecules:
        assert all(
            i == atom.metadata["pdb_index"]
            for i, atom in enumerate(
                molecule.atoms,
                start=molecule.atom(0).metadata["pdb_index"],  # type: ignore
            )
        )


def test_5eil_is_three_proteins_with_ncaa_plus_fe3_and_water():
    topology = topology_from_pdb(
        get_test_data_path("prepared_pdbs/5eil_fixed.pdb"),
        verbose_errors=True,
    )

    protein_a = topology.molecule(0)
    protein_b = topology.molecule(1)
    protein_c = topology.molecule(2)

    fe = topology.molecule(3)
    assert [atom.symbol for atom in fe.atoms] == ["Fe"]
    assert fe.atom(0).formal_charge == 3 * unit.elementary_charge  # type: ignore

    assert all([mol.n_atoms == 3 for mol in topology.molecules][4:])

    assert not protein_a._has_multiple_molecules()  # type: ignore
    assert not protein_b._has_multiple_molecules()  # type: ignore
    assert not protein_c._has_multiple_molecules()  # type: ignore

    assert "BP5" in [res.identifier[3] for res in protein_a.residues]
    assert "BP5" in [res.identifier[3] for res in protein_b.residues]
    assert "BP5" in [res.identifier[3] for res in protein_c.residues]


@pytest.mark.slow
def test_3ip9_loads_with_additional_residue():
    smiles = "[c:1]1([H:41])[c:2]([H:42])[c:3]2[c:4]([c:5]([H:43])[c:6]1[N:7]1[C:8](=[O:9])[C:10]([H:44])([H:45])[C@@:11]([S:12][C:13]([C@:14]([N:15]([H:16])[H:50])([C:17](=[O:18])[O:19][H:59])[H:49])([H:47])[H:48])([H:46])[C:20]1=[O:21])[C:22](=[O:23])[O:24][C:25]21[c:26]2[c:27]([c:28]([H:51])[c:29]([O:32][H:54])[c:30]([H:52])[c:31]2[H:53])[O:33][c:34]2[c:35]1[c:36]([H:55])[c:37]([H:56])[c:38]([O:40][H:58])[c:39]2[H:57]"
    leavers = {16, 19, 59}
    pablo_top = topology_from_pdb(
        get_test_data_path("prepared_pdbs/3ip9_dye_solvated.pdb"),
        verbose_errors=True,
        residue_database=CCD_RESIDUE_DEFINITION_CACHE.with_(
            {
                "DYE": [
                    ResidueDefinition.from_smiles(
                        residue_name="DYE",
                        mapped_smiles=smiles,
                        atom_names={
                            15: "N",
                            16: "H2",
                            50: "H",
                            14: "CA",
                            49: "HA",
                            17: "C",
                            18: "O",
                            19: "OXT",
                            59: "HXT",
                            13: "CB",
                            47: "2HB",
                            48: "3HB",
                            12: "SG",
                            11: "C1x",
                            46: "H1x",
                            20: "C24x",
                            21: "O7x",
                            7: "N1x",
                            8: "C3x",
                            9: "O1x",
                            10: "C2x",
                            44: "H2x",
                            45: "H3x",
                            6: "C4x",
                            1: "C5x",
                            41: "H4x",
                            2: "C6x",
                            42: "H5x",
                            5: "C9x",
                            43: "H6x",
                            4: "C8x",
                            3: "C7x",
                            22: "C10x",
                            23: "O2x",
                            24: "O3x",
                            25: "C11x",
                            26: "C23x",
                            31: "C22x",
                            53: "H14x",
                            30: "C21x",
                            52: "H13x",
                            29: "C20x",
                            32: "O6x",
                            54: "H12x",
                            28: "C19x",
                            51: "H11x",
                            27: "C18x",
                            33: "O5x",
                            34: "C17x",
                            35: "C12x",
                            36: "C13x",
                            55: "H7x",
                            37: "C14x",
                            56: "H8x",
                            38: "C15x",
                            40: "O4x",
                            58: "H9x",
                            39: "C16x",
                            57: "H10x",
                        },
                        linking_bond=PEPTIDE_BOND,
                        leaving_atoms=leavers,
                        description="CYSTEINE-CONJUGATED FLUOROPHORE MALEIMIDE",
                    ),
                ],
            },
        ),
    )
    assert "DYE" in [res.identifier[3] for res in pablo_top.molecule(0).residues]

    substructure_mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
    for i, atom in enumerate(substructure_mol.atoms, start=1):
        if i in leavers:
            atom.metadata["substructure_atom"] = False
        else:
            atom.metadata["substructure_atom"] = True
    legacy_top: Topology = Topology.from_pdb(
        get_test_data_path("prepared_pdbs/3ip9_dye_solvated.pdb"),
        # _additional_substructures is a PROTOTYPE.
        # Its behavior and input type are likely to change.
        _additional_substructures=[substructure_mol],
    )

    assert pablo_top.n_molecules == legacy_top.n_molecules
    for pablo_mol, legacy_mol in zip(pablo_top.molecules, legacy_top.molecules):
        assert pablo_mol.n_atoms == legacy_mol.n_atoms
        for pablo_atom, legacy_atom in zip(pablo_mol.atoms, legacy_mol.atoms):
            assert (
                pablo_atom.name == legacy_atom.name
                or pablo_atom.metadata["canonical_name"] == legacy_atom.name
            )
            assert pablo_atom.symbol == legacy_atom.symbol
            for key in [
                "residue_name",
                "chain_id",
                "residue_number",
                "insertion_code",
            ]:
                assert pablo_atom.metadata[key] == legacy_atom.metadata[key]

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


@pytest.mark.xfail
def test_cannot_load_arg_alternate_resonance_form():
    """One day this will pass, but not just yet"""
    topology_from_pdb(
        get_test_data_path("capped_arg_altresonance.pdb"),
    )


@pytest.mark.xfail
def test_misplaced_ter_with_custom_resdef_gives_clear_error():
    """This needs a clearer error"""
    with pytest.raises(ValueError):
        topology_from_pdb(
            get_test_data_path("capped_ser_extrater.pdb"),
        )
