from collections import defaultdict
from collections.abc import Mapping, Sequence

import pytest
from openff.toolkit import Molecule, Topology, unit

from openff.pablo import STD_CCD_CACHE
from openff.pablo._pdb import topology_from_pdb
from openff.pablo._tests.utils import get_test_data_path
from openff.pablo._utils import sort_tuple, unwrap
from openff.pablo.ccd._ccdcache import CcdCache
from openff.pablo.chem import PEPTIDE_BOND
from openff.pablo.exceptions import PdbResidueMatchError
from openff.pablo.residue import BondDefinition, ResidueDefinition


@pytest.mark.slow
def test_2zuq_cross_chain_disulfide_discontinuous(tmp_ccd_cache: CcdCache):
    pdbfn = get_test_data_path("prepared_pdbs/2zuq_prepared.pdb")
    with pytest.warns():
        topology = topology_from_pdb(pdbfn, residue_library=tmp_ccd_cache)

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


def test_2mum_neutralized_has_all_neutral_aas(
    all_aa_resnames: set[str],
    tmp_ccd_cache: CcdCache,
):
    pdbfn = get_test_data_path("prepared_pdbs/2MUM_neutralized.pdb")
    topology = topology_from_pdb(pdbfn, residue_library=tmp_ccd_cache)
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
def test_1p3q_loads_chains_without_ter(tmp_ccd_cache: CcdCache):
    pdbfn = get_test_data_path("prepared_pdbs/1p3q_noter.pdb")
    topology = topology_from_pdb(pdbfn, residue_library=tmp_ccd_cache)

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


def test_5eil_is_three_proteins_with_ncaa_plus_fe3_and_water(tmp_ccd_cache: CcdCache):
    tmp_ccd_cache.get_from_ccd("BP5")
    tmp_ccd_cache.get_from_ccd("FE")

    topology = topology_from_pdb(
        get_test_data_path("prepared_pdbs/5eil_fixed.pdb"),
        residue_library=tmp_ccd_cache,
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
def test_3ip9_loads_with_augmented_resdb(tmp_ccd_cache: CcdCache):
    smiles = "[c:1]1([H:41])[c:2]([H:42])[c:3]2[c:4]([c:5]([H:43])[c:6]1[N:7]1[C:8](=[O:9])[C:10]([H:44])([H:45])[C@@:11]([S:12][C:13]([C@:14]([N:15]([H:16])[H:50])([C:17](=[O:18])[O:19][H:59])[H:49])([H:47])[H:48])([H:46])[C:20]1=[O:21])[C:22](=[O:23])[O:24][C:25]21[c:26]2[c:27]([c:28]([H:51])[c:29]([O:32][H:54])[c:30]([H:52])[c:31]2[H:53])[O:33][c:34]2[c:35]1[c:36]([H:55])[c:37]([H:56])[c:38]([O:40][H:58])[c:39]2[H:57]"
    leavers = {16, 19, 59}
    pablo_top = topology_from_pdb(
        get_test_data_path("prepared_pdbs/3ip9_dye_solvated.pdb"),
        residue_library=tmp_ccd_cache.with_(
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
                assert str(pablo_atom.metadata[key]) == str(legacy_atom.metadata[key])

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


@pytest.mark.slow
def test_3ip9_loads_via_conects(tmp_ccd_cache: CcdCache):
    path = get_test_data_path("prepared_pdbs/3ip9_dye_solvated.pdb")
    smiles = "[c:1]1([H:41])[c:2]([H:42])[c:3]2[c:4]([c:5]([H:43])[c:6]1[N:7]1[C:8](=[O:9])[C:10]([H:44])([H:45])[C@@:11]([S:12][C:13]([C@:14]([N:15]([H:16])[H:50])([C:17](=[O:18])[O:19][H:59])[H:49])([H:47])[H:48])([H:46])[C:20]1=[O:21])[C:22](=[O:23])[O:24][C:25]21[c:26]2[c:27]([c:28]([H:51])[c:29]([O:32][H:54])[c:30]([H:52])[c:31]2[H:53])[O:33][c:34]2[c:35]1[c:36]([H:55])[c:37]([H:56])[c:38]([O:40][H:58])[c:39]2[H:57]"

    substructure_mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
    substructure_mol.generate_unique_atom_names()
    linking_bond = PEPTIDE_BOND
    for i, atom in enumerate(substructure_mol.atoms):
        if i == 14:
            atom.name = linking_bond.atom2
            # linking_bond = linking_bond.replace(atom2=atom.name)
        if i == 16:
            atom.name = linking_bond.atom1
            # linking_bond = linking_bond.replace(atom1=atom.name)
        if i in {15, 18, 58}:
            atom.metadata["substructure_atom"] = False
            atom.metadata["leaving_atom"] = True
        else:
            atom.metadata["substructure_atom"] = True
            atom.metadata["leaving_atom"] = False
    assert "CA" not in {atom.name for atom in substructure_mol.atoms}
    assert "H" not in {atom.name for atom in substructure_mol.atoms}

    pablo_top = topology_from_pdb(
        path,
        residue_library=tmp_ccd_cache.with_(
            [
                ResidueDefinition.from_molecule(
                    molecule=substructure_mol,
                    residue_name="DYE",
                    linking_bond=linking_bond,
                    description="CYSTEINE-CONJUGATED FLUOROPHORE MALEIMIDE",
                ),
            ],
        ),
    )
    assert "DYE" in [res.identifier[3] for res in pablo_top.molecule(0).residues]

    legacy_top: Topology = Topology.from_pdb(
        path,
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
                assert str(pablo_atom.metadata[key]) == str(legacy_atom.metadata[key])

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


def test_e2_loads_via_conects(tmp_ccd_cache: CcdCache):
    """Check that a file that consists of a single residue non-polymer molecule
    with the wrong atom names and correct CONECT records matches"""
    path = get_test_data_path("e2_7nel.pdb")
    substructure_mol = unwrap(tmp_ccd_cache.get_from_ccd("EST")).to_openff_molecule()
    for atom in substructure_mol.atoms:
        atom.name = ""
        atom.metadata["synonyms"] = ""
    old_names = [atom.name for atom in substructure_mol.atoms]
    substructure_mol.generate_unique_atom_names()
    assert all(
        atom.name != old_name
        for atom, old_name in zip(substructure_mol.atoms, old_names)
    )

    resdef = ResidueDefinition.from_molecule(substructure_mol)
    assert all(len(atom.synonyms) == 0 for atom in resdef.atoms)

    top = topology_from_pdb(
        path,
        residue_library={"EST": [resdef]},
    )
    assert top.n_molecules == 1
    top_mol = top.molecule(0)
    assert isinstance(top_mol, Molecule)
    is_isomorphic, mapping = Molecule.are_isomorphic(
        top_mol,
        substructure_mol,
        return_atom_map=True,
        atom_stereochemistry_matching=False,
        bond_stereochemistry_matching=False,
    )
    assert is_isomorphic
    assert mapping is not None
    for i, j in mapping.items():
        i_atom = top_mol.atom(i)
        j_atom = substructure_mol.atom(j)

        assert i_atom.metadata["used_synonym"] != j_atom.name
        assert i_atom.metadata["canonical_name"] == j_atom.name


def test_3ip9_trimmed_loads_via_conects_like_legacy(tmp_ccd_cache: CcdCache):
    path = get_test_data_path("3ip9_dye_trimmed.pdb")
    smiles = "[c:1]1([H:41])[c:2]([H:42])[c:3]2[c:4]([c:5]([H:43])[c:6]1[N:7]1[C:8](=[O:9])[C:10]([H:44])([H:45])[C@@:11]([S:12][C:13]([C@:14]([N:15]([H:16])[H:50])([C:17](=[O:18])[O:19][H:59])[H:49])([H:47])[H:48])([H:46])[C:20]1=[O:21])[C:22](=[O:23])[O:24][C:25]21[c:26]2[c:27]([c:28]([H:51])[c:29]([O:32][H:54])[c:30]([H:52])[c:31]2[H:53])[O:33][c:34]2[c:35]1[c:36]([H:55])[c:37]([H:56])[c:38]([O:40][H:58])[c:39]2[H:57]"

    substructure_mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
    substructure_mol.generate_unique_atom_names()
    linking_bond = PEPTIDE_BOND
    for i, atom in enumerate(substructure_mol.atoms):
        if i == 14:
            atom.name = linking_bond.atom2
            # linking_bond = linking_bond.replace(atom2=atom.name)
        if i == 16:
            atom.name = linking_bond.atom1
            # linking_bond = linking_bond.replace(atom1=atom.name)
        if i in {15, 18, 58}:
            atom.metadata["substructure_atom"] = False
            atom.metadata["leaving_atom"] = True
        else:
            atom.metadata["substructure_atom"] = True
            atom.metadata["leaving_atom"] = False
    assert "CA" not in {atom.name for atom in substructure_mol.atoms}
    assert "H" not in {atom.name for atom in substructure_mol.atoms}

    pablo_top = topology_from_pdb(
        path,
        residue_library=tmp_ccd_cache.with_(
            [
                ResidueDefinition.from_molecule(
                    molecule=substructure_mol,
                    residue_name="DYE",
                    linking_bond=linking_bond,
                    description="CYSTEINE-CONJUGATED FLUOROPHORE MALEIMIDE",
                ),
            ],
        ),
    )
    assert "DYE" in [res.identifier[3] for res in pablo_top.molecule(0).residues]

    legacy_top: Topology = Topology.from_pdb(
        path,
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
                assert str(pablo_atom.metadata[key]) == str(legacy_atom.metadata[key])

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


def test_3ip9_trimmed_loads_via_conects(tmp_ccd_cache: CcdCache):
    resdef = ResidueDefinition.from_smiles(
        mapped_smiles="[c:1]1([H:41])[c:2]([H:42])[c:3]2[c:4]([c:5]([H:43])[c:6]1[N:7]1[C:8](=[O:9])[C:10]([H:44])([H:45])[C@@:11]([S:12][C:13]([C@:14]([N:15]([H:16])[H:50])([C:17](=[O:18])[O:19][H:59])[H:49])([H:47])[H:48])([H:46])[C:20]1=[O:21])[C:22](=[O:23])[O:24][C:25]21[c:26]2[c:27]([c:28]([H:51])[c:29]([O:32][H:54])[c:30]([H:52])[c:31]2[H:53])[O:33][c:34]2[c:35]1[c:36]([H:55])[c:37]([H:56])[c:38]([O:40][H:58])[c:39]2[H:57]",
        leaving_atoms=[16, 19, 59],
        atom_names={**{i: str(i) for i in range(1, 60)}, 15: "N", 17: "C"},
        residue_name="DYE",
        linking_bond=PEPTIDE_BOND,
        description="CYSTEINE-CONJUGATED FLUOROPHORE MALEIMIDE",
    )

    pdb_path = get_test_data_path("3ip9_dye_trimmed.pdb")

    top = topology_from_pdb(
        pdb_path,
        residue_library=tmp_ccd_cache.with_([resdef]),
    )

    sdf_path = get_test_data_path("3ip9_dye_trimmed.sdf")
    ref_mol = Molecule.from_file(sdf_path, "SDF")
    if not isinstance(ref_mol, Molecule):
        ref_mol = unwrap(ref_mol)

    top_mol = unwrap(top.molecules)
    assert isinstance(top_mol, Molecule)

    assert top_mol.to_smiles() == ref_mol.to_smiles()
    assert top_mol.is_isomorphic_with(ref_mol)


def test_3ip9_trimmed_loads_via_dye_additional_definitions(tmp_ccd_cache: CcdCache):
    resdef = ResidueDefinition.from_smiles(
        mapped_smiles="[c:1]1([H:41])[c:2]([H:42])[c:3]2[c:4]([c:5]([H:43])[c:6]1[N:7]1[C:8](=[O:9])[C:10]([H:44])([H:45])[C@@:11]([S:12][C:13]([C@:14]([N:15]([H:16])[H:50])([C:17](=[O:18])[O:19][H:59])[H:49])([H:47])[H:48])([H:46])[C:20]1=[O:21])[C:22](=[O:23])[O:24][C:25]21[c:26]2[c:27]([c:28]([H:51])[c:29]([O:32][H:54])[c:30]([H:52])[c:31]2[H:53])[O:33][c:34]2[c:35]1[c:36]([H:55])[c:37]([H:56])[c:38]([O:40][H:58])[c:39]2[H:57]",
        leaving_atoms=[16, 19, 59],
        atom_names={**{i: str(i) for i in range(1, 60)}, 15: "N", 17: "C"},
        residue_name="DYE",
        linking_bond=PEPTIDE_BOND,
        description="CYSTEINE-CONJUGATED FLUOROPHORE MALEIMIDE",
    )

    pdb_path = get_test_data_path("3ip9_dye_trimmed.pdb")

    top = topology_from_pdb(
        pdb_path,
        residue_library=tmp_ccd_cache,
        additional_definitions=[resdef],
    )

    sdf_path = get_test_data_path("3ip9_dye_trimmed.sdf")
    ref_mol = Molecule.from_file(sdf_path, "SDF")
    if not isinstance(ref_mol, Molecule):
        ref_mol = unwrap(ref_mol)

    top_mol = unwrap(top.molecules)
    assert isinstance(top_mol, Molecule)

    assert top_mol.is_isomorphic_with(ref_mol)


@pytest.mark.parametrize(
    "filename",
    [
        "3ip9_dye_trimmed.pdb",
        # TODO: Implement this somehow
        pytest.param("3ip9_dye_trimmed_dyecys.pdb", marks=pytest.mark.xfail),
    ],
)
def test_3ip9_trimmed_loads_via_anon_dye_additional_definitions(
    filename: str,
    tmp_ccd_cache: CcdCache,
):
    resdef = ResidueDefinition.anon_from_smiles_marked_leaving(
        smiles="[H:1]N[C@@H](CS[C@H]1CC(=O)N(c2ccc3c(c2)C(=O)OC32c3ccc(O)cc3Oc3cc(O)ccc32)C1=O)C(=O)[O:2][H:3]",
        # smiles="[c]1([H])[c]([H])[c]2[c]([c]([H])[c]1[N]1[C](=[O])[C]([H])([H])[C@@]([S][C]([C@]([N]([H:16])[H])([C](=[O])[O:19][H:59])[H])([H])[H])([H])[C]1=[O])[C](=[O])[O][C]21[c]2[c]([c]([H])[c]([O][H])[c]([H])[c]2[H])[O][c]2[c]1[c]([H])[c]([H])[c]([O][H])[c]2[H]",
        description="CYSTEINE-CONJUGATED FLUOROPHORE MALEIMIDE",
    )

    pdb_path = get_test_data_path(filename)

    top = topology_from_pdb(
        pdb_path,
        residue_library=tmp_ccd_cache,
        additional_definitions=[resdef],
    )

    sdf_path = get_test_data_path("3ip9_dye_trimmed.sdf")
    ref_mol = Molecule.from_file(sdf_path, "SDF")
    if not isinstance(ref_mol, Molecule):
        ref_mol = unwrap(ref_mol)

    top_mol = unwrap(top.molecules)
    assert isinstance(top_mol, Molecule)

    assert top_mol.to_smiles() == ref_mol.to_smiles()
    assert top_mol.is_isomorphic_with(ref_mol)


def test_3ip9_trimmed_loads_via_anon_whole_molecule_additional_definitions(
    tmp_ccd_cache: CcdCache,
):
    pdb_path = get_test_data_path("3ip9_dye_trimmed.pdb")
    sdf_path = get_test_data_path("3ip9_dye_trimmed.sdf")

    resdef = ResidueDefinition.anon_from_sdf(sdf_path)

    top = topology_from_pdb(
        pdb_path,
        residue_library=tmp_ccd_cache,
        additional_definitions=[resdef],
    )

    ref_mol = Molecule.from_file(sdf_path, "SDF")
    if not isinstance(ref_mol, Molecule):
        ref_mol = unwrap(ref_mol)

    top_mol = unwrap(top.molecules)
    assert isinstance(top_mol, Molecule)

    assert top_mol.to_smiles() == ref_mol.to_smiles()
    assert top_mol.is_isomorphic_with(ref_mol)


def test_3ip9_trimmed_loads_via_whole_molecule_unique_molecules():
    pdb_path = get_test_data_path("3ip9_dye_trimmed.pdb")
    sdf_path = get_test_data_path("3ip9_dye_trimmed.sdf")

    mol = Molecule.from_file(sdf_path, "SDF")
    assert isinstance(mol, Molecule)
    top = Topology.from_pdb(pdb_path, unique_molecules=[mol])

    ref_mol = Molecule.from_file(sdf_path, "SDF")
    if not isinstance(ref_mol, Molecule):
        ref_mol = unwrap(ref_mol)

    top_mol = unwrap(top.molecules)
    assert isinstance(top_mol, Molecule)

    assert top_mol.to_smiles() == ref_mol.to_smiles()
    assert top_mol.is_isomorphic_with(ref_mol)


@pytest.mark.slow
def test_big_bilayer(tmp_ccd_cache: CcdCache):
    """
    Test file with 100k+ atoms w/ CONECT records and 4-letter residue names

    This tests that atoms with hexadecimal SERIAL entries work, even in CONECT
    records. It also checks that 4-letter residue names work and that
    hexadecimal residue numbers work.
    """
    top = topology_from_pdb(
        get_test_data_path("big_bilayer.pdb"),
        residue_library=tmp_ccd_cache.with_(
            {
                "DLPC": [
                    ResidueDefinition.from_smiles(
                        "[C:1]([C@@:2]([C:3]([O:36][C:21]([C:22]([C:23]([C:24]([C:25]([C:26]([C:27]([C:28]([C:29]([C:30]([C:31]([C:32]([H:104])([H:105])[H:106])([H:102])[H:103])([H:100])[H:101])([H:98])[H:99])([H:96])[H:97])([H:94])[H:95])([H:92])[H:93])([H:90])[H:91])([H:88])[H:89])([H:86])[H:87])([H:84])[H:85])=[O:34])([H:46])[H:47])([O:35][C:9]([C:10]([C:11]([C:12]([C:13]([C:14]([C:15]([C:16]([C:17]([C:18]([C:19]([C:20]([H:81])([H:82])[H:83])([H:79])[H:80])([H:77])[H:78])([H:75])[H:76])([H:73])[H:74])([H:71])[H:72])([H:69])[H:70])([H:67])[H:68])([H:65])[H:66])([H:63])[H:64])([H:61])[H:62])=[O:33])[H:45])([O:39][P@:42](=[O:37])([O-:38])[O:40][C:4]([C:5]([N+:41]([C:6]([H:52])([H:53])[H:54])([C:7]([H:55])([H:56])[H:57])[C:8]([H:58])([H:59])[H:60])([H:50])[H:51])([H:48])[H:49])([H:43])[H:44]",
                        residue_name="DLPC",
                        atom_names={
                            1: "C1",
                            2: "C2",
                            3: "C3",
                            4: "C11",
                            5: "C12",
                            6: "C13",
                            7: "C14",
                            8: "C15",
                            9: "C21",
                            10: "C22",
                            11: "C23",
                            12: "C24",
                            13: "C25",
                            14: "C26",
                            15: "C27",
                            16: "C28",
                            17: "C29",
                            18: "0C21",
                            19: "1C21",
                            20: "2C21",
                            21: "C31",
                            22: "C32",
                            23: "C33",
                            24: "C34",
                            25: "C35",
                            26: "C36",
                            27: "C37",
                            28: "C38",
                            29: "C39",
                            30: "0C31",
                            31: "1C31",
                            32: "2C31",
                            33: "O22",
                            34: "O32",
                            35: "O21",
                            36: "O31",
                            37: "O14",
                            38: "O13",
                            39: "O11",
                            40: "O12",
                            41: "N",
                            42: "P",
                            43: "HA",
                            44: "HB",
                            45: "HS",
                            46: "HY",
                            47: "HX",
                            48: "H11A",
                            49: "H11B",
                            50: "H12A",
                            51: "H12B",
                            52: "H13A",
                            53: "H13B",
                            54: "H13C",
                            55: "H14A",
                            56: "H14B",
                            57: "H14C",
                            58: "H15A",
                            59: "H15B",
                            60: "H15C",
                            61: "H2S",
                            62: "H2R",
                            63: "H3S",
                            64: "H3R",
                            65: "H4S",
                            66: "H4R",
                            67: "H5S",
                            68: "H5R",
                            69: "H6S",
                            70: "H6R",
                            71: "H7S",
                            72: "H7R",
                            73: "H8S",
                            74: "H8R",
                            75: "H9S",
                            76: "H9R",
                            77: "H10S",
                            78: "H10R",
                            79: "H11S",
                            80: "H11R",
                            81: "H12S",
                            82: "H12R",
                            83: "H12T",
                            84: "H2X",
                            85: "H2Y",
                            86: "H3X",
                            87: "H3Y",
                            88: "H4X",
                            89: "H4Y",
                            90: "H5X",
                            91: "H5Y",
                            92: "H6X",
                            93: "H6Y",
                            94: "H7X",
                            95: "H7Y",
                            96: "H8X",
                            97: "H8Y",
                            98: "H9X",
                            99: "H9Y",
                            100: "H10X",
                            101: "H10Y",
                            102: "H11X",
                            103: "H11Y",
                            104: "H12X",
                            105: "H12Y",
                            106: "H12Z",
                        },
                    ),
                ],
            },
        ),
    )

    # Quick, fast-fail checks
    assert top.n_atoms == 200468
    assert top.n_molecules == 31740

    # Checks that the chemistry is correct
    unique_molecules = sorted(
        top.unique_molecules,
        key=lambda mol: (mol.n_atoms, mol.hill_formula),
    )
    assert len(unique_molecules) > 4
    assert unique_molecules[0].hill_formula == "Cl"
    assert unique_molecules[1].hill_formula == "Na"
    assert unique_molecules[2].hill_formula == "H2O"
    # Remainder should be DLPC stereoisomers
    dlpc_mol = Molecule.from_smiles(
        "CCCCCCCCCCCC(=O)OC[C@H](COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCC",
        allow_undefined_stereo=True,
    )
    for mol in unique_molecules[3:]:
        assert mol.is_isomorphic_with(dlpc_mol, atom_stereochemistry_matching=False)

    # Check that correct residues are still around
    assert {residue.identifier[3] for residue in top.residues} == {
        "DLPC",
        "NA",
        "CL",
        "HOH",
    }


@pytest.mark.xfail
def test_cannot_load_arg_alternate_resonance_form(tmp_ccd_cache: CcdCache):
    """One day this will pass, but not just yet"""
    topology_from_pdb(
        get_test_data_path("capped_arg_altresonance.pdb"),
        residue_library=tmp_ccd_cache,
    )


def test_can_load_arg_alternate_resonance_form_with_conects(tmp_ccd_cache: CcdCache):
    """Polymer residue with non-standard resonance form and CONECT records"""
    topology_from_pdb(
        get_test_data_path("capped_arg_altresonance_conect.pdb"),
        residue_library=tmp_ccd_cache,
    )


def test_misplaced_ter_with_custom_resdef_gives_clear_error(tmp_ccd_cache: CcdCache):
    with pytest.raises(
        ValueError,
        match="didn't match: Bond to next residue expected but cannot form polymer bond across TER record",
    ):
        topology_from_pdb(
            get_test_data_path("capped_ser_extrater.pdb"),
            residue_library=tmp_ccd_cache,
        )


def test_unknown_residue_gives_clear_error(tmp_ccd_cache: CcdCache):
    path = get_test_data_path("5ap1_prepared.pdb").absolute()

    def check_err(err: ValueError) -> bool:
        expected_error = "\n".join(
            [
                "some residues could not be identified",
                "A topology cannot be created without chemical information for every",
                "atom and bond. The following residues present in PDB file",
                str(path),
                "could not be identified from the provided chemical library:",
                "  C:UNK#1 (l4980-5038): No residue definitions",
            ],
        )

        assert err.args[0] == expected_error
        return True

    with pytest.raises(
        ValueError,
        check=check_err,
    ):
        topology_from_pdb(path, residue_library=tmp_ccd_cache)


def test_unmatched_residues_give_clear_error(
    cys_def_deprotonated_sidechain: ResidueDefinition,
):
    path = get_test_data_path("3cu9_vicinal_disulfide.pdb").absolute()

    def check_err(err: ValueError) -> bool:
        expected_error = "\n".join(
            [
                "some residues could not be identified",
                "A topology cannot be created without chemical information for every",
                "atom and bond. The following residues present in PDB file",
                str(path),
                "could not be identified from the provided chemical library:",
                "  A:CYS#221 (l1-11): No matching residue definitions:",
                "    ╰─ CYSTEINE didn't match: found CONECT record that could not be matched with a bond",
                "",
                "  A:CYS#222 (l12-23): No matching residue definitions:",
                "    ╰─ CYSTEINE didn't match: found CONECT record that could not be matched with a bond",
            ],
        )

        assert err.args[0] == expected_error
        return err.args[0] == expected_error

    with pytest.raises(
        ValueError,
        check=check_err,
    ):
        topology_from_pdb(
            path,
            residue_library={
                "CYS": [cys_def_deprotonated_sidechain.replace(crosslink=None)],
            },
        )


def test_polyglycines_loads_with_augmented_ccd(tmp_ccd_cache: CcdCache):
    # Loading this PDB file with this augmented CCD cache will test all kinds
    # of residue-residue interface; see prepared_pdbs/polyglycines.py
    topology = topology_from_pdb(
        get_test_data_path("prepared_pdbs/polyglycines.pdb"),
        residue_library=tmp_ccd_cache.with_(
            [
                ResidueDefinition.from_smiles(
                    mapped_smiles="[N-:1]([H:2])[C:3]([H:4])([H:5])[C:6](=[O:7])[O:8][H:9]",
                    atom_names={
                        1: "N",
                        2: "H",
                        3: "CA",
                        4: "HA1",
                        5: "HA2",
                        6: "C",
                        7: "O",
                        8: "OXT",
                        9: "HXT",
                    },
                    residue_name="GLY",
                    leaving_atoms=(8, 9),
                    linking_bond=PEPTIDE_BOND,
                    description="GLYCINE w/ negative formal charge on N",
                ),
                ResidueDefinition.from_smiles(
                    mapped_smiles="[N:1]([H:2])([H:8])[C:3]([H:4])([H:5])[C-:6]=[O:7]",
                    atom_names={
                        1: "N",
                        2: "H",
                        3: "CA",
                        4: "HA1",
                        5: "HA2",
                        6: "C",
                        7: "O",
                        8: "H2",
                    },
                    residue_name="GLY",
                    leaving_atoms=(8,),
                    linking_bond=PEPTIDE_BOND,
                    description="GLYCINE w/ negative formal charge on C",
                ),
            ],
        ),
    )
    assert topology.n_molecules == 103

    triglycine = Molecule.from_mapped_smiles(
        "[N-:1]([H:2])[C:3]([H:4])([H:5])[C:6](=[O:7])"
        + "[N:8]([H:9])[C:10]([H:11])([H:12])[C:13](=[O:14])"
        + "[N:15]([H:16])[C:17]([H:18])([H:19])[C-:20](=[O:21])",
    )

    assert topology.molecule(0).is_isomorphic_with(triglycine)
    assert topology.molecule(1).is_isomorphic_with(triglycine)
    assert topology.molecule(2).is_isomorphic_with(triglycine)
    for i in range(3, 103):
        molecule = topology.molecule(i)
        assert molecule.n_atoms == 3
        assert molecule.hill_formula == "H2O"


@pytest.mark.parametrize(
    "resdb",
    [
        {**STD_CCD_CACHE.without({"PRO"})},
        {**STD_CCD_CACHE.without({"GLY"})},
        {},
    ],
    ids=[
        "no_pro",
        "no_gly",
        "empty",
    ],
)
@pytest.mark.parametrize(
    "filename",
    [
        pytest.param("5ap1_prepared.pdb", marks=pytest.mark.slow),
        "5ap1_nosol.pdb",
    ],
)
def test_5ap1_fails_with_broken_resdefs(
    resdb: Mapping[str, Sequence[ResidueDefinition]],
    filename: str,
):
    path = get_test_data_path(filename).absolute()

    with pytest.raises(PdbResidueMatchError):
        topology_from_pdb(
            path,
            residue_library=resdb,
            additional_definitions=[
                ResidueDefinition.anon_from_smiles(
                    "O=C([O-])Cn1cc(cn1)c2ccc(cc2OCC#N)Nc3ccc(c(n3)NC4CCCCC4)C#N",
                ),
            ],
        )


def test_microviridin_crosslinks(tmp_ccd_cache: CcdCache):
    from openff.pablo import topology_from_pdb

    custom_residue_database = tmp_ccd_cache.with_(
        {
            "TH4": [
                resdef.replace(
                    residue_name="TH4",
                    crosslink=BondDefinition.with_defaults(atom1="OG1", atom2="CG"),
                    atoms=[
                        atom.replace(leaving=True) if atom.name == "HG1" else atom
                        for atom in resdef.atoms
                    ],
                    description=resdef.description + " w/ crosslink",
                )
                for resdef in tmp_ccd_cache["THR"]
                if resdef.description == "THREONINE"
            ],
            "A10": [
                resdef.replace(
                    residue_name="A10",
                    crosslink=BondDefinition.with_defaults(atom1="CG", atom2="OG1"),
                    atoms=[
                        (
                            atom.replace(leaving=True)
                            if atom.name in {"OD2", "HD2"}
                            else atom
                        )
                        for atom in resdef.atoms
                    ],
                    description=resdef.description + " w/ crosslink",
                )
                for resdef in tmp_ccd_cache["ASP"]
                if resdef.description == "ASPARTIC ACID"
            ],
            "LY6": [
                resdef.replace(
                    residue_name="LY6",
                    crosslink=BondDefinition.with_defaults(atom1="NZ", atom2="CD"),
                    atoms=[
                        (
                            atom.replace(leaving=True)
                            if atom.name == "HZ2"
                            else (
                                atom.replace(synonyms=(*atom.synonyms, "HZ"))
                                if atom.name == "HZ1"
                                else atom
                            )
                        )
                        for atom in resdef.atoms
                    ],
                    description=resdef.description + " w/ crosslink",
                )
                for resdef in tmp_ccd_cache["LYS"]
                if resdef.description == "LYSINE -HZ3"
            ],
            "G13": [
                resdef.replace(
                    residue_name="G13",
                    crosslink=BondDefinition.with_defaults(atom1="CD", atom2="NZ"),
                    atoms=[
                        atom.replace(leaving=True)
                        if atom.name in ("OE2", "HE2")
                        else atom
                        for atom in resdef.atoms
                    ],
                    description=resdef.description + " w/ crosslink",
                )
                for resdef in tmp_ccd_cache["GLU"]
                if resdef.description == "GLUTAMIC ACID"
            ],
            "SE9": [
                resdef.replace(
                    residue_name="SE9",
                    crosslink=BondDefinition.with_defaults(atom1="OG", atom2="CD"),
                    atoms=[
                        atom.replace(leaving=True) if atom.name == "HG" else atom
                        for atom in resdef.atoms
                    ],
                    description=resdef.description + " w/ crosslink",
                )
                for resdef in tmp_ccd_cache["SER"]
                if resdef.description == "SERINE"
            ],
            "G12": [
                resdef.replace(
                    residue_name="G12",
                    crosslink=BondDefinition.with_defaults(atom1="CD", atom2="OG"),
                    atoms=[
                        atom.replace(leaving=True)
                        if atom.name in ("OE2", "HE2")
                        else atom
                        for atom in resdef.atoms
                    ],
                    description=resdef.description + " w/ crosslink",
                )
                for resdef in tmp_ccd_cache["GLU"]
                if resdef.description == "GLUTAMIC ACID"
            ],
        },
    )

    topology_from_pdb(
        get_test_data_path("microviridin_edited.pdb").absolute(),
        residue_library=custom_residue_database,
    )


@pytest.mark.slow
def test_complex_pdb_1flr(tmp_ccd_cache: CcdCache):
    ligand = ResidueDefinition.anon_from_sdf(
        get_test_data_path("prepared_pdbs/1FLR_Ligand.sdf"),
    )

    with pytest.warns(
        UserWarning,
        match="Alt locs not supported; only empty or 'A' alt locs will be read",
    ):
        topology = topology_from_pdb(
            get_test_data_path("prepared_pdbs/1FLR_prepared.pdb"),
            residue_library=tmp_ccd_cache.with_patch(
                "THR",
                lambda resdef: [resdef.with_synonyms({"OXT": ["O2"]})],
            ),
            additional_definitions=[ligand],
        )

    uniques = {}
    for mol in topology.molecules:
        assert isinstance(mol, Molecule)
        if mol.n_atoms > 3:
            assert mol.hill_formula not in uniques.keys()
            uniques[mol.hill_formula] = mol
        else:
            if mol.to_smiles in uniques.keys():
                pass
            else:
                uniques[mol.to_smiles()] = mol

    chains: defaultdict[int | str, list[int]] = defaultdict(list)
    for i, atom in enumerate(topology.atoms):
        chains[atom.metadata["chain_id"]].append(i)

    # Hardcorded from match counts of regexes:
    #   (ATOM  |HETATM).{10}(A| )
    #   (ATOM  |HETATM).{10}(A| ).{4}H
    #   (ATOM  |HETATM).{10}(A| ).{4}L
    assert topology.n_atoms == 7539
    assert {k: len(v) for k, v in chains.items()} == {"H": 3712, "L": 3827}


# TODO: Debug this
@pytest.mark.xfail
@pytest.mark.slow
def test_7yv1_using_sdf(tmp_ccd_cache: CcdCache):
    ligand = ResidueDefinition.anon_from_sdf(
        get_test_data_path("7yv1/7yv1_ligand.sdf"),
    )

    with pytest.warns(UserWarning, match="Alt locs not supported"):
        _topology = topology_from_pdb(
            file=get_test_data_path("7yv1/7yv1_prepped.pdb"),
            residue_library=tmp_ccd_cache,
            additional_definitions=[ligand],
        )


def test_7yv1_peptide_using_sdf(tmp_ccd_cache: CcdCache):
    ligand = ResidueDefinition.anon_from_sdf(
        get_test_data_path("7yv1/7yv1_ligand.sdf"),
    )

    _topology = topology_from_pdb(
        file=get_test_data_path("7yv1/7yv1_prepped_cyclicpeptide.pdb"),
        residue_library=tmp_ccd_cache,
        additional_definitions=[ligand],
    )


# TODO: Test that the correct atom metadata are written out (as documented in topology_from_pdb())
