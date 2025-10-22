from openff.toolkit import Molecule

from openff.pablo._additional_definitions import _get_pdb_graph
from openff.pablo._matching import SuccessfulMatch, unwrap_successful
from openff.pablo._pdb_data import PdbData, ResidueMatch
from openff.pablo._std_ccd_cache import STD_CCD_CACHE
from openff.pablo._tests.utils import get_test_data_path
from openff.pablo.chem import PEPTIDE_BOND
from openff.pablo.residue import AtomDefinition, BondDefinition, ResidueDefinition


def test_get_pdb_graph_creates_complete_single_residue_graph(
    cys_match: ResidueMatch,
    cys_data: PdbData,
):
    graph, atoms, bonds = _get_pdb_graph(cys_data, [cys_match])
    assert len(atoms) == graph.n_nodes
    assert len(bonds) == graph.n_edges
    assert not any(atom is None for atom in atoms.values())
    assert not any(bond is None for bond in bonds.values())
    assert set(atoms.values()) == set(cys_match.residue_definition.atoms)
    assert set(bonds.values()) == set(cys_match.residue_definition.bonds)


def test_get_pdb_graph_partial():
    data = PdbData(conects=[{1}, {0}], model=[None, None])
    resdef = ResidueDefinition(
        residue_name="AAA",
        description="demo",
        atoms=(AtomDefinition.with_defaults("A", "A"),),
        bonds=(),
        virtual_sites=(),
        crosslink=None,
        linking_bond=BondDefinition.with_defaults("A", "B"),
    )
    match = ResidueMatch(
        residue_definition=resdef,
        index_to_atomdef={0: resdef.atoms[0]},
        vsite_idcs=(),
        prior_bond_idcs=None,
        posterior_bond_idcs=(0, 1),
        crosslink_idcs=None,
    )

    graph, atoms, bonds = _get_pdb_graph(data, [match])

    assert graph.n_nodes == 2
    assert graph.n_edges == 1
    assert atoms == {0: resdef.atoms[0], 1: None}
    assert bonds == {(0, 1): resdef.linking_bond}


def test_get_pdb_graph_vicinal_disulfide(
    vicinal_disulfide_data: PdbData,
    vicinal_disulfide_molecule: Molecule,
):
    matches: list[SuccessfulMatch] = []
    for match in vicinal_disulfide_data.match_residues(
        STD_CCD_CACHE,
        (),
    ):
        try:
            success = unwrap_successful(match)
        except ValueError:
            continue
        matches.append(success)

    graph, atoms, bonds = _get_pdb_graph(vicinal_disulfide_data, matches)
    assert len(atoms) == graph.n_nodes
    assert len(atoms) == vicinal_disulfide_molecule.n_atoms
    assert len(bonds) == graph.n_edges
    assert len(bonds) == vicinal_disulfide_molecule.n_bonds
    assert not any(bond is None for bond in bonds.values())
    assert graph.is_connected()

    for i, atom in enumerate(vicinal_disulfide_molecule.atoms):
        atomdef = atoms[i]
        assert atomdef is not None
        assert atomdef.symbol == atom.symbol
        assert atomdef.charge == atom.formal_charge.m


def test_get_pdb_graph_3ip9_trimmed_partial():
    """
    VAL-NCAA-ASP PDB file with canonical residues provided as matches vs SDF
    file with intended chemistry. NCAA has CONECT records.
    """
    pdb_path = get_test_data_path("3ip9_dye_trimmed.pdb")
    sdf_path = get_test_data_path("3ip9_dye_trimmed.sdf")

    mol = Molecule.from_file(sdf_path, "SDF")
    assert isinstance(mol, Molecule)
    data = PdbData.from_file(pdb_path)

    matches: list[SuccessfulMatch] = []
    for match in data.match_residues(
        STD_CCD_CACHE,
        (),
    ):
        try:
            success = unwrap_successful(match)
        except ValueError:
            continue
        matches.append(success)
    assert len(matches) == 2
    assert ["VAL", "ASP"] == [
        match.residue_definition.residue_name for match in matches
    ]

    graph, atoms, bonds = _get_pdb_graph(data, matches)
    assert len(atoms) == graph.n_nodes
    assert len(atoms) == mol.n_atoms
    assert len(bonds) == graph.n_edges
    assert len(bonds) == mol.n_bonds
    assert graph.is_connected()

    for i, atom in enumerate(mol.atoms):
        atomdef = atoms[i]
        if atomdef is not None:
            assert atomdef.symbol == atom.symbol
            assert atomdef.charge == atom.formal_charge.m


def test_get_pdb_graph_3ip9_trimmed_complete():
    """
    VAL-NCAA-ASP PDB file with all three residues provided as matches vs SDF file
    with intended chemistry. NCAA has CONECT records.
    """
    pdb_path = get_test_data_path("3ip9_dye_trimmed.pdb")
    sdf_path = get_test_data_path("3ip9_dye_trimmed.sdf")

    mol = Molecule.from_file(sdf_path, "SDF")
    assert isinstance(mol, Molecule)
    data = PdbData.from_file(pdb_path)

    smiles = "[c:1]1([H:41])[c:2]([H:42])[c:3]2[c:4]([c:5]([H:43])[c:6]1[N:7]1[C:8](=[O:9])[C:10]([H:44])([H:45])[C@@:11]([S:12][C:13]([C@:14]([N:15]([H:16])[H:50])([C:17](=[O:18])[O:19][H:59])[H:49])([H:47])[H:48])([H:46])[C:20]1=[O:21])[C:22](=[O:23])[O:24][C:25]21[c:26]2[c:27]([c:28]([H:51])[c:29]([O:32][H:54])[c:30]([H:52])[c:31]2[H:53])[O:33][c:34]2[c:35]1[c:36]([H:55])[c:37]([H:56])[c:38]([O:40][H:58])[c:39]2[H:57]"

    resdef = ResidueDefinition.from_smiles(
        mapped_smiles=smiles,
        atom_names={**{i + 1: str(i + 1) for i in range(59)}, 15: "N", 17: "C"},
        residue_name="DYE",
        linking_bond=PEPTIDE_BOND,
        leaving_atoms=[16, 19, 59],
        description="CYSTEINE-CONJUGATED FLUOROPHORE MALEIMIDE",
    )

    matches: list[SuccessfulMatch] = []
    for match in data.match_residues(
        STD_CCD_CACHE.with_([resdef]),
        (),
    ):
        try:
            success = unwrap_successful(match)
        except ValueError:
            continue
        matches.append(success)
    assert len(matches) == 3
    assert ["VAL", "DYE", "ASP"] == [
        match.residue_definition.residue_name for match in matches
    ]

    graph, atoms, bonds = _get_pdb_graph(data, matches)
    assert len(atoms) == graph.n_nodes
    assert len(atoms) == mol.n_atoms
    assert len(bonds) == graph.n_edges
    assert len(bonds) == mol.n_bonds
    assert graph.is_connected()

    for i, atom in enumerate(mol.atoms):
        atomdef = atoms[i]
        if atomdef is not None:
            assert atomdef.symbol == atom.symbol
            assert atomdef.charge == atom.formal_charge.m
