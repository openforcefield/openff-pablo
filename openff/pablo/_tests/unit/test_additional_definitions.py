from openff.pablo._additional_definitions import _get_residue_graph
from openff.pablo._matching import (
    NoResidueDefinitions,
)
from openff.pablo._pdb_data import PdbData, ResidueMatch
from openff.pablo.ccd import CCD_RESIDUE_DEFINITION_CACHE
from openff.pablo.residue import AtomDefinition, BondDefinition, ResidueDefinition


def test_get_residue_map_creates_complete_single_residue_graph(
    cys_match: ResidueMatch,
    cys_data: PdbData,
):
    graph, atoms, bonds = _get_residue_graph(cys_data, [[cys_match]])
    assert len(atoms) == graph.n_nodes
    assert len(bonds) == graph.n_edges
    assert not any(atom is None for atom in atoms.values())
    assert not any(bond is None for bond in bonds.values())
    assert set(atoms.values()) == set(cys_match.residue_definition.atoms)
    assert set(bonds.values()) == set(cys_match.residue_definition.bonds)


def test_get_residue_map_mixed():
    data = PdbData(conects=[{1}, {0}])
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
    mismatch = NoResidueDefinitions("BBB", {1: None})

    graph, atoms, bonds = _get_residue_graph(data, [[match], [mismatch]])

    assert graph.n_nodes == 2
    assert graph.n_edges == 1
    assert atoms == {0: resdef.atoms[0], 1: None}
    assert bonds == {(0, 1): resdef.linking_bond}


def test_get_residue_map_vicinal_disulfide(vicinal_disulfide_data: PdbData):
    matches = vicinal_disulfide_data.match_residues(CCD_RESIDUE_DEFINITION_CACHE, ())
    graph, atoms, bonds = _get_residue_graph(vicinal_disulfide_data, matches)
    assert len(atoms) == graph.n_nodes
    assert len(bonds) == graph.n_edges
    assert not any(atom is None for atom in atoms.values())
    assert not any(bond is None for bond in bonds.values())
    assert graph.is_connected()
