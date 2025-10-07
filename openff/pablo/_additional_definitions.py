from collections.abc import Iterable, Mapping, Sequence

from openff.pablo._graph import Graph
from openff.pablo._pdb_data import PdbData

from ._matching import (
    PossibleResidueMatch,
    only_matched,
)
from ._utils import (
    no_none_in_values,
)
from .exceptions import (
    AmbiguousResidueMatch,
)
from .residue import AtomDefinition, BondDefinition, ResidueDefinition


def apply_additional_definitions(
    data: PdbData,
    matches: Iterable[Sequence[PossibleResidueMatch]],
    additional_definitions: Iterable[ResidueDefinition],
) -> tuple[Mapping[int, AtomDefinition], Mapping[tuple[int, int], BondDefinition]]:
    pdb_graph, atoms, bonds = _get_residue_graph(data, matches)

    for resdef in additional_definitions:
        new_atoms: dict[int, AtomDefinition]
        new_bonds: dict[tuple[int, int], BondDefinition]
        # TODO: `try` block
        new_atoms, new_bonds = _apply_resdef_to_graph(
            data,
            resdef,
            pdb_graph,
            atoms,
            bonds,
        )
        atoms.update(new_atoms)
        bonds.update(new_bonds)

    if not (no_none_in_values(atoms) and no_none_in_values(bonds)):
        raise ValueError(
            "additional_definitions did not cover all unknown chemistries",
        )

    return (atoms, bonds)


def _get_residue_graph(
    data: PdbData,
    matches: Iterable[Sequence[PossibleResidueMatch]],
) -> tuple[
    Graph[int, tuple[int, int]],
    dict[int, AtomDefinition | None],
    dict[tuple[int, int], BondDefinition | None],
]:
    """
    Compute a graph of assigned and expected chemical information
    """
    graph: Graph[int, tuple[int, int]] = Graph(
        node_count_hint=len(data.name),
    )
    atoms: dict[int, AtomDefinition | None] = {}
    bonds: dict[tuple[int, int], BondDefinition | None] = {
        (i, j): None for i, js in enumerate(data.conects) for j in js if i < j
    }

    for possible_matches in matches:
        successful_matches = list(only_matched(possible_matches))
        match successful_matches:
            case []:
                res_atom_idcs = possible_matches[0].res_atom_idcs
                graph.add_nodes_from(res_atom_idcs)
                atoms.update({i: None for i in res_atom_idcs})
            case [match, *redundant_matches] if all(
                other_match.agrees_with(match) for other_match in redundant_matches
            ):
                graph.add_nodes_from(match.res_atom_idcs)
                atoms.update(match.index_to_atomdef.items())
                new_bonds = match.get_sorted_bond_map()
                for (i, j), bond in new_bonds.items():
                    assert i < j, f"{i} >= {j}"
                    if (i, j) in bonds and bonds[(i, j)] is not None:
                        assert bonds[i, j] == bond, (
                            f"incompatible matches: {(i, j)=}: {bonds[i, j]=} != {bond}"
                        )
                    bonds[i, j] = bond
            case _:
                raise AmbiguousResidueMatch(data, successful_matches)

    graph.add_edges_from((i, j, (i, j)) for (i, j) in bonds)

    return (graph, atoms, bonds)


def _apply_resdef_to_graph(
    data: PdbData,
    resdef: ResidueDefinition,
    pdb_graph: Graph[int, tuple[int, int]],
    atoms: dict[int, AtomDefinition | None],
    bonds: dict[tuple[int, int], BondDefinition | None],
) -> tuple[dict[int, AtomDefinition], dict[tuple[int, int], BondDefinition]]:
    def node_matcher(pdb_idx: int, new_atomdef: AtomDefinition) -> bool:
        old_atomdef = atoms[pdb_idx]
        if old_atomdef is None:
            return True
        return new_atomdef.symbol == "" or (
            (old_atomdef.symbol == new_atomdef.symbol)
            and (old_atomdef.charge == new_atomdef.charge)
        )

    def edge_matcher(
        pdb_idcs: tuple[int, int],
        new_bonddef: BondDefinition,
    ) -> bool:
        old_bonddef = bonds[pdb_idcs]
        if old_bonddef is None:
            return True
        return old_bonddef.order == new_bonddef.order

    mappings: list[
        tuple[dict[int, AtomDefinition], dict[tuple[int, int], BondDefinition]]
    ] = []
    for resdef_graph in resdef._to_graphs():
        mappings.extend(
            (mapping, _get_bond_mapping(mapping, resdef_graph))
            for mapping in pdb_graph.get_mappings(
                resdef_graph,
                node_matcher=node_matcher,
                edge_matcher=edge_matcher,
                subgraph=True,
                induced=True,
            )
            if _has_valid_connectivity(
                data,
                resdef_graph,
                pdb_graph,
                mapping,
            )
        )

    mapping = mappings.pop()
    for other_mapping in mappings:
        if not _are_chemically_equivalent(
            mapping,
            other_mapping,
        ):
            raise ValueError("resdef does not unambiguously map")

    return mapping


def _get_bond_mapping(
    atom_mapping: dict[int, AtomDefinition],
    resdef_graph: Graph[AtomDefinition, BondDefinition],
) -> dict[tuple[int, int], BondDefinition]:
    """Compute the bond mapping for a particular atom mapping"""
    name_to_idx = {v.name: k for k, v in atom_mapping.items()}
    bond_mapping: dict[tuple[int, int], BondDefinition] = {}
    for bond in resdef_graph.edges:
        atom1_idx = name_to_idx[bond.atom1]
        atom2_idx = name_to_idx[bond.atom2]
        bond_mapping[atom1_idx, atom2_idx] = bond
    return bond_mapping


def _has_valid_connectivity(
    data: PdbData,
    resdef_graph: Graph[AtomDefinition, BondDefinition],
    pdb_graph: Graph[int, tuple[int, int]],
    mapping: dict[int, AtomDefinition],
) -> bool:
    """
    True if ``mapping`` could be valid chemical information for ``pdb_graph``.

    "Valid" means all of the following must be true:

    1. Any atom whose elemental symbol in ``resdef_graph`` is not the empty
    string has the same number of neighbours in both graphs
    2. Any atom whose elemental symbol in ``resdef_graph`` is the empty
    string has a name in the PDB file that is either the canonical name
    of the atom definition or one of its synonyms.
    """
    for pdb_idx, resdef_atom in mapping.items():
        if resdef_atom.symbol == "":
            if data.name[pdb_idx] not in resdef_atom.names:
                return False
            continue

        n_resdef_neighbours = sum(1 for _ in resdef_graph.neighbours(resdef_atom))
        n_pdb_neighbours = sum(1 for _ in pdb_graph.neighbours(pdb_idx))
        if n_resdef_neighbours != n_pdb_neighbours:
            return False

    return True


def _are_chemically_equivalent(
    mapping_a: tuple[
        dict[int, AtomDefinition],
        dict[tuple[int, int], BondDefinition],
    ],
    mapping_b: tuple[
        dict[int, AtomDefinition],
        dict[tuple[int, int], BondDefinition],
    ],
) -> bool:
    atoms_a, bonds_a = mapping_a
    atoms_b, bonds_b = mapping_b

    if len(atoms_a.keys()) != len(atoms_b.keys()):
        return False
    if len(bonds_a.keys()) != len(bonds_b.keys()):
        return False

    charge_a, charge_b = 0, 0
    for i, atom_a in atoms_a.items():
        atom_b = atoms_b[i]
        if atom_a.symbol != atom_b.symbol:
            return False
        charge_a += atom_a.charge
        charge_b += atom_b.charge

    if charge_a != charge_b:
        return False

    return True
