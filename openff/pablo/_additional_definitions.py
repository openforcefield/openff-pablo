from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Self

from openff.pablo._graph import Graph
from openff.pablo._pdb_data import PdbData

from ._matching import (
    PossibleResidueMatch,
    ResidueMatch,
    only_matched,
)
from .exceptions import (
    AmbiguousResidueMatch,
)
from .residue import AtomDefinition, BondDefinition, ResidueDefinition


def apply_additional_definitions(
    data: PdbData,
    matches: Iterable[Sequence[PossibleResidueMatch]],
    additional_definitions: Iterable[ResidueDefinition],
) -> "list[AdditionalDefMatch]":
    """
    Get a list of additional matches that can straddle different residues

    Raises
    ======
    AmbiguousResidueMatch
        If any residue has more than one successful match in ``matches``. This
        function can only fill in missing information, not adjudicate
        ambiguities. It is OK for a residue to have zero matches - that's what
        this function is for!
    ValueError
        If an additional definition can be mapped to an otherwise unknown atom
        in multiple chemically distinct ways.
    AssertionError
        Asserts are used only to verify certain local invariants. Any assert
        error in user code is a bug - please report it!
    """
    pdb_graph, atoms, bonds = _get_residue_graph(data, matches)

    new_matches: list[AdditionalDefMatch] = []
    for resdef in additional_definitions:
        # TODO: `try` block for better ambiguity handling
        new_matches.append(
            _apply_resdef_to_graph(
                data,
                resdef,
                pdb_graph,
                atoms,
                bonds,
            ),
        )

    return new_matches


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
) -> "AdditionalDefMatch":
    """Find a way to match this additional residue definition to the graph

    Raises
    ======
    ValueError
        If the residue definition can be mapped to an otherwise unknown atom
        in multiple chemically distinct ways.
    """

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

    mappings: list[AdditionalDefMatch] = []
    for resdef_graph in resdef._to_graphs():
        mappings.extend(
            AdditionalDefMatch.from_mapping(mapping, resdef)
            for mapping in pdb_graph.get_mappings(
                resdef_graph,
                node_matcher=node_matcher,
                edge_matcher=edge_matcher,
                subgraph=True,
                induced=True,
            )
            if (
                _has_valid_connectivity(
                    data,
                    resdef_graph,
                    pdb_graph,
                    mapping,
                )
                # Also require that this mapping would cover an unknown atom
                # TODO: Allow covering an unknown bond as well
                and any(atoms[i] is None for i in mapping)
            )
        )

    mapping = mappings.pop()
    for other_mapping in mappings:
        if not mapping.agrees_with(other_mapping):
            raise ValueError("resdef does not unambiguously map")

    return mapping


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


@dataclass(frozen=True)
class AdditionalDefMatch(ResidueMatch):
    @classmethod
    def from_mapping(
        cls,
        atom_mapping: dict[int, AtomDefinition],
        resdef: ResidueDefinition,
    ) -> Self:
        """Compute the bond mapping for a particular atom mapping"""
        matched_atoms = {
            atom.name: i for i, atom in atom_mapping.items() if atom.symbol != ""
        }
        neighbouring_atoms = {
            atom.name: i for i, atom in atom_mapping.items() if atom.symbol == ""
        }

        if resdef.prior_bond_leaving_atoms.isdisjoint(matched_atoms):
            prior_bond = (
                neighbouring_atoms[resdef.prior_bond_linking_atom],
                matched_atoms[resdef.posterior_bond_linking_atom],
            )
        elif resdef.prior_bond_leaving_atoms.issubset(matched_atoms):
            prior_bond = None
        else:
            assert False

        if resdef.posterior_bond_leaving_atoms.isdisjoint(matched_atoms):
            posterior_bond = (
                matched_atoms[resdef.prior_bond_linking_atom],
                neighbouring_atoms[resdef.posterior_bond_linking_atom],
            )
        elif resdef.posterior_bond_leaving_atoms.issubset(matched_atoms):
            posterior_bond = None
        else:
            assert False

        if resdef.crosslink_leaving_atoms.isdisjoint(matched_atoms):
            assert resdef.crosslink is not None
            crosslink = (
                matched_atoms[resdef.crosslink.atom1],
                neighbouring_atoms[resdef.crosslink.atom2],
            )
        elif resdef.crosslink_leaving_atoms.issubset(matched_atoms):
            crosslink = None
        else:
            assert False

        return cls(
            residue_definition=resdef,
            index_to_atomdef=atom_mapping,
            vsite_idcs=(),
            prior_bond_idcs=prior_bond,
            posterior_bond_idcs=posterior_bond,
            crosslink_idcs=crosslink,
        )
