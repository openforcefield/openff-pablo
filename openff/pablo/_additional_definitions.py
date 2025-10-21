import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from openff.pablo._graph import Graph
from openff.pablo._utils import flatten, sort_tuple

from ._matching import (
    ResidueMatch,
    SuccessfulMatch,
)
from .residue import AtomDefinition, BondDefinition, ResidueDefinition

if TYPE_CHECKING:
    from openff.pablo._pdb_data import PdbData


def apply_additional_definitions(
    data: "PdbData",
    matches: Sequence[SuccessfulMatch],
    additional_definitions: Iterable[ResidueDefinition],
) -> "list[AdditionalDefMatch]":
    """
    Get a list of additional matches that can straddle different residues

    Raises
    ======
    ValueError
        If an additional definition can be mapped to an otherwise unknown atom
        in multiple chemically distinct ways.
    AssertionError
        Asserts are used only to verify certain local invariants that cannot be
        verified statically by the type checker. Any assert error in user code
        is a bug and may indicate that results are incorrect - please report it!
    """
    logging.debug("Generating graph of known info")
    pdb_graph, atoms, bonds = _get_pdb_graph(data, matches)
    logging.debug(
        "Generated graph with"
        + f" {len(atoms)} atoms"
        + f" ({sum(1 for atom in atoms.values() if atom is None)} unknown),"
        + f" {len(bonds)} bonds"
        + f" ({sum(1 for bond in bonds.values() if bond is None)} unknown)",
    )

    new_matches: list[AdditionalDefMatch] = []
    for resdef in additional_definitions:
        logging.debug(f"checking {resdef.description}")
        match = _apply_resdef_to_graph(
            data,
            resdef,
            pdb_graph,
            atoms,
            bonds,
        )
        if match is None:
            logging.debug("  no matches")
            continue
        new_matches.append(
            match,
        )

    return new_matches


def _get_pdb_graph(
    data: "PdbData",
    matches: Sequence[SuccessfulMatch],
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
    # Get a set of all the atom record indices - we'll filter out vsites as we go
    non_vsite_idcs: set[int] = set(range(data.n_atoms))
    # Initialize all atoms referenced by CONECT records as None (unknown chemistry)
    atoms: dict[int, AtomDefinition | None] = {
        **{i: None for i, _ in enumerate(data.conects)},
        **{i: None for i in flatten(data.conects)},
    }
    graph.add_nodes_from(atoms)
    # Initialize all bonds referenced by CONECT records as None (unknown chemistry)
    bonds: dict[tuple[int, int], BondDefinition | None] = {
        sort_tuple((i, j)): None for i, js in enumerate(data.conects) for j in js
    }
    graph.add_edges_from((i, j, (i, j)) for i, j in bonds)

    # Add chemical information from successful matches
    for match in matches:
        # Add the nodes (or at least those that haven't been added yet)
        graph.add_nodes_from(match.res_atom_idcs, skip_existing=True)
        # Update atomic chemical info
        atoms.update(match.index_to_atomdef.items())
        # Remove virtual sites from the list of atom idcs we'll add later
        for i in match.vsite_idcs:
            assert i not in atoms, (
                "Conect records to virtual sites should have already been filtered out"
            )
            non_vsite_idcs.remove(i)
        # Update bond chemical info
        new_bonds = match.get_sorted_bond_map()
        for (i, j), bond in new_bonds.items():
            assert i < j, f"{i} >= {j}"
            if (i, j) in bonds and bonds[(i, j)] is not None:
                assert bonds[i, j] == bond, (
                    f"incompatible matches: {(i, j)=}: {bonds[i, j]=} != {bond}"
                )
            graph.add_nodes_from([i, j], skip_existing=True)
            graph.add_edge(i, j, (i, j))
            bonds[i, j] = bond

    # Add any remaining atoms that were not virtual sites (eg, those not
    # referenced in matches or conect records)
    graph.add_nodes_from(non_vsite_idcs, skip_existing=True)

    return (graph, atoms, bonds)


def _apply_resdef_to_graph(
    data: "PdbData",
    resdef: ResidueDefinition,
    pdb_graph: Graph[int, tuple[int, int]],
    atoms: dict[int, AtomDefinition | None],
    bonds: dict[tuple[int, int], BondDefinition | None],
) -> "AdditionalDefMatch | None":
    """Find a way to match this additional residue definition to the graph

    Raises
    ======
    ValueError
        If the residue definition can be mapped to an otherwise unknown atom
        in multiple chemically distinct ways.
    """

    def node_matcher(pdb_idx: int, new_atomdef: AtomDefinition) -> bool:
        old_atomdef = atoms[pdb_idx]
        # If the new atom is a non-matching atom, then it matches any node
        if new_atomdef.symbol == "":
            return True
        # If this atom is unidentified, check the element matches the PDB file
        if old_atomdef is None:
            return new_atomdef.symbol == data.element[pdb_idx]
        # Otherwise, check that the chemical information matches
        # between PDB file, old chem info, and new chem info
        return (  # nofmt
            data.element[pdb_idx] == new_atomdef.symbol
            and old_atomdef.symbol == new_atomdef.symbol
            and old_atomdef.charge == new_atomdef.charge
        )

    def edge_matcher(
        pdb_idcs: tuple[int, int],
        new_bonddef: BondDefinition,
    ) -> bool:
        old_bonddef = bonds[pdb_idcs]
        # If this bond is unidentified, accept the new chemical information
        if old_bonddef is None:
            return True
        # Otherwise, accept this match if both assign the same chemical information
        return old_bonddef.order == new_bonddef.order

    mappings: list[AdditionalDefMatch] = []
    resdef_graphs = list(resdef._to_graphs())
    logging.debug(f"{resdef.description} has {len(resdef_graphs)} graphs")
    for resdef_graph in resdef_graphs:
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
    logging.debug(f"Found {len(mappings)} mappings to {resdef.description}")

    if len(mappings) == 0:
        return None

    mapping = mappings.pop()
    for other_mapping in mappings:
        if not mapping.agrees_with(other_mapping):
            raise ValueError("resdef does not unambiguously map")

    return mapping


def _has_valid_connectivity(
    data: "PdbData",
    resdef_graph: Graph[AtomDefinition, BondDefinition],
    pdb_graph: Graph[int, tuple[int, int]],
    mapping: dict[int, AtomDefinition],
) -> bool:
    """
    True if ``mapping`` could be valid chemical information for ``pdb_graph``.

    "Valid" means all of the following must be true:

    1. Any atom whose elemental symbol in ``resdef_graph`` is not the empty
    string has the same number of neighbours in both graphs
    2. Any atom whose (a) elemental symbol in ``resdef_graph`` is the empty
    string and whose (b) name does not begin with ``"NON_MATCHING_ATOM_"`` has a
    name in the PDB file that is either the canonical name of the atom
    definition or one of its synonyms.
    3. All atoms have the same number of neighbours in the PDB graph as the
    residue definition

    """
    for pdb_idx, resdef_atom in mapping.items():
        if (  # nofmt
            resdef_atom.symbol == ""
            and resdef_atom.name.startswith("NON_MATCHING_ATOM_")
        ):
            continue

        if resdef_atom.symbol == "":
            if data.name[pdb_idx] not in resdef_atom.names:
                logging.debug(
                    f"  MAPPING INVALID: invalid name {data.name[pdb_idx]} in PDB for {resdef_atom}",
                )
                return False
            continue

        n_resdef_neighbours = sum(1 for _ in resdef_graph.neighbours(resdef_atom))
        n_pdb_neighbours = sum(1 for _ in pdb_graph.neighbours(pdb_idx))
        if n_resdef_neighbours != n_pdb_neighbours:
            logging.debug(
                f"  MAPPING INVALID: {n_pdb_neighbours=} but {n_resdef_neighbours=} for {resdef_atom}",
            )
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
        neighbouring_atoms: dict[str, int] = {}
        matched_atoms: dict[str, int] = {}
        for i, atom in atom_mapping.items():
            if atom.symbol == "":
                neighbouring_atoms[atom.name] = i
            else:
                matched_atoms[atom.name] = i

        if (  # nofmt
            resdef.linking_bond is None
            or resdef.prior_bond_leaving_atoms.issubset(matched_atoms)
        ):
            prior_bond = None
        elif resdef.prior_bond_leaving_atoms.isdisjoint(matched_atoms):
            prior_bond = (
                neighbouring_atoms[resdef.prior_bond_linking_atom],
                matched_atoms[resdef.posterior_bond_linking_atom],
            )
        else:
            assert False, "this is not a valid mapping"

        if (  # nofmt
            resdef.linking_bond is None
            or resdef.posterior_bond_leaving_atoms.issubset(matched_atoms)
        ):
            posterior_bond = None
        elif resdef.posterior_bond_leaving_atoms.isdisjoint(matched_atoms):
            posterior_bond = (
                matched_atoms[resdef.prior_bond_linking_atom],
                neighbouring_atoms[resdef.posterior_bond_linking_atom],
            )
        else:
            assert False

        if (  # nofmt
            resdef.crosslink is None
            or resdef.crosslink_leaving_atoms.issubset(matched_atoms)
        ):
            crosslink = None
        elif resdef.crosslink_leaving_atoms.isdisjoint(matched_atoms):
            crosslink = (
                matched_atoms[resdef.crosslink.atom1],
                neighbouring_atoms[resdef.crosslink.atom2],
            )
        else:
            assert False

        return cls(
            residue_definition=resdef,
            index_to_atomdef={k: v for k, v in atom_mapping.items() if v.symbol != ""},
            vsite_idcs=(),
            prior_bond_idcs=prior_bond,
            posterior_bond_idcs=posterior_bond,
            crosslink_idcs=crosslink,
        )
