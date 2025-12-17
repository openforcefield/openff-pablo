from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from typing import Literal

import rustworkx as rx

from openff.pablo._utils import unwrap
from openff.pablo.exceptions import PabloError

__all__ = ["Graph"]


class Graph[NodeT: Hashable, EdgeT: Hashable]:
    def __init__(
        self,
        *,
        node_count_hint: int | None = None,
        edge_count_hint: int | None = None,
    ):
        self._graph: rx.PyGraph[NodeT, EdgeT] = rx.PyGraph(
            multigraph=False,
            node_count_hint=node_count_hint,
            edge_count_hint=edge_count_hint,
        )
        self._node_idcs: dict[NodeT, int] = {}
        self._edge_idcs: dict[EdgeT, int] = {}
        super().__init__()

    def __contains__(self, item: NodeT | EdgeT):
        return item in self._node_idcs or item in self._edge_idcs

    @property
    def n_nodes(self) -> int:
        return self._graph.num_nodes()

    @property
    def n_edges(self) -> int:
        return self._graph.num_edges()

    @property
    def nodes(self) -> Iterator[NodeT]:
        yield from self._node_idcs.keys()

    @property
    def edges(self) -> Iterator[EdgeT]:
        yield from self._edge_idcs.keys()

    def neighbours(self, node: NodeT) -> Iterator[NodeT]:
        node_idx = self._node_idcs[node]
        neighbour_idcs = self._graph.neighbors(node_idx)
        for neighbour_idx in neighbour_idcs:
            yield self._graph.get_node_data(neighbour_idx)

    def _add_edge_between_indices(self, node_a_idx: int, node_b_idx: int, edge: EdgeT):
        self._edge_idcs[edge] = self._graph.add_edge(node_a_idx, node_b_idx, edge)

    def add_edge(self, node_a: NodeT, node_b: NodeT, edge: EdgeT):
        node_a_idx = self._node_idcs[node_a]
        node_b_idx = self._node_idcs[node_b]
        self._add_edge_between_indices(node_a_idx, node_b_idx, edge)

    def add_edges_from(self, obj_list: Iterable[tuple[NodeT, NodeT, EdgeT]]):
        for params in obj_list:
            self.add_edge(*params)

    def add_node(self, node: NodeT, *, skip_existing: bool = False):
        if node in self._node_idcs:
            if skip_existing:
                return
            raise PabloError("Cannot add existing node")
        self._node_idcs[node] = self._graph.add_node(node)

    def add_nodes_from(self, nodes: Iterable[NodeT], *, skip_existing: bool = False):
        for node in nodes:
            self.add_node(node, skip_existing=skip_existing)

    def is_isomorphic_to[OtherNodeT: Hashable, OtherEdgeT: Hashable](
        self,
        other: "Graph[OtherNodeT, OtherEdgeT]",
        node_matcher: Callable[[NodeT, OtherNodeT], bool] | None = None,
        edge_matcher: Callable[[EdgeT, OtherEdgeT], bool] | None = None,
        id_order: bool = True,
        call_limit: int | None = None,
    ) -> bool:
        iterator = _vf2_mapping(
            first=self._graph,
            second=other._graph,
            node_matcher=node_matcher,
            edge_matcher=edge_matcher,
            id_order=id_order,
            call_limit=call_limit,
        )
        try:
            next(iterator)
        except StopIteration:
            return False
        else:
            return True

    def is_subgraph_of[OtherNodeT: Hashable, OtherEdgeT: Hashable](
        self,
        other: "Graph[OtherNodeT, OtherEdgeT]",
        node_matcher: Callable[[NodeT, OtherNodeT], bool] | None = None,
        edge_matcher: Callable[[EdgeT, OtherEdgeT], bool] | None = None,
        id_order: bool = True,
        call_limit: int | None = None,
    ) -> bool:
        iterator = _vf2_mapping(
            first=other._graph,
            second=self._graph,
            node_matcher=(
                None if node_matcher is None else lambda x, y: node_matcher(y, x)
            ),
            edge_matcher=(
                None if edge_matcher is None else lambda x, y: edge_matcher(y, x)
            ),
            id_order=id_order,
            call_limit=call_limit,
            subgraph=True,
            induced=True,
        )
        try:
            next(iterator)
        except StopIteration:
            return False
        else:
            return True

    def get_mappings[OtherNodeT: Hashable, OtherEdgeT: Hashable](
        self,
        other: "Graph[OtherNodeT, OtherEdgeT]",
        node_matcher: Callable[[NodeT, OtherNodeT], bool] | None = None,
        edge_matcher: Callable[[EdgeT, OtherEdgeT], bool] | None = None,
        id_order: bool = True,
        subgraph: bool = False,
        induced: bool = True,
        call_limit: int | None = None,
    ) -> Iterator[dict[NodeT, OtherNodeT]]:
        nodes = {v: k for k, v in self._node_idcs.items()}
        other_nodes = {v: k for k, v in other._node_idcs.items()}
        for mapping in _vf2_mapping(
            first=self._graph,
            second=other._graph,
            node_matcher=node_matcher,
            edge_matcher=edge_matcher,
            id_order=id_order,
            subgraph=subgraph,
            induced=induced,
            call_limit=call_limit,
        ):
            yield {nodes[i]: other_nodes[j] for i, j in mapping.items()}

    def is_connected(self) -> bool:
        return rx.is_connected(self._graph)

    def desymmetrize_leaf_nodes(self) -> "Graph[tuple[NodeT, int], EdgeT]":
        new_graph = Graph[tuple[NodeT, int], EdgeT]()

        old_to_new: dict[int, int] = {}
        leaf_nodes: dict[int, list[int]] = {}
        for node_idx in self._graph.node_indices():
            n_edges = self._graph.degree(node_idx)
            if n_edges != 1:
                node = (self._graph[node_idx], 0)
                new_graph.add_node(node)
                old_to_new[node_idx] = new_graph._node_idcs[node]
                continue
            # node_idx is a leaf node
            parent = unwrap(self._graph.neighbors(node_idx))
            leaf_nodes.setdefault(parent, []).append(node_idx)

        for sibling_leaf_nodes in leaf_nodes.values():
            for i, node_idx in enumerate(sibling_leaf_nodes):
                node = (self._graph[node_idx], i)
                new_graph.add_node(node)
                old_to_new[node_idx] = new_graph._node_idcs[node]

        for (
            old_node_a_idx,
            old_node_b_idx,
            edge,
        ) in self._graph.edge_index_map().values():
            node_a_idx = old_to_new[old_node_a_idx]
            node_b_idx = old_to_new[old_node_b_idx]
            new_graph._add_edge_between_indices(node_a_idx, node_b_idx, edge)

        return new_graph


def _vf2_mapping[
    NodeT: Hashable,
    EdgeT: Hashable,
    OtherNodeT: Hashable,
    OtherEdgeT: Hashable,
](
    first: rx.PyGraph[NodeT, EdgeT],
    second: rx.PyGraph[OtherNodeT, OtherEdgeT],
    node_matcher: Callable[[NodeT, OtherNodeT], bool] | None = None,
    edge_matcher: Callable[[EdgeT, OtherEdgeT], bool] | None = None,
    id_order: bool = True,
    subgraph: bool = False,
    induced: bool = True,
    call_limit: int | None = None,
) -> Iterator[Mapping[int, int]]:
    first_labeled: rx.PyGraph[
        tuple[Literal["FIRST", "SECOND"], NodeT | OtherNodeT],
        tuple[Literal["FIRST", "SECOND"], EdgeT | OtherEdgeT],
    ] = rx.PyGraph()
    first_comp_dict = first_labeled.compose(
        # compose type annotation is too restrictive
        first,  # pyright: ignore[reportArgumentType]
        {},
        node_map_func=lambda x: ("FIRST", x),  # pyright: ignore[reportArgumentType]
        edge_map_func=lambda x: ("FIRST", x),  # pyright: ignore[reportArgumentType]
    )
    assert first_comp_dict == {i: i for i in range(first.num_nodes())}
    second_labeled: rx.PyGraph[
        tuple[Literal["FIRST", "SECOND"], NodeT | OtherNodeT],
        tuple[Literal["FIRST", "SECOND"], EdgeT | OtherEdgeT],
    ] = rx.PyGraph()
    second_comp_dict = second_labeled.compose(
        # compose type annotation is too restrictive
        second,  # pyright: ignore[reportArgumentType]
        {},
        node_map_func=lambda x: ("SECOND", x),  # pyright: ignore[reportArgumentType]
        edge_map_func=lambda x: ("SECOND", x),  # pyright: ignore[reportArgumentType]
    )
    assert second_comp_dict == {i: i for i in range(second.num_nodes())}
    return rx.vf2_mapping(
        first_labeled,
        second_labeled,
        node_matcher=(
            None
            if node_matcher is None
            else lambda a, b: (
                node_matcher(a[1], b[1])  # pyright: ignore[reportArgumentType]
                if a[0] == "FIRST"
                else node_matcher(b[1], a[1])  # pyright: ignore[reportArgumentType]
            )
        ),
        edge_matcher=(
            None
            if edge_matcher is None
            else lambda a, b: (
                edge_matcher(a[1], b[1])  # pyright: ignore[reportArgumentType]
                if a[0] == "FIRST"
                else edge_matcher(b[1], a[1])  # pyright: ignore[reportArgumentType]
            )
        ),
        id_order=id_order,
        subgraph=subgraph,
        induced=induced,
        call_limit=call_limit,
    )
