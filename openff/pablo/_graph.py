from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from typing import Generic, Literal, TypeVar

import rustworkx as rx

NodeT = TypeVar("NodeT", bound=Hashable)
EdgeT = TypeVar("EdgeT", bound=Hashable)
OtherNodeT = TypeVar("OtherNodeT", bound=Hashable)
OtherEdgeT = TypeVar("OtherEdgeT", bound=Hashable)

__all__ = ["Graph"]


class Graph(Generic[NodeT, EdgeT]):
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

    def add_edge(self, node_a: NodeT, node_b: NodeT, edge: EdgeT):
        node_a_idx = self._node_idcs[node_a]
        node_b_idx = self._node_idcs[node_b]
        self._edge_idcs[edge] = self._graph.add_edge(node_a_idx, node_b_idx, edge)

    def add_edges_from(self, obj_list: Iterable[tuple[NodeT, NodeT, EdgeT]]):
        for params in obj_list:
            self.add_edge(*params)

    def add_node(self, node: NodeT, *, skip_existing: bool = False):
        if node in self._node_idcs:
            if skip_existing:
                return
            raise ValueError("Cannot add existing node")
        self._node_idcs[node] = self._graph.add_node(node)

    def add_nodes_from(self, nodes: Iterable[NodeT], *, skip_existing: bool = False):
        for node in nodes:
            self.add_node(node, skip_existing=skip_existing)

    def is_isomorphic_to(
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

    def is_subgraph_of(
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

    def get_mappings(
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


def _vf2_mapping(
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
        first,  # type: ignore[compose type annotation is incorrect]
        {},
        node_map_func=lambda x: ("FIRST", x),  # type: ignore
        edge_map_func=lambda x: ("FIRST", x),  # type: ignore
    )
    assert first_comp_dict == {i: i for i in range(first.num_nodes())}
    second_labeled: rx.PyGraph[
        tuple[Literal["FIRST", "SECOND"], NodeT | OtherNodeT],
        tuple[Literal["FIRST", "SECOND"], EdgeT | OtherEdgeT],
    ] = rx.PyGraph()
    second_comp_dict = second_labeled.compose(
        second,  # type: ignore[compose type annotation is incorrect]
        {},
        node_map_func=lambda x: ("SECOND", x),  # type: ignore
        edge_map_func=lambda x: ("SECOND", x),  # type: ignore
    )
    assert second_comp_dict == {i: i for i in range(second.num_nodes())}
    return rx.vf2_mapping(
        first_labeled,
        second_labeled,
        node_matcher=(
            None
            if node_matcher is None
            else lambda a, b: (
                node_matcher(a[1], b[1])  # type: ignore
                if a[0] == "FIRST"
                else node_matcher(b[1], a[1])  # type: ignore
            )
        ),
        edge_matcher=(
            None
            if edge_matcher is None
            else lambda a, b: (
                edge_matcher(a[1], b[1])  # type: ignore
                if a[0] == "FIRST"
                else edge_matcher(b[1], a[1])  # type: ignore
            )
        ),
        id_order=id_order,
        subgraph=subgraph,
        induced=induced,
        call_limit=call_limit,
    )
