from collections.abc import Mapping
from typing import Generic, TypeVar
from collections.abc import Callable, Hashable, Iterable, Iterator

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
        self._out_of_bounds_edge: int = 1
        self._out_of_bounds_node: int = 1
        super().__init__()

    def add_edge(self, node_a: NodeT, node_b: NodeT, edge: EdgeT):
        node_a_idx = self._node_idcs.get(node_a, self._out_of_bounds_node)
        node_b_idx = self._node_idcs.get(node_b, self._out_of_bounds_node)
        self._edge_idcs[edge] = self._graph.add_edge(node_a_idx, node_b_idx, edge)
        self._out_of_bounds_edge += 1

    def add_edges_from(self, obj_list: Iterable[tuple[NodeT, NodeT, EdgeT]]):
        for params in obj_list:
            self.add_edge(*params)

    def add_node(self, node: NodeT):
        if node in self._node_idcs:
            raise ValueError("Cannot add existing node")
        self._node_idcs[node] = self._graph.add_node(node)
        self._out_of_bounds_node += 1

    def add_nodes_from(self, nodes: Iterable[NodeT]):
        for node in nodes:
            self.add_node(node)

    def is_isomorphic_to(
        self,
        other: "Graph[OtherNodeT, OtherEdgeT]",
        node_matcher: Callable[[NodeT, OtherNodeT], bool] | None = None,
        edge_matcher: Callable[[EdgeT, OtherEdgeT], bool] | None = None,
        id_order: bool = True,
        call_limit: int | None = None,
    ) -> bool:
        return _is_isomorphic(
            first=self._graph,
            second=other._graph,
            node_matcher=node_matcher,
            edge_matcher=edge_matcher,
            id_order=id_order,
            call_limit=call_limit,
        )

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


def _is_isomorphic(
    first: rx.PyGraph[NodeT, EdgeT],
    second: rx.PyGraph[OtherNodeT, OtherEdgeT],
    node_matcher: Callable[[NodeT, OtherNodeT], bool] | None = None,
    edge_matcher: Callable[[EdgeT, OtherEdgeT], bool] | None = None,
    id_order: bool = True,
    call_limit: int | None = None,
) -> bool:
    return rx.is_isomorphic(
        first=first,
        second=second,  # type: ignore
        node_matcher=node_matcher,  # type: ignore
        edge_matcher=edge_matcher,  # type: ignore
        id_order=id_order,
        call_limit=call_limit,
    )


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
    return rx.vf2_mapping(
        first=first,
        second=second,  # type: ignore
        node_matcher=node_matcher,  # type: ignore
        edge_matcher=edge_matcher,  # type: ignore
        id_order=id_order,
        subgraph=subgraph,
        induced=induced,
        call_limit=call_limit,
    )
