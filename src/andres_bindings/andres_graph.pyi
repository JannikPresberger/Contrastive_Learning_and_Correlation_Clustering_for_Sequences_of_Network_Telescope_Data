from __future__ import annotations
import numpy
import numpy.typing
import typing
__all__: list[str] = ['edge_labels_to_node_labels', 'greedy_additive_edge_contraction', 'greedy_fixation', 'kernighan_lin', 'multicut_ilp']
def edge_labels_to_node_labels(num_vertices: typing.SupportsInt | typing.SupportsIndex, edges: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], edge_labels: typing.Annotated[numpy.typing.ArrayLike, numpy.int8]) -> numpy.typing.NDArray[numpy.uint64]:
    """
    Converts edge labels to node labels
    """
def greedy_additive_edge_contraction(num_vertices: typing.SupportsInt | typing.SupportsIndex, edges: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], edge_values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.int8]:
    """
    Greedy additive edge contraction
    """
def greedy_fixation(num_vertices: typing.SupportsInt | typing.SupportsIndex, edges: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], edge_values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], edge_labels: typing.Annotated[numpy.typing.ArrayLike, numpy.int8]) -> None:
    """
    Greedy fixation
    """
def kernighan_lin(num_vertices: typing.SupportsInt | typing.SupportsIndex, edges: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], edge_values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], edge_labels: typing.Annotated[numpy.typing.ArrayLike, numpy.int8]) -> None:
    """
    Kernighan-Lin
    """
def multicut_ilp(num_vertices: typing.SupportsInt | typing.SupportsIndex, edges: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], edge_values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], edge_labels: typing.Annotated[numpy.typing.ArrayLike, numpy.int8]) -> None:
    """
    ILP
    """
