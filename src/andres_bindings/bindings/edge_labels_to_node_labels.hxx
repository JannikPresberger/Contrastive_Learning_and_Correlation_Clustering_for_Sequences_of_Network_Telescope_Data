#pragma once
#ifndef ANDRES_GRAPH_EDGE_LABELS_TO_NODE_LABELS_HXX
#define ANDRES_GRAPH_EDGE_LABELS_TO_NODE_LABELS_HXX

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include <andres/graph/graph.hxx>
#include <andres/graph/components.hxx>
#include "helpers.hxx"

namespace py = pybind11;

inline
py::array_t<size_t> edge_labels_to_node_labels(
    size_t const num_vertices,
    py::array_t<size_t> const & edges,
    py::array_t<char> const & edge_labels
) {
    typedef andres::graph::Graph<> Graph;
    Graph graph(num_vertices);

    auto const edges_ptr = edges.unchecked<2>();

    for (size_t i = 0; i < edges.shape(0); ++i) {
        graph.insertEdge(edges_ptr(i, 0), edges_ptr(i, 1));
    }

    MulticutSubgraphMask const mask(edge_labels.data(0));
    andres::graph::ComponentsBySearch<Graph> componentsBySearch;
    componentsBySearch.build(graph, mask);

    py::array_t<size_t> node_labels(num_vertices);
    auto n_ptr = node_labels.mutable_unchecked<1>();
    for (size_t i = 0; i < num_vertices; ++i) {
        n_ptr(i) = componentsBySearch.labels_[i];
    }
    return node_labels;
}

#endif //ANDRES_GRAPH_EDGE_LABELS_TO_NODE_LABELS_HXX