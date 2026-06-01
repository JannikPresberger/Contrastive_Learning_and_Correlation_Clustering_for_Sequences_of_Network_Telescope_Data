#pragma once
#ifndef ANDRES_GRAPH_GREEDY_ADDITIVE_EDGE_CONTRACTION_HXX
#define ANDRES_GRAPH_GREEDY_ADDITIVE_EDGE_CONTRACTION_HXX

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <andres/graph/graph.hxx>
#include <andres/graph/multicut/greedy-additive.hxx>
#include "helpers.hxx"

namespace py = pybind11;


inline
py::array_t<char> greedy_additive_edge_contraction(
    size_t const num_vertices,
    py::array_t<size_t> const & edges,
    py::array_t<double> const & edge_values
) {
    typedef andres::graph::Graph<> Graph;

    if (edges.ndim() != 2 || edges.shape(1) != 2) {
        throw std::runtime_error("Edges must be an Nx2 array");
    }

    Graph graph(num_vertices);

    auto const edges_ptr = edges.unchecked<2>();
    auto const values_ptr = edge_values.unchecked<1>();
    size_t const num_edges = edges.shape(0);

    for (size_t i = 0; i < num_edges; ++i) {
        graph.insertEdge(edges_ptr(i, 0), edges_ptr(i, 1));
    }

    py::array_t<char> edgeLabels(num_edges);
    char* label_ptr = edgeLabels.mutable_data();
    EdgeWeights const weights(values_ptr.data(0));

    andres::graph::multicut::greedyAdditiveEdgeContraction(graph, weights, label_ptr);

    return edgeLabels;
}

#endif //ANDRES_GRAPH_GREEDY_ADDITIVE_EDGE_CONTRACTION_HXX