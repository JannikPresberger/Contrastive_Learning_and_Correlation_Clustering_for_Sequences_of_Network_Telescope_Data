#pragma once
#ifndef ANDRES_GRAPH_GREEDY_MOVING_HXX
#define ANDRES_GRAPH_GREEDY_MOVING_HXX


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <andres/graph/graph.hxx>
#include <andres/graph/multicut/greedy-fixation.hxx>
#include "helpers.hxx"

namespace py = pybind11;


inline
void
greedy_fixation(
    size_t const num_vertices,
    py::array_t<size_t> const & edges,
    py::array_t<double> const & edge_values,
    py::array_t<char> & edgeLabels
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

    EdgeWeights const weights(values_ptr.data(0));
    char* edgeLabel_ptr = edgeLabels.mutable_data();

    andres::graph::multicut::greedyFixation(graph, weights, edgeLabel_ptr);
}

#endif //ANDRES_GRAPH_GREEDY_MOVING_HXX