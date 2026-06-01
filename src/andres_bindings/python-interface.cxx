#include "bindings/kernighan-lin.hxx"
#include "bindings/multicut_ilp.hxx"
#include "bindings/greedy_fixation.hxx"
#include "bindings/greedy_additive_edge_contraction.hxx"
#include "bindings/edge_labels_to_node_labels.hxx"

PYBIND11_MODULE(andres_graph, m) {
    m.def("greedy_additive_edge_contraction", &greedy_additive_edge_contraction, "Greedy additive edge contraction",
          py::arg("num_vertices"),
          py::arg("edges"),
          py::arg("edge_values"));
    m.def("greedy_fixation", &greedy_fixation, "Greedy fixation",
          py::arg("num_vertices"),
          py::arg("edges"),
          py::arg("edge_values"),
          py::arg("edge_labels"));
    m.def("kernighan_lin", &kernighan_lin, "Kernighan-Lin",
          py::arg("num_vertices"),
          py::arg("edges"),
          py::arg("edge_values"),
          py::arg("edge_labels"));
    m.def("multicut_ilp", &multicut_ilp, "ILP",
          py::arg("num_vertices"),
          py::arg("edges"),
          py::arg("edge_values"),
          py::arg("edge_labels"));
    m.def("edge_labels_to_node_labels", &edge_labels_to_node_labels, "Converts edge labels to node labels",
          py::arg("num_vertices"),
          py::arg("edges"),
          py::arg("edge_labels"));
}
