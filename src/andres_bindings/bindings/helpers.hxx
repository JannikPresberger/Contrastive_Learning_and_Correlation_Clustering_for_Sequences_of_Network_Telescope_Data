#pragma once
#ifndef ANDRES_GRAPH_HELPERS_HXX
#define ANDRES_GRAPH_HELPERS_HXX

struct EdgeWeights {
    const double* weights_;
    typedef double value_type;

    explicit EdgeWeights(const double* weights) : weights_(weights) {}
    double operator[](std::size_t const i) const { return weights_[i]; }

    [[nodiscard]] const double* data() const { return weights_; }
};

struct MulticutSubgraphMask {

    MulticutSubgraphMask(const char* edgeLabels): edgeLabels_(edgeLabels){};

    [[nodiscard]] bool vertex(const size_t v) const
    { return true; }
    [[nodiscard]] bool edge(const size_t e) const
    { return edgeLabels_[e] == 0; }

private:
    const char* edgeLabels_;
};

template<typename T>
struct ArrayProxy {
    T* ptr;
    ArrayProxy(T* p) : ptr(p) {}
    T* begin() const { return ptr; }
    T& operator[](size_t i) { return ptr[i]; }
};

#endif //ANDRES_GRAPH_HELPERS_HXX