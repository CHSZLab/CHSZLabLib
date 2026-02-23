#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstring>

extern "C" {
#include "chils.h"
}

namespace py = pybind11;

PYBIND11_MODULE(_chils, m) {
    m.doc() = "CHILS maximum weight independent set bindings";

    m.def("mwis", [](
        py::array_t<long long> xadj,
        py::array_t<int> adjncy,
        py::array_t<long long> weights,
        double time_limit,
        int num_concurrent,
        unsigned int seed
    ) {
        int n = static_cast<int>(xadj.size() - 1);

        void* solver = chils_initialize();
        chils_set_graph(solver, n, xadj.data(), adjncy.data(), weights.data());
        chils_run_full(solver, time_limit, num_concurrent, seed);

        long long total_weight = chils_solution_get_weight(solver);
        int set_size = chils_solution_get_size(solver);
        int* is_ptr = chils_solution_get_independent_set(solver);

        py::array_t<int> vertices(set_size);
        if (set_size > 0 && is_ptr) {
            std::memcpy(vertices.mutable_data(), is_ptr, set_size * sizeof(int));
        }

        chils_release(solver);
        return py::make_tuple(total_weight, vertices);
    }, "Solve maximum weight independent set",
       py::arg("xadj"), py::arg("adjncy"), py::arg("weights"),
       py::arg("time_limit"), py::arg("num_concurrent"), py::arg("seed"));
}
