#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>

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
        const long long* xadj_ptr = xadj.data();

        // CHILS requires sorted adjacency lists — sort a local copy
        std::vector<int> sorted_adj(adjncy.data(), adjncy.data() + adjncy.size());
        for (int i = 0; i < n; i++) {
            std::sort(sorted_adj.begin() + xadj_ptr[i],
                      sorted_adj.begin() + xadj_ptr[i + 1]);
        }

        // Suppress C stdout/stderr at fd level (CHILS uses printf)
        fflush(stdout);
        fflush(stderr);
        int old_stdout = dup(STDOUT_FILENO);
        int old_stderr = dup(STDERR_FILENO);
        int devnull = open("/dev/null", O_WRONLY);
        dup2(devnull, STDOUT_FILENO);
        dup2(devnull, STDERR_FILENO);
        close(devnull);

        void* solver = chils_initialize();
        chils_set_graph(solver, n, xadj_ptr, sorted_adj.data(), weights.data());
        chils_run_full(solver, time_limit, num_concurrent, seed);

        long long total_weight = chils_solution_get_weight(solver);
        int set_size = chils_solution_get_size(solver);
        int* is_ptr = chils_solution_get_independent_set(solver);

        py::array_t<int> vertices(set_size);
        if (set_size > 0 && is_ptr) {
            std::memcpy(vertices.mutable_data(), is_ptr, set_size * sizeof(int));
        }

        chils_release(solver);

        // Restore stdout/stderr
        fflush(stdout);
        fflush(stderr);
        dup2(old_stdout, STDOUT_FILENO);
        dup2(old_stderr, STDERR_FILENO);
        close(old_stdout);
        close(old_stderr);
        return py::make_tuple(total_weight, vertices);
    }, "Solve maximum weight independent set",
       py::arg("xadj"), py::arg("adjncy"), py::arg("weights"),
       py::arg("time_limit"), py::arg("num_concurrent"), py::arg("seed"));
}
