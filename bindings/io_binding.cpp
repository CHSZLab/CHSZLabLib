// Fast METIS / hMETIS file parser exposed as a pybind11 module (_io).
//
// Both parsers slurp the entire file with fread and use a hand-rolled
// parse_int for maximum throughput (avoids sscanf / strtol overhead).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// Read entire file into a std::string.
static std::string slurp(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f)
        throw std::runtime_error(std::string("Cannot open file: ") + path);
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string buf(static_cast<size_t>(sz), '\0');
    size_t rd = std::fread(&buf[0], 1, static_cast<size_t>(sz), f);
    std::fclose(f);
    buf.resize(rd);
    return buf;
}

// Skip whitespace (spaces and tabs only, NOT newlines).
static inline void skip_ws(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t'))
        ++p;
}

// Skip to end of current line (past the newline).
static inline void skip_line(const char*& p, const char* end) {
    while (p < end && *p != '\n')
        ++p;
    if (p < end)
        ++p; // skip '\n'
}

// Parse one integer (possibly negative); advance p past trailing whitespace.
// Returns false when no digit is found (end of line / stream).
static inline bool parse_int(const char*& p, const char* end, int64_t& out) {
    skip_ws(p, end);
    if (p >= end || *p == '\n' || *p == '\r')
        return false;
    bool negative = false;
    if (*p == '-') {
        negative = true;
        ++p;
    }
    int64_t val = 0;
    bool found = false;
    while (p < end && *p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        ++p;
        found = true;
    }
    if (!found)
        return false;
    out = negative ? -val : val;
    // skip trailing spaces/tabs (not newline)
    skip_ws(p, end);
    return true;
}

// Skip blank lines and comment lines (starting with % or c for hmetis).
// `allow_c_comments` controls whether 'c' is treated as a comment prefix.
static inline void skip_comments(const char*& p, const char* end,
                                 bool allow_c_comments) {
    while (p < end) {
        // skip blank lines
        if (*p == '\n' || *p == '\r') {
            ++p;
            if (p < end && *(p - 1) == '\r' && *p == '\n')
                ++p;
            continue;
        }
        if (*p == '%') {
            skip_line(p, end);
            continue;
        }
        if (allow_c_comments && *p == 'c' &&
            (p + 1 >= end || *(p + 1) == ' ' || *(p + 1) == '\t' ||
             *(p + 1) == '\n' || *(p + 1) == '\r')) {
            skip_line(p, end);
            continue;
        }
        break;
    }
}

// Consume rest of current line (past the newline), useful after parsing.
static inline void finish_line(const char*& p, const char* end) {
    while (p < end && *p != '\n')
        ++p;
    if (p < end)
        ++p;
}

// -----------------------------------------------------------------------
// read_metis_cpp
// -----------------------------------------------------------------------

static py::tuple read_metis_cpp(const std::string& path) {
    std::string buf = slurp(path.c_str());
    const char* p = buf.data();
    const char* end = p + buf.size();

    // Skip leading comments / blank lines (% only for METIS)
    skip_comments(p, end, /*allow_c_comments=*/false);

    // Parse header: n m [fmt]
    int64_t n_val = 0, m_val = 0, fmt_val = 0;
    if (!parse_int(p, end, n_val))
        throw std::runtime_error("METIS: cannot parse n from header");
    if (!parse_int(p, end, m_val))
        throw std::runtime_error("METIS: cannot parse m from header");
    parse_int(p, end, fmt_val); // optional
    finish_line(p, end);

    int n = static_cast<int>(n_val);
    int fmt = static_cast<int>(fmt_val);
    bool has_node_weights = (fmt == 10 || fmt == 11);
    bool has_edge_weights = (fmt == 1 || fmt == 11);

    // Pre-allocate
    std::vector<int64_t> xadj;
    xadj.reserve(n + 1);
    xadj.push_back(0);

    // Rough estimate: 2*m entries in adjncy for undirected graphs
    std::vector<int32_t> adjncy;
    adjncy.reserve(static_cast<size_t>(m_val) * 2);

    std::vector<int64_t> node_weights;
    if (has_node_weights)
        node_weights.reserve(n);

    std::vector<int64_t> edge_weights;
    if (has_edge_weights)
        edge_weights.reserve(static_cast<size_t>(m_val) * 2);

    int lines_parsed = 0;
    while (lines_parsed < n && p < end) {
        // Skip comment lines (% only) but NOT blank lines —
        // a blank line represents an isolated node with no neighbors.
        while (p < end && *p == '%')
            skip_line(p, end);
        if (p >= end)
            break;

        // Node weight
        if (has_node_weights) {
            int64_t nw = 0;
            if (parse_int(p, end, nw))
                node_weights.push_back(nw);
            else
                node_weights.push_back(1);
        }

        // Neighbors (and optional edge weights)
        int64_t count = 0;
        while (true) {
            int64_t neighbor = 0;
            // Check for newline or end
            if (p >= end || *p == '\n' || *p == '\r')
                break;
            if (!parse_int(p, end, neighbor))
                break;
            adjncy.push_back(static_cast<int32_t>(neighbor - 1)); // 1-indexed -> 0-indexed

            if (has_edge_weights) {
                int64_t ew = 1;
                parse_int(p, end, ew);
                edge_weights.push_back(ew);
            }
            ++count;
        }

        xadj.push_back(xadj.back() + count);
        finish_line(p, end);
        ++lines_parsed;
    }

    if (lines_parsed != n) {
        throw std::runtime_error(
            "METIS: expected " + std::to_string(n) + " adjacency lines, got " +
            std::to_string(lines_parsed));
    }

    // Build NumPy arrays
    size_t xadj_sz = xadj.size();
    size_t adj_sz = adjncy.size();

    py::array_t<int64_t> py_xadj(static_cast<py::ssize_t>(xadj_sz));
    std::memcpy(py_xadj.mutable_data(), xadj.data(), xadj_sz * sizeof(int64_t));

    py::array_t<int32_t> py_adjncy(static_cast<py::ssize_t>(adj_sz));
    std::memcpy(py_adjncy.mutable_data(), adjncy.data(), adj_sz * sizeof(int32_t));

    py::object py_nw = py::none();
    if (has_node_weights) {
        py::array_t<int64_t> arr(static_cast<py::ssize_t>(node_weights.size()));
        std::memcpy(arr.mutable_data(), node_weights.data(),
                    node_weights.size() * sizeof(int64_t));
        py_nw = std::move(arr);
    }

    py::object py_ew = py::none();
    if (has_edge_weights) {
        py::array_t<int64_t> arr(static_cast<py::ssize_t>(edge_weights.size()));
        std::memcpy(arr.mutable_data(), edge_weights.data(),
                    edge_weights.size() * sizeof(int64_t));
        py_ew = std::move(arr);
    }

    return py::make_tuple(py_xadj, py_adjncy, py_nw, py_ew);
}

// -----------------------------------------------------------------------
// read_hmetis_cpp
// -----------------------------------------------------------------------

static py::tuple read_hmetis_cpp(const std::string& path) {
    std::string buf = slurp(path.c_str());
    const char* p = buf.data();
    const char* end = p + buf.size();

    // Skip leading comments (c or %)
    skip_comments(p, end, /*allow_c_comments=*/true);

    // Parse header: M N [W]
    int64_t m_val = 0, n_val = 0, w_val = 0;
    if (!parse_int(p, end, m_val))
        throw std::runtime_error("hMETIS: cannot parse M from header");
    if (!parse_int(p, end, n_val))
        throw std::runtime_error("hMETIS: cannot parse N from header");
    parse_int(p, end, w_val); // optional
    finish_line(p, end);

    int m = static_cast<int>(m_val); // num hyperedges
    int n = static_cast<int>(n_val); // num vertices
    int w = static_cast<int>(w_val);

    bool has_edge_weights = (w == 1 || w == 11);
    bool has_node_weights = (w == 10 || w == 11);

    // --- Parse M edge lines: build eptr / everts (edge -> vertex CSR) ---
    std::vector<int64_t> eptr;
    eptr.reserve(m + 1);
    eptr.push_back(0);

    std::vector<int32_t> everts;
    everts.reserve(static_cast<size_t>(m) * 3); // rough estimate

    std::vector<int64_t> edge_weights;
    if (has_edge_weights)
        edge_weights.reserve(m);

    // Also collect per-edge vertex lists to build vertex->edge later
    // We store edge id for each (vertex, edge) pair
    // vertex_edges[v] = list of edge ids
    // But for efficiency, we build it in a second pass from eptr/everts.

    for (int eid = 0; eid < m; ++eid) {
        skip_comments(p, end, /*allow_c_comments=*/true);

        if (has_edge_weights) {
            int64_t ew = 1;
            parse_int(p, end, ew);
            edge_weights.push_back(ew);
        }

        int64_t count = 0;
        while (true) {
            if (p >= end || *p == '\n' || *p == '\r')
                break;
            int64_t v = 0;
            if (!parse_int(p, end, v))
                break;
            everts.push_back(static_cast<int32_t>(v - 1)); // 1-indexed -> 0-indexed
            ++count;
        }
        eptr.push_back(eptr.back() + count);
        finish_line(p, end);
    }

    // --- Parse N node weight lines (if present) ---
    std::vector<int64_t> node_weights;
    if (has_node_weights) {
        node_weights.reserve(n);
        for (int vid = 0; vid < n; ++vid) {
            skip_comments(p, end, /*allow_c_comments=*/true);
            int64_t nw = 1;
            parse_int(p, end, nw);
            node_weights.push_back(nw);
            finish_line(p, end);
        }
    }

    // --- Build vertex -> edge CSR (vptr / vedges) ---
    // Count edges per vertex
    std::vector<int64_t> degree(n, 0);
    for (size_t i = 0; i < everts.size(); ++i) {
        int v = everts[i];
        if (v >= 0 && v < n)
            ++degree[v];
    }

    std::vector<int64_t> vptr(n + 1, 0);
    for (int v = 0; v < n; ++v)
        vptr[v + 1] = vptr[v] + degree[v];

    std::vector<int32_t> vedges(vptr[n], 0);
    // Reuse degree as a write cursor
    std::fill(degree.begin(), degree.end(), 0);

    for (int eid = 0; eid < m; ++eid) {
        int64_t start = eptr[eid];
        int64_t stop  = eptr[eid + 1];
        for (int64_t j = start; j < stop; ++j) {
            int v = everts[j];
            int64_t pos = vptr[v] + degree[v];
            vedges[pos] = static_cast<int32_t>(eid);
            ++degree[v];
        }
    }

    // --- Build NumPy arrays ---
    py::array_t<int64_t> py_eptr(static_cast<py::ssize_t>(eptr.size()));
    std::memcpy(py_eptr.mutable_data(), eptr.data(), eptr.size() * sizeof(int64_t));

    py::array_t<int32_t> py_everts(static_cast<py::ssize_t>(everts.size()));
    std::memcpy(py_everts.mutable_data(), everts.data(), everts.size() * sizeof(int32_t));

    py::array_t<int64_t> py_vptr(static_cast<py::ssize_t>(vptr.size()));
    std::memcpy(py_vptr.mutable_data(), vptr.data(), vptr.size() * sizeof(int64_t));

    py::array_t<int32_t> py_vedges(static_cast<py::ssize_t>(vedges.size()));
    std::memcpy(py_vedges.mutable_data(), vedges.data(), vedges.size() * sizeof(int32_t));

    py::object py_nw = py::none();
    if (has_node_weights) {
        py::array_t<int64_t> arr(static_cast<py::ssize_t>(node_weights.size()));
        std::memcpy(arr.mutable_data(), node_weights.data(),
                    node_weights.size() * sizeof(int64_t));
        py_nw = std::move(arr);
    }

    py::object py_ew = py::none();
    if (has_edge_weights) {
        py::array_t<int64_t> arr(static_cast<py::ssize_t>(edge_weights.size()));
        std::memcpy(arr.mutable_data(), edge_weights.data(),
                    edge_weights.size() * sizeof(int64_t));
        py_ew = std::move(arr);
    }

    return py::make_tuple(py_eptr, py_everts, py_vptr, py_vedges,
                          py_nw, py_ew, n);
}

// -----------------------------------------------------------------------
// Module definition
// -----------------------------------------------------------------------

PYBIND11_MODULE(_io, mod) {
    mod.doc() = "Fast METIS / hMETIS file parser (C++)";

    mod.def("read_metis_cpp", &read_metis_cpp,
            py::arg("path"),
            "Parse a METIS file and return (xadj, adjncy, node_weights_or_None, "
            "edge_weights_or_None).");

    mod.def("read_hmetis_cpp", &read_hmetis_cpp,
            py::arg("path"),
            "Parse an hMETIS file and return (eptr, everts, vptr, vedges, "
            "node_weights_or_None, edge_weights_or_None, num_nodes).");
}
