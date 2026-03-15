#ifndef SORT_ADJACENCY_H
#define SORT_ADJACENCY_H

#include <algorithm>

/// Sort each node's adjacency list segment by neighbor index.
/// KaMIS assumes sorted adjacency lists internally; this ensures correctness
/// for CSR arrays that may arrive unsorted (e.g. via Graph.from_csr()).
inline void sort_adjacency_lists(int n, int *xadj, int *adjncy) {
    for (int v = 0; v < n; ++v) {
        std::sort(adjncy + xadj[v], adjncy + xadj[v + 1]);
    }
}

#endif // SORT_ADJACENCY_H
