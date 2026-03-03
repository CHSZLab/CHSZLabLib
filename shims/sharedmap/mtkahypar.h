// Redirect: SharedMap includes <mtkahypar.h>, our Mt-KaHyPar provides libmtkahypar.h.
// Provides inline compatibility wrappers for the newer Mt-KaHyPar C API that
// SharedMap expects (error struct overloads, context_from_preset, initialize).
#pragma once

// Include our shimmed types first (provides mt_kahypar_error_t, SUCCESS, etc.)
#include "mtkahypartypes.h"

// Include the old C API
#include "libmtkahypar.h"

// ---------------------------------------------------------------------------
// Thread-local storage for preset tracking (used by create_graph wrapper)
// ---------------------------------------------------------------------------
namespace _mtkahypar_shim {
    inline thread_local mt_kahypar_preset_type_t last_preset = DEFAULT;
}

// ---------------------------------------------------------------------------
// mt_kahypar_initialize (newer name for mt_kahypar_initialize_thread_pool)
// ---------------------------------------------------------------------------
inline void mt_kahypar_initialize(size_t num_threads, bool interleaved) {
    mt_kahypar_initialize_thread_pool(num_threads, interleaved);
}

// ---------------------------------------------------------------------------
// mt_kahypar_context_from_preset (combines context_new + load_preset)
// ---------------------------------------------------------------------------
inline mt_kahypar_context_t* mt_kahypar_context_from_preset(mt_kahypar_preset_type_t preset) {
    _mtkahypar_shim::last_preset = preset;
    mt_kahypar_context_t* ctx = mt_kahypar_context_new();
    if (ctx) mt_kahypar_load_preset(ctx, preset);
    return ctx;
}

// ---------------------------------------------------------------------------
// 4-arg overload of mt_kahypar_set_context_parameter (old API has 3 args)
// ---------------------------------------------------------------------------
inline void mt_kahypar_set_context_parameter(mt_kahypar_context_t* ctx,
                                              mt_kahypar_context_parameter_type_t type,
                                              const char* value,
                                              mt_kahypar_error_t* error) {
    int rc = mt_kahypar_set_context_parameter(ctx, type, value);
    if (error) {
        error->status = (rc == 0) ? SUCCESS : ERROR_GENERIC;
        error->msg = nullptr;
        error->msg_len = 0;
    }
}

// ---------------------------------------------------------------------------
// 7-arg overload of mt_kahypar_create_graph (old API has 6 args, takes preset)
// ---------------------------------------------------------------------------
inline mt_kahypar_hypergraph_t mt_kahypar_create_graph(
        mt_kahypar_context_t* /* context */,
        mt_kahypar_hypernode_id_t num_vertices,
        mt_kahypar_hyperedge_id_t num_edges,
        const mt_kahypar_hypernode_id_t* edges,
        const mt_kahypar_hyperedge_weight_t* edge_weights,
        const mt_kahypar_hypernode_weight_t* vertex_weights,
        mt_kahypar_error_t* error) {
    mt_kahypar_hypergraph_t hg = mt_kahypar_create_graph(
        _mtkahypar_shim::last_preset, num_vertices, num_edges,
        edges, edge_weights, vertex_weights);
    if (error) {
        error->status = SUCCESS;
        error->msg = nullptr;
        error->msg_len = 0;
    }
    return hg;
}

// ---------------------------------------------------------------------------
// 3-arg overload of mt_kahypar_partition (old API has 2 args)
// ---------------------------------------------------------------------------
inline mt_kahypar_partitioned_hypergraph_t mt_kahypar_partition(
        mt_kahypar_hypergraph_t hg,
        mt_kahypar_context_t* ctx,
        mt_kahypar_error_t* error) {
    mt_kahypar_partitioned_hypergraph_t phg = mt_kahypar_partition(hg, ctx);
    if (error) {
        error->status = SUCCESS;
        error->msg = nullptr;
        error->msg_len = 0;
    }
    return phg;
}
