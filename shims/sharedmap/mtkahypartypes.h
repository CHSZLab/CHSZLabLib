// Redirect: SharedMap includes <mtkahypartypes.h>, our Mt-KaHyPar provides libmtkahypartypes.h.
// Also provides mt_kahypar_error_t that the newer API uses.
#pragma once
#include "libmtkahypartypes.h"

#include <cstddef>

// --- Compatibility types for newer Mt-KaHyPar C API ---

typedef enum {
    SUCCESS = 0,
    ERROR_GENERIC = 1
} mt_kahypar_status_t;

typedef struct {
    mt_kahypar_status_t status;
    const char* msg;
    size_t msg_len;
} mt_kahypar_error_t;
