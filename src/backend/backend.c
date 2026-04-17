#include "backend_internal.h"

#include <stdlib.h>

Backend* backend_create(const BackendConfig* cfg) {
    BackendConfig local = {0};
    if (cfg) {
        local = *cfg;
    } else {
        local.kind = BACKEND_CPU;
        local.threads = 1;
        local.device_id = 0;
    }

    if (local.kind == BACKEND_CUDA) {
#if defined(USE_CUDA) && (USE_CUDA == 1)
        return backend_cuda_create(&local);
#else
        return NULL;
#endif
    }

    return backend_cpu_create(&local);
}

void backend_destroy(Backend* b) {
    if (!b) return;

    if (b->cfg.kind == BACKEND_CUDA) {
#if defined(USE_CUDA) && (USE_CUDA == 1)
        backend_cuda_destroy(b);
        return;
#else
        free(b);
        return;
#endif
    }

    backend_cpu_destroy(b);
}

const KernelVTable* backend_kernels(const Backend* b) {
    return b ? &b->kernels : NULL;
}

BackendKind backend_kind(const Backend* b) {
    return b ? b->cfg.kind : BACKEND_CPU;
}

BackendConfig backend_config(const Backend* b) {
    BackendConfig cfg = {0};
    if (b) cfg = b->cfg;
    return cfg;
}
