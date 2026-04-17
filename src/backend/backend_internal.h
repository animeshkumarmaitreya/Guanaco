#ifndef LLMRT_BACKEND_INTERNAL_H
#define LLMRT_BACKEND_INTERNAL_H

#include "backend.h"

/* Internal backend object shared across backend compilation units. */
struct Backend {
    BackendConfig cfg;
    KernelVTable kernels;
    void* impl; /* backend-specific state (CUDA handles, threadpool, etc.) */
};

Backend* backend_cpu_create(const BackendConfig* cfg);
void     backend_cpu_destroy(Backend* b);

Backend* backend_cuda_create(const BackendConfig* cfg);
void     backend_cuda_destroy(Backend* b);

#endif /* LLMRT_BACKEND_INTERNAL_H */
