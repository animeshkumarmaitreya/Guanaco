#ifndef LLMRT_BACKEND_H
#define LLMRT_BACKEND_H

#include "types.h"

/*============================================================================
 * Backend abstraction (MUST)
 *
 * Goal: engine code calls kernels via a selected backend (CPU/CUDA).
 * This header is additive scaffolding; nothing in the current build depends on it yet.
 *============================================================================*/

typedef enum {
    BACKEND_CPU  = 0,
    BACKEND_CUDA = 1,
} BackendKind;

typedef struct Backend Backend;

typedef struct {
    /* Core math */
    void (*gemm_f32)(const Tensor* A, const Tensor* B, Tensor* C);

    /* Standard GEMM variant for attention context: C = A(M,K) * B(K,N) */
    void (*gemm_f32_nn)(const Tensor* A, const Tensor* B, Tensor* C);

    void (*rmsnorm)(const Tensor* input, const Tensor* weight, Tensor* output, float eps);
    void (*softmax_inplace)(Tensor* scores, int seq_len);
    void (*silu_inplace)(Tensor* x);
    void (*rope)(Tensor* q, Tensor* k, int pos, int head_dim);
    void (*residual_add)(Tensor* x, const Tensor* residual);
    void (*elemwise_mul)(Tensor* a, const Tensor* b);

    /* Quantized decode hot path (Phase B) */
    int (*matvec_q4k_f32)(const Tensor* W_q4k, const float* x, float* y);
    int (*matvec_q8_0_f32)(const Tensor* W_q8_0, const float* x, float* y);
} KernelVTable;

typedef struct {
    BackendKind kind;
    int threads;   /* CPU threads (or host threads used by CUDA path) */
    int device_id; /* CUDA device id; ignored for CPU */
} BackendConfig;

Backend*           backend_create(const BackendConfig* cfg);
void               backend_destroy(Backend* b);
const KernelVTable* backend_kernels(const Backend* b);
BackendKind        backend_kind(const Backend* b);
BackendConfig      backend_config(const Backend* b);

#endif /* LLMRT_BACKEND_H */
