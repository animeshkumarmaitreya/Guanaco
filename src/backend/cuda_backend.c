#include "backend_internal.h"

#include <stdlib.h>

/*
 * CUDA backend stubs (Phase A MUST).
 *
 * This file is intentionally pure C (no CUDA headers) so CPU-only builds work.
 * The real implementation will live behind USE_CUDA=1 and call into .cu wrappers.
 */

static void no_op_gemm(const Tensor* A, const Tensor* B, Tensor* C) {
    (void)A;
    (void)B;
    (void)C;
}

static void no_op_rmsnorm(const Tensor* input, const Tensor* weight, Tensor* output, float eps) {
    (void)input;
    (void)weight;
    (void)output;
    (void)eps;
}

static void no_op_softmax(Tensor* scores, int seq_len) {
    (void)scores;
    (void)seq_len;
}

static void no_op_silu(Tensor* x) { (void)x; }
static void no_op_rope(Tensor* q, Tensor* k, int pos, int head_dim) {
    (void)q;
    (void)k;
    (void)pos;
    (void)head_dim;
}
static void no_op_residual_add(Tensor* x, const Tensor* residual) {
    (void)x;
    (void)residual;
}
static void no_op_elemwise_mul(Tensor* a, const Tensor* b) {
    (void)a;
    (void)b;
}

static int unimpl_matvec(const Tensor* W, const float* x, float* y) {
    (void)W;
    (void)x;
    (void)y;
    return -1;
}

Backend* backend_cuda_create(const BackendConfig* cfg) {
    Backend* b = (Backend*)calloc(1, sizeof(Backend));
    if (!b) return NULL;

    b->cfg = *cfg;
    b->cfg.kind = BACKEND_CUDA;

    /* Stub vtable: real CUDA kernels will be wired here. */
    b->kernels.gemm_f32 = no_op_gemm;
    b->kernels.gemm_f32_nn = no_op_gemm;
    b->kernels.rmsnorm = no_op_rmsnorm;
    b->kernels.softmax_inplace = no_op_softmax;
    b->kernels.silu_inplace = no_op_silu;
    b->kernels.rope = no_op_rope;
    b->kernels.residual_add = no_op_residual_add;
    b->kernels.elemwise_mul = no_op_elemwise_mul;
    b->kernels.matvec_q4k_f32 = unimpl_matvec;
    b->kernels.matvec_q8_0_f32 = unimpl_matvec;

    /* Not implemented yet: until Phase A lands, treat as unavailable. */
    free(b);
    return NULL;
}

void backend_cuda_destroy(Backend* b) {
    free(b);
}
