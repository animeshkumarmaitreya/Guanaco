#ifndef LLMRT_KERNELS_H
#define LLMRT_KERNELS_H

#include "types.h"

/*============================================================================
 * Kernel API — Person A owns the implementation
 * 
 * All kernels are STATELESS: they read inputs, write outputs, never allocate.
 * All buffer pointers must be 64-byte aligned.
 *
 * Layout contract (important):
 * - Today, CPU kernels assume contiguous row-major buffers.
 * - `Tensor.stride[]` is primarily debug metadata; many kernels effectively
 *   ignore it for performance and will misbehave on strided/views.
 * - If you need to operate on a strided slice, pack/copy into a contiguous
 *   scratch tensor before calling kernels.
 *============================================================================*/

/* Matrix multiply (matches GGUF weight layout): C = A × B^T
 *
 * Naming note:
 * - The name `gemm_f32` is kept for historical/compatibility reasons.
 * - Semantically, it behaves like a GEMM where the right operand is treated as
 *   transposed: B is stored as (N, K) but used as B^T.
 * - This matches GGUF weight layout (rows = output features), so the runtime
 *   can compute dot-products against weight rows without materializing a
 *   transposed copy.
 *
 * Contract:
 * - A: (M, K) row-major
 * - B: (N, K) row-major (each row is one output feature / vector of length K)
 * - C: (M, N) row-major
 */
void gemm_f32(const Tensor* A, const Tensor* B, Tensor* C);

/* RMSNorm: output = (input / rms(input)) * weight
 * input: (T, H), weight: (H,), output: (T, H)
 * eps: small constant to avoid division by zero (use 1e-6) */
void cpu_rmsnorm(const Tensor *input, const Tensor *weight, Tensor *output, float eps);

/* GPU Fused Layer Entry (CUDA) */
int cuda_transformer_layer_gpu(
    void* wq, int wq_r, int wq_c, int wq_dt,
    void* wk, int wk_r, int wk_c, int wk_dt,
    void* wv, int wv_r, int wv_c, int wv_dt,
    void* wo, int wo_r, int wo_c, int wo_dt,
    void* gate, int gate_r, int gate_c, int gate_dt,
    void* up, int up_r, int up_c, int up_dt,
    void* down, int down_r, int down_c, int down_dt,
    const float* rms_att, const float* rms_ffn,
    float* h_hidden,
    float* h_k_cache, float* h_v_cache, int head_stride,
    int H, int n_heads, int n_kv_heads, int head_dim,
    int pos, int layer, int max_seq);

/* Softmax in-place over the last dimension
 * scores: (..., seq_len) — the last dimension is softmax'd
 * Numerically stable (subtract max first).
 * Must handle all -INFINITY rows without producing NaN. */
void softmax_inplace(Tensor* scores, int seq_len);

/* SiLU activation in-place: x = x * sigmoid(x) */
void silu_inplace(Tensor* x);

/* Rotary Position Embedding: rotate Q and K at the given position
 * q: (n_heads, head_dim), k: (n_kv_heads, head_dim) */
void rope(Tensor* q, Tensor* k, int pos, int head_dim);

/* Residual add in-place: x = x + residual
 * Both tensors must have identical shape. */
void residual_add(Tensor* x, const Tensor* residual);

/* Elementwise multiply in-place: a = a * b
 * Both tensors must have identical shape. */
void elemwise_mul(Tensor* a, const Tensor* b);

#endif /* LLMRT_KERNELS_H */
