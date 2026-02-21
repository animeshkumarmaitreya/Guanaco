#ifndef LLMRT_KERNELS_H
#define LLMRT_KERNELS_H

#include "types.h"

/*============================================================================
 * Kernel API — Person A owns the implementation
 * 
 * All kernels are STATELESS: they read inputs, write outputs, never allocate.
 * All buffer pointers must be 64-byte aligned.
 *============================================================================*/

/* Matrix multiply: C = A × B
 * A: (M, K), B: (K, N), C: (M, N)   — all row-major, FP32 */
void gemm_f32(const Tensor* A, const Tensor* B, Tensor* C);

/* RMSNorm: output = (input / rms(input)) * weight
 * input: (T, H), weight: (H,), output: (T, H)
 * eps: small constant to avoid division by zero (use 1e-6) */
void rmsnorm(const Tensor* input, const Tensor* weight, Tensor* output, float eps);

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
