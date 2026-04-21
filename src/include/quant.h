#ifndef LLMRT_QUANT_H
#define LLMRT_QUANT_H

#include "types.h"

/*============================================================================
 * Quantization (MUST) — Phase B
 *
 * CPU decode matvec entrypoints for quantized weights.
 * These are stubs; the real implementation will match GGML's on-disk block layouts.
 *============================================================================*/

/* y(N) = W(N,K) * x(K) where W is Q4_K and x/y are float. Returns 0 on success. */
int matvec_q4k_f32(const Tensor* W_q4k, const float* x, float* y);

/* y(N) = W(N,K) * x(K) where W is Q6_K and x/y are float. Returns 0 on success. */
int matvec_q6k_f32(const Tensor* W_q6k, const float* x, float* y);

/* y(N) = W(N,K) * x(K) where W is Q8_0 and x/y are float. Returns 0 on success. */
int matvec_q8_0_f32(const Tensor* W_q8_0, const float* x, float* y);

#endif /* LLMRT_QUANT_H */
