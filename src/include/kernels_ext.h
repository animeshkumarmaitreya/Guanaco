#ifndef LLMRT_KERNELS_EXT_H
#define LLMRT_KERNELS_EXT_H

#include "types.h"

/*============================================================================
 * Kernel extensions (MUST) — scaffolding
 *
 * This keeps new kernel entrypoints additive without changing kernels.h yet.
 *
 * Layout contract matches kernels.h: contiguous row-major buffers are required;
 * stride is not a general strided-tensor contract.
 *============================================================================*/

/* Standard GEMM variant: C = A(M,K) * B(K,N)
 *
 * Naming note:
 * - The `_nn` suffix follows the conventional GEMM transpose-flag notation:
 *   A is Not-transposed, B is Not-transposed.
 * - This contrasts with `gemm_f32()` (in kernels.h), which computes A * B^T
 *   for GGUF weight layout.
 *
 * Contract: A: (M,K) row-major, B: (K,N) row-major, C: (M,N) row-major.
 */
void gemm_f32_nn(const Tensor* A, const Tensor* B, Tensor* C);

#endif /* LLMRT_KERNELS_EXT_H */
