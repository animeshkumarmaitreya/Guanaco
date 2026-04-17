#ifndef LLMRT_KERNELS_EXT_H
#define LLMRT_KERNELS_EXT_H

#include "types.h"

/*============================================================================
 * Kernel extensions (MUST) — scaffolding
 *
 * This keeps new kernel entrypoints additive without changing kernels.h yet.
 *============================================================================*/

/* Standard GEMM variant: C = A(M,K) * B(K,N)
 * A: (M,K) row-major, B: (K,N) row-major, C: (M,N) row-major */
void gemm_f32_nn(const Tensor* A, const Tensor* B, Tensor* C);

#endif /* LLMRT_KERNELS_EXT_H */
