#ifndef LLMRT_KERNELS_CUDA_H
#define LLMRT_KERNELS_CUDA_H

#include "types.h"

/*============================================================================
 * CUDA kernel wrappers (MUST Phase A) — scaffolding
 *
 * The real implementation will live in .cu files and expose C-callable wrappers.
 * This header is intentionally minimal and can be included from pure C.
 *============================================================================*/

#ifdef __cplusplus
extern "C" {
#endif

int cuda_kernels_init(int device_id);
void cuda_kernels_shutdown(void);

int cuda_gemm_f32(const Tensor* A, const Tensor* B, Tensor* C);
int cuda_gemm_f32_nn(const Tensor* A, const Tensor* B, Tensor* C);

#ifdef __cplusplus
}
#endif

#endif /* LLMRT_KERNELS_CUDA_H */
