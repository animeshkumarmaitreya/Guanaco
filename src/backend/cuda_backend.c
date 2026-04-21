#include "backend_internal.h"

#include <stdlib.h>

/*
 * CUDA backend stubs (Phase A MUST).
 *
 * This file is intentionally pure C (no CUDA headers) so CPU-only builds work.
 * The real implementation will live behind USE_CUDA=1 and call into .cu wrappers.
 */

#include "kernels.h"
#include "kernels_ext.h"
#include "quant.h"
#include "threadpool.h"

extern void kernels_cpu_set_threadpool(ThreadPool* tp);

#ifdef USE_CUDA
#include <nvml.h>
extern int cuda_matvec_q4k_f32_impl(const void *W_data, const float *x, float *y, int num_rows, int num_cols);

static nvmlDevice_t g_nvml_device = NULL;
static int g_nvml_ok = 0;

int cuda_get_temperature(void) {
    if (!g_nvml_ok) return 0;
    unsigned int temp = 0;
    if (nvmlDeviceGetTemperature(g_nvml_device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
        return (int)temp;
    }
    return 0;
}

static int cuda_matvec_q4k_wrapper(const Tensor* W, const float* x, float* y) {
    if (W->device_residency == 0) {
        return matvec_q4k_f32(W, x, y);
    }
    return cuda_matvec_q4k_f32_impl(W->data, x, y, W->shape[0], W->shape[1]);
}
#endif

Backend* backend_cuda_create(const BackendConfig* cfg) {
    Backend* b = (Backend*)calloc(1, sizeof(Backend));
    if (!b) return NULL;

    b->cfg = *cfg;
    b->cfg.kind = BACKEND_CUDA;

#ifdef USE_CUDA
    if (nvmlInit() == NVML_SUCCESS) {
        if (nvmlDeviceGetHandleByIndex(b->cfg.device_id, &g_nvml_device) == NVML_SUCCESS) {
            g_nvml_ok = 1;
        }
    }

    ThreadPool* tp = threadpool_create(b->cfg.threads);
    b->impl = tp;
    kernels_cpu_set_threadpool(tp);

    /* Bind heterogeneous CPU fallback wrappers! */
    b->kernels.gemm_f32 = gemm_f32;
    b->kernels.gemm_f32_nn = gemm_f32_nn;
    b->kernels.rmsnorm = rmsnorm; 
    b->kernels.softmax_inplace = softmax_inplace;
    b->kernels.silu_inplace = silu_inplace;
    b->kernels.rope = rope;
    b->kernels.residual_add = residual_add;
    b->kernels.elemwise_mul = elemwise_mul;
    
    b->kernels.matvec_q4k_f32 = cuda_matvec_q4k_wrapper;
    b->kernels.matvec_q6k_f32 = matvec_q6k_f32;
    b->kernels.matvec_q8_0_f32 = matvec_q8_0_f32;

    return b;
#else
    /* Not implemented yet: until Phase A lands, treat as unavailable. */
    free(b);
    return NULL;
#endif
}

void backend_cuda_destroy(Backend* b) {
    if (!b) return;
#ifdef USE_CUDA
    if (g_nvml_ok) {
        nvmlShutdown();
        g_nvml_ok = 0;
    }
    if (b->impl) {
        kernels_cpu_set_threadpool(NULL);
        threadpool_destroy((ThreadPool*)b->impl);
        b->impl = NULL;
    }
#endif
    free(b);
}
