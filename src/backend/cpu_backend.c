#include "backend_internal.h"

#include "kernels.h"
#include "kernels_ext.h"
#include "quant.h"

#include <stdlib.h>

Backend* backend_cpu_create(const BackendConfig* cfg) {
    Backend* b = (Backend*)calloc(1, sizeof(Backend));
    if (!b) return NULL;

    b->cfg = *cfg;
    b->cfg.kind = BACKEND_CPU;

    /* Wire existing CPU kernels directly into the vtable. */
    b->kernels.gemm_f32 = gemm_f32;
    b->kernels.gemm_f32_nn = gemm_f32_nn;
    b->kernels.rmsnorm = rmsnorm;
    b->kernels.softmax_inplace = softmax_inplace;
    b->kernels.silu_inplace = silu_inplace;
    b->kernels.rope = rope;
    b->kernels.residual_add = residual_add;
    b->kernels.elemwise_mul = elemwise_mul;

    /* Quant decode hooks (Phase B) */
    b->kernels.matvec_q4k_f32 = matvec_q4k_f32;
    b->kernels.matvec_q8_0_f32 = matvec_q8_0_f32;

    return b;
}

void backend_cpu_destroy(Backend* b) {
    free(b);
}
