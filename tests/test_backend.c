#include "backend.h"
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PASS(name) printf("  PASS: %s\n", name)
#define FAIL(name, msg) do { printf("  FAIL: %s — %s\n", name, msg); failures++; } while(0)

static int failures = 0;

static Tensor make_tensor(float* data, int d0, int d1) {
    Tensor t;
    t.data = data;
    t.ndim = 2;
    t.shape[0] = d0; t.shape[1] = d1;
    t.shape[2] = 0;  t.shape[3] = 0;
    t.stride[0] = d1; t.stride[1] = 1;
    t.stride[2] = 0;  t.stride[3] = 0;
    t.dtype = DTYPE_F32;
    t.byte_size = 0;
    return t;
}

static void test_backend_cpu_gemm_vtable(void) {
    BackendConfig cfg = {0};
    cfg.kind = BACKEND_CPU;
    cfg.threads = 1;
    cfg.device_id = 0;

    Backend* b = backend_create(&cfg);
    if (!b) {
        FAIL("backend_cpu_gemm_vtable", "backend_create returned NULL");
        return;
    }

    if (backend_kind(b) != BACKEND_CPU) {
        FAIL("backend_cpu_gemm_vtable", "expected BACKEND_CPU");
        backend_destroy(b);
        return;
    }

    const KernelVTable* k = backend_kernels(b);
    if (!k || !k->gemm_f32 || !k->rmsnorm || !k->softmax_inplace) {
        FAIL("backend_cpu_gemm_vtable", "missing required vtable entries");
        backend_destroy(b);
        return;
    }

    float a[] = {1,2,3, 4,5,6};
    float bmat[] = {7,8,9, 10,11,12};
    float c[4] = {0};
    float expected[] = {50, 68, 122, 167};

    Tensor A = make_tensor(a, 2, 3);
    Tensor B = make_tensor(bmat, 2, 3);
    Tensor C = make_tensor(c, 2, 2);

    k->gemm_f32(&A, &B, &C);

    for (int i = 0; i < 4; i++) {
        if (fabsf(c[i] - expected[i]) > 1e-5f) {
            FAIL("backend_cpu_gemm_vtable", "gemm_f32 output mismatch");
            backend_destroy(b);
            return;
        }
    }

    /* Quant matvec hooks should exist and currently return failure. */
    if (k->matvec_q4k_f32 && k->matvec_q8_0_f32) {
        Tensor W = {0};
        float x[1] = {0};
        float y[1] = {0};
        if (k->matvec_q4k_f32(&W, x, y) == 0) {
            FAIL("backend_cpu_gemm_vtable", "matvec_q4k_f32 unexpectedly succeeded");
            backend_destroy(b);
            return;
        }
        if (k->matvec_q8_0_f32(&W, x, y) == 0) {
            FAIL("backend_cpu_gemm_vtable", "matvec_q8_0_f32 unexpectedly succeeded");
            backend_destroy(b);
            return;
        }
    }

    PASS("backend_cpu_gemm_vtable");
    backend_destroy(b);
}

static void test_backend_cuda_guard(void) {
    BackendConfig cfg = {0};
    cfg.kind = BACKEND_CUDA;
    cfg.threads = 1;
    cfg.device_id = 0;

    Backend* b = backend_create(&cfg);

#if !defined(USE_CUDA) || (USE_CUDA == 0)
    if (b != NULL) {
        FAIL("backend_cuda_guard", "expected NULL backend when USE_CUDA=0");
        backend_destroy(b);
        return;
    }
    PASS("backend_cuda_guard");
#else
    /* If CUDA is compiled in, it's OK for the stub to still return NULL for now. */
    if (b) backend_destroy(b);
    PASS("backend_cuda_guard");
#endif
}

int main(void) {
    printf("=== Backend Test Suite (scaffolding) ===\n");

    test_backend_cpu_gemm_vtable();
    test_backend_cuda_guard();

    printf("\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    return failures;
}
