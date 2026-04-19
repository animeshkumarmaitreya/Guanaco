#include "backend.h"
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void test_backend_threads_gemm_matches_single_thread(void) {
    /* This validates correctness (not speed): multi-threaded GEMM must match single-thread GEMM. */
    const int M = 3;
    const int K = 128;
    const int N = 96;

    float* a = (float*)calloc((size_t)M * K, sizeof(float));
    float* b = (float*)calloc((size_t)N * K, sizeof(float));
    float* c1 = (float*)calloc((size_t)M * N, sizeof(float));
    float* c4 = (float*)calloc((size_t)M * N, sizeof(float));
    if (!a || !b || !c1 || !c4) {
        FAIL("backend_threads_gemm_matches_single_thread", "allocation failed");
        free(a); free(b); free(c1); free(c4);
        return;
    }

    srand(123);
    for (int i = 0; i < M * K; i++) a[i] = (float)(rand() % 200 - 100) / 100.0f;
    for (int i = 0; i < N * K; i++) b[i] = (float)(rand() % 200 - 100) / 100.0f;

    Tensor A = make_tensor(a, M, K);
    Tensor B = make_tensor(b, N, K);
    Tensor C1 = make_tensor(c1, M, N);
    Tensor C4 = make_tensor(c4, M, N);

    BackendConfig cfg1 = {0};
    cfg1.kind = BACKEND_CPU;
    cfg1.threads = 1;
    cfg1.device_id = 0;

    Backend* b1 = backend_create(&cfg1);
    if (!b1) {
        FAIL("backend_threads_gemm_matches_single_thread", "backend_create threads=1 returned NULL");
        free(a); free(b); free(c1); free(c4);
        return;
    }
    const KernelVTable* k1 = backend_kernels(b1);
    if (!k1 || !k1->gemm_f32 || !k1->gemm_f32_nn) {
        FAIL("backend_threads_gemm_matches_single_thread", "missing gemm vtable entries");
        backend_destroy(b1);
        free(a); free(b); free(c1); free(c4);
        return;
    }

    memset(c1, 0, (size_t)M * N * sizeof(float));
    k1->gemm_f32(&A, &B, &C1);

    /* gemm_f32_nn path uses B(K,N) layout; build one from b (currently N,K). */
    float* b_kn = (float*)calloc((size_t)K * N, sizeof(float));
    float* c1_nn = (float*)calloc((size_t)M * N, sizeof(float));
    float* c4_nn = (float*)calloc((size_t)M * N, sizeof(float));
    if (!b_kn || !c1_nn || !c4_nn) {
        FAIL("backend_threads_gemm_matches_single_thread", "allocation failed (nn)");
        backend_destroy(b1);
        free(a); free(b); free(c1); free(c4);
        free(b_kn); free(c1_nn); free(c4_nn);
        return;
    }

    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            b_kn[(size_t)k * N + n] = b[(size_t)n * K + k];
        }
    }
    Tensor Bnn = make_tensor(b_kn, K, N);
    Tensor C1nn = make_tensor(c1_nn, M, N);
    Tensor C4nn = make_tensor(c4_nn, M, N);

    memset(c1_nn, 0, (size_t)M * N * sizeof(float));
    k1->gemm_f32_nn(&A, &Bnn, &C1nn);
    backend_destroy(b1);

    BackendConfig cfg4 = {0};
    cfg4.kind = BACKEND_CPU;
    cfg4.threads = 4;
    cfg4.device_id = 0;

    Backend* b4 = backend_create(&cfg4);
    if (!b4) {
        FAIL("backend_threads_gemm_matches_single_thread", "backend_create threads=4 returned NULL");
        free(a); free(b); free(c1); free(c4);
        free(b_kn); free(c1_nn); free(c4_nn);
        return;
    }
    const KernelVTable* k4 = backend_kernels(b4);

    memset(c4, 0, (size_t)M * N * sizeof(float));
    k4->gemm_f32(&A, &B, &C4);

    memset(c4_nn, 0, (size_t)M * N * sizeof(float));
    k4->gemm_f32_nn(&A, &Bnn, &C4nn);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(c1[i] - c4[i]);
        if (err > max_err) max_err = err;
    }
    float max_err_nn = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(c1_nn[i] - c4_nn[i]);
        if (err > max_err_nn) max_err_nn = err;
    }

    if (max_err > 1e-5f || max_err_nn > 1e-5f) {
        FAIL("backend_threads_gemm_matches_single_thread", "output mismatch between threads=1 and threads=4");
    } else {
        PASS("backend_threads_gemm_matches_single_thread");
    }

    backend_destroy(b4);
    free(a); free(b); free(c1); free(c4);
    free(b_kn); free(c1_nn); free(c4_nn);
}

int main(void) {
    printf("=== Backend Test Suite (scaffolding) ===\n");

    test_backend_cpu_gemm_vtable();
    test_backend_threads_gemm_matches_single_thread();
    test_backend_cuda_guard();

    printf("\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    return failures;
}
