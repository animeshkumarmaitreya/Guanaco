/*============================================================================
 * Kernel Test Suite — Person A's test harness
 *
 * Run: make test_kernels && ./build/test_kernels
 * All tests use assert() + tolerance checks. PASS/FAIL printed per test.
 *============================================================================*/

#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

/* Define STUB_MODE to skip tests that require real implementations.
 * The Makefile passes -DSTUB_MODE when building against stubs. */
#ifndef STUB_MODE
#define STUB_MODE 0
#endif

#define TOLERANCE 1e-5f
#define PASS(name)  printf("  PASS: %s\n", name)
#define FAIL(name, msg) do { printf("  FAIL: %s — %s\n", name, msg); failures++; } while(0)
#define SKIP(name)  printf("  SKIP: %s (stub mode)\n", name)

static int failures = 0;

/* Helper: create a simple FP32 tensor */
static Tensor make_tensor(float* data, int d0, int d1) {
    Tensor t;
    t.data = data;
    t.ndim = 2;
    t.shape[0] = d0; t.shape[1] = d1;
    t.shape[2] = 0;  t.shape[3] = 0;
    t.stride[0] = d1; t.stride[1] = 1;
    t.stride[2] = 0;  t.stride[3] = 0;
    t.dtype = DTYPE_F32;
    return t;
}

static Tensor make_tensor_1d(float* data, int d0) {
    Tensor t;
    t.data = data;
    t.ndim = 1;
    t.shape[0] = d0;
    t.shape[1] = 0; t.shape[2] = 0; t.shape[3] = 0;
    t.stride[0] = 1;
    t.stride[1] = 0; t.stride[2] = 0; t.stride[3] = 0;
    t.dtype = DTYPE_F32;
    return t;
}

/* ---- GEMM tests ---- */

static void test_gemm_small(void) {
    if (STUB_MODE) { SKIP("gemm_small"); return; }
    /* gemm_f32 computes C = A(2x3) × B(2x3)^T = C(2x2)
     * i.e., C(i,j) = dot(A[i,:], B[j,:]) */
    float a[] = {1,2,3, 4,5,6};
    float b[] = {7,8,9, 10,11,12};
    float c[4] = {0};
    float expected[] = {50, 68, 122, 167};

    Tensor A = make_tensor(a, 2, 3);
    Tensor B = make_tensor(b, 2, 3);
    Tensor C = make_tensor(c, 2, 2);

    gemm_f32(&A, &B, &C);

    for (int i = 0; i < 4; i++) {
        if (fabsf(c[i] - expected[i]) > TOLERANCE) {
            FAIL("gemm_small", "output mismatch");
            return;
        }
    }
    PASS("gemm_small");
}

static void test_gemm_matvec(void) {
    if (STUB_MODE) { SKIP("gemm_matvec"); return; }
    /* gemm_f32 computes C = A(1xK) × B(NxK)^T = C(1xN)
     * i.e., each output is a dot-product against one row of B */
    int K = 64, N = 64;
    float* a = (float*)calloc(K, sizeof(float));
    float* b = (float*)calloc(K * N, sizeof(float));
    float* c = (float*)calloc(N, sizeof(float));
    float* ref = (float*)calloc(N, sizeof(float));

    srand(42);
    for (int i = 0; i < K; i++) a[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) b[i] = (float)(rand() % 100) / 100.0f;

    /* Reference: ref[j] = dot(a[:], b_row_j[:]) */
    for (int j = 0; j < N; j++) {
        ref[j] = 0;
        for (int k = 0; k < K; k++) {
            ref[j] += a[k] * b[j * K + k];
        }
    }

    Tensor A = make_tensor(a, 1, K);
    Tensor B = make_tensor(b, N, K);
    Tensor C = make_tensor(c, 1, N);
    gemm_f32(&A, &B, &C);

    float max_err = 0;
    for (int j = 0; j < N; j++) {
        float err = fabsf(c[j] - ref[j]);
        if (err > max_err) max_err = err;
    }

    if (max_err > TOLERANCE) {
        FAIL("gemm_matvec", "max error too high");
    } else {
        PASS("gemm_matvec");
    }
    free(a); free(b); free(c); free(ref);
}

/* ---- Softmax tests ---- */

static void test_softmax_basic(void) {
    float scores[] = {1.0f, 2.0f, -0.5f, 3.0f, 0.1f, -2.0f, 4.0f, 1.5f};
    Tensor t = make_tensor_1d(scores, 8);
    softmax_inplace(&t, 8);

    float sum = 0;
    for (int i = 0; i < 8; i++) sum += scores[i];
    if (fabsf(sum - 1.0f) > 1e-6f) {
        FAIL("softmax_basic", "doesn't sum to 1.0");
    } else {
        PASS("softmax_basic");
    }
}

static void test_softmax_all_same(void) {
    float scores[] = {5.0f, 5.0f, 5.0f, 5.0f};
    Tensor t = make_tensor_1d(scores, 4);
    softmax_inplace(&t, 4);

    int ok = 1;
    for (int i = 0; i < 4; i++) {
        if (fabsf(scores[i] - 0.25f) > 1e-6f) ok = 0;
    }
    if (ok) PASS("softmax_all_same");
    else    FAIL("softmax_all_same", "expected uniform 0.25");
}

static void test_softmax_neginf(void) {
    float scores[] = {-INFINITY, -INFINITY, -INFINITY};
    Tensor t = make_tensor_1d(scores, 3);
    softmax_inplace(&t, 3);

    int has_nan = 0;
    for (int i = 0; i < 3; i++) {
        if (isnan(scores[i])) has_nan = 1;
    }
    if (has_nan) FAIL("softmax_neginf", "produced NaN");
    else         PASS("softmax_neginf");
}

static void test_softmax_single(void) {
    float scores[] = {42.0f};
    Tensor t = make_tensor_1d(scores, 1);
    softmax_inplace(&t, 1);

    if (fabsf(scores[0] - 1.0f) > 1e-6f) {
        FAIL("softmax_single", "expected 1.0");
    } else {
        PASS("softmax_single");
    }
}

/* ---- RMSNorm tests ---- */

static void test_rmsnorm_unit(void) {
    int H = 64;
    float input[64], weight[64], output[64];
    srand(123);
    for (int i = 0; i < H; i++) {
        input[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
        weight[i] = 1.0f;
    }

    Tensor in = make_tensor(input, 1, H);
    Tensor w  = make_tensor_1d(weight, H);
    Tensor out = make_tensor(output, 1, H);
    rmsnorm(&in, &w, &out, 1e-6f);

    /* Check RMS of output ≈ 1.0 */
    float ss = 0;
    for (int i = 0; i < H; i++) ss += output[i] * output[i];
    float rms = sqrtf(ss / H);

    if (fabsf(rms - 1.0f) > 0.01f) {
        FAIL("rmsnorm_unit", "RMS of output not ≈ 1.0");
    } else {
        PASS("rmsnorm_unit");
    }
}

/* ---- SiLU test ---- */

static void test_silu(void) {
    float x[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float ref[5];
    for (int i = 0; i < 5; i++) {
        ref[i] = x[i] / (1.0f + expf(-x[i]));
    }

    Tensor t = make_tensor_1d(x, 5);
    silu_inplace(&t);

    float max_err = 0;
    for (int i = 0; i < 5; i++) {
        float err = fabsf(x[i] - ref[i]);
        if (err > max_err) max_err = err;
    }
    if (max_err > 1e-6f) FAIL("silu", "output mismatch");
    else                  PASS("silu");
}

/* ---- Residual add test ---- */

static void test_residual_add(void) {
    float x[] = {1, 2, 3, 4};
    float r[] = {10, 20, 30, 40};
    float expected[] = {11, 22, 33, 44};

    Tensor tx = make_tensor_1d(x, 4);
    Tensor tr = make_tensor_1d(r, 4);
    residual_add(&tx, &tr);

    int ok = 1;
    for (int i = 0; i < 4; i++) {
        if (fabsf(x[i] - expected[i]) > 1e-6f) ok = 0;
    }
    if (ok) PASS("residual_add");
    else    FAIL("residual_add", "output mismatch");
}

/* ---- Main ---- */

int main(void) {
    printf("=== Kernel Test Suite ===\n");

    test_gemm_small();
    test_gemm_matvec();
    test_softmax_basic();
    test_softmax_all_same();
    test_softmax_neginf();
    test_softmax_single();
    test_rmsnorm_unit();
    test_silu();
    test_residual_add();

    printf("\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    return failures;
}
