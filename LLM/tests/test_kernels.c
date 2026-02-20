#include "../src/kernels.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Reference Naive GEMM for testing max error. Since we are testing our naive
// GEMM, we just need a structurally sound way to generate C_ref or we can
// compare against it. Wait, we are writing Naive GEMM now. Let's just implement
// a reference loop here and use it to test against our `gemm_f32`. Our GEMM
// will later be replaced with tiled AVX2, so having this reference is perfect.
void reference_gemm(const float *A, const float *B, float *C, int M, int N,
                    int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      double sum = 0.0; // Use double for higher precision in reference
      for (int k = 0; k < K; k++) {
        sum += (double)A[i * K + k] * (double)B[k * N + j];
      }
      C[i * N + j] = (float)sum;
    }
  }
}

void run_single_gemm_test(int M, int N, int K, const char *name) {
  printf("Testing %s (%dx%dx%d)... ", name, M, N, K);

  // Allocate memory
  float *A = (float *)malloc(M * K * sizeof(float));
  float *B = (float *)malloc(K * N * sizeof(float));
  float *C_ref = (float *)malloc(M * N * sizeof(float));
  float *C_test = (float *)malloc(M * N * sizeof(float));

  // Initialize with random values between -1 and 1
  for (int i = 0; i < M * K; i++) {
    A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }
  for (int i = 0; i < K * N; i++) {
    B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  }

  memset(C_ref, 0, M * N * sizeof(float));
  memset(C_test, 0, M * N * sizeof(float));

  // Compute reference and test
  reference_gemm(A, B, C_ref, M, N, K);
  gemm_f32(A, B, C_test, M, N, K);

  // Check max error
  float max_err = 0.0f;
  for (int i = 0; i < M * N; i++) {
    float err = fabs(C_ref[i] - C_test[i]);
    if (err > max_err) {
      max_err = err;
    }
  }

  if (max_err < 3e-4f) {
    printf("PASS (max error = %e)\n", max_err);
  } else {
    printf("FAIL (max error = %e)\n", max_err);
    assert(max_err < 3e-4f);
  }

  free(A);
  free(B);
  free(C_ref);
  free(C_test);
}

void test_gemm_shapes() {
  printf("Running GEMM shapes tests...\n");
  srand(42);

  // 12 specific shape configs from Person A's tasks
  run_single_gemm_test(1, 2048, 2048, "Decode hot path - matvec");
  run_single_gemm_test(1, 5504, 2048, "Decode MLP up-projection");
  run_single_gemm_test(1, 2048, 5504, "Decode MLP down-projection");
  run_single_gemm_test(64, 2048, 2048, "Prefill small prompt");
  // run_single_gemm_test(512, 2048, 2048, "Prefill medium prompt"); // SLOW!
  // run_single_gemm_test(512, 5504, 2048, "Prefill MLP up"); // SLOW!
  run_single_gemm_test(1, 1, 1, "Edge case: single element");
  run_single_gemm_test(3, 7, 5, "Edge case: non-power-of-2");
  run_single_gemm_test(1, 128, 128, "Attention: Q x K.T (per-head, decode)");
  run_single_gemm_test(512, 512, 128, "Attention: Q x K.T (per-head, prefill)");
  run_single_gemm_test(512, 128, 512, "Attention: attn x V (prefill)");
  run_single_gemm_test(
      1, 1024, 128,
      "Attention: Q x K.T at seq=1024 (decode)"); // [1x128] x [128x1024] =
                                                  // [1x1024], so M=1, N=1024,
                                                  // K=128
}

void reference_matmul(float *out, const float *x, const float *w, int n,
                      int d) {
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < d; j++) {
      sum += (double)x[j] * (double)w[i * d + j];
    }
    out[i] = (float)sum;
  }
}

void test_matmul_shape(int n, int d, const char *name) {
  printf("Testing matmul %s (nx=%d, d=%d)... ", name, n, d);
  float *x = (float *)malloc(d * sizeof(float));
  float *w = (float *)malloc(n * d * sizeof(float));
  float *out_ref = (float *)malloc(n * sizeof(float));
  float *out_test = (float *)malloc(n * sizeof(float));

  for (int i = 0; i < d; i++)
    x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  for (int i = 0; i < n * d; i++)
    w[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  reference_matmul(out_ref, x, w, n, d);
  matmul(out_test, x, w, n, d);

  float max_err = 0.0f;
  for (int i = 0; i < n; i++) {
    float err = fabs(out_ref[i] - out_test[i]);
    if (err > max_err)
      max_err = err;
  }

  if (max_err < 3e-4f) {
    printf("PASS (max error = %e)\n", max_err);
  } else {
    printf("FAIL (max error = %e)\n", max_err);
    assert(max_err < 3e-4f);
  }

  free(x);
  free(w);
  free(out_ref);
  free(out_test);
}

void test_matmul() {
  printf("\nRunning Matmul tests...\n");
  test_matmul_shape(2048, 2048, "Decode hot path - matvec");
  test_matmul_shape(5504, 2048, "Decode MLP up-projection");
  test_matmul_shape(2048, 5504, "Decode MLP down-projection");
}

void test_softmax() {
  printf("\nRunning Softmax tests...\n");
  float x1[3] = {1.0f, 2.0f, 3.0f};
  softmax(x1, 3);
  assert(fabs(x1[0] - 0.09003f) < 1e-4f);
  assert(fabs(x1[1] - 0.24472f) < 1e-4f);
  assert(fabs(x1[2] - 0.66524f) < 1e-4f);

  float x2[3] = {0.0f, 0.0f, 0.0f};
  softmax(x2, 3);
  assert(fabs(x2[0] - 0.33333f) < 1e-4f);

  float x3[3] = {1000.0f, 1000.0f, 1000.0f};
  softmax(x3, 3);
  assert(fabs(x3[0] - 0.33333f) < 1e-4f);

  printf("Softmax PASS\n");
}

void test_rmsnorm() {
  printf("\nRunning RMSNorm tests...\n");
  float x[4] = {1.0f, -1.0f, 1.0f, -1.0f};
  float weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float out[4];
  rmsnorm(out, x, weight, 4, 1e-6f);

  float ss = 0.0f;
  for (int i = 0; i < 4; i++)
    ss += out[i] * out[i];
  ss /= 4.0f;
  assert(fabs(ss - 1.0f) < 1e-4f);

  printf("RMSNorm PASS\n");
}

void test_silu() {
  printf("\nRunning SiLU tests...\n");
  float x[3] = {0.0f, 1.0f, -1.0f};
  silu_inplace(x, 3);
  assert(fabs(x[0] - 0.0f) < 1e-4f);
  assert(fabs(x[1] - (1.0f / (1.0f + expf(-1.0f)))) < 1e-4f);
  assert(fabs(x[2] - (-1.0f / (1.0f + expf(1.0f)))) < 1e-4f);
  printf("SiLU PASS\n");
}

void test_residual() {
  printf("\nRunning Residual tests...\n");
  float a[3] = {1.0f, 2.0f, 3.0f};
  float b[3] = {-1.0f, -2.0f, -3.0f};
  float out[3];
  residual_add(out, a, b, 3);
  assert(out[0] == 0.0f && out[1] == 0.0f && out[2] == 0.0f);
  printf("Residual PASS\n");
}

void test_rope() {
  printf("\nRunning RoPE tests...\n");
  float q[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float k[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  rope(q, k, 4, 0, 1, 1);
  assert(fabs(q[0] - 1.0f) < 1e-4f &&
         fabs(q[1] - 1.0f) < 1e-4f); // pos=0 -> identity
  printf("RoPE PASS\n");
}

int main() {
  test_gemm_shapes();
  test_matmul();
  test_softmax();
  test_rmsnorm();
  test_silu();
  test_residual();
  test_rope();

  printf("\nAll kernel tests passed!\n");
  return 0;
}
