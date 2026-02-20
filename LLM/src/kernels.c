#include "kernels.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>

void gemm_f32(const float *A, const float *B, float *C, int M, int N, int K) {
  memset(C, 0, M * N * sizeof(float));
  int BM = 64, BN = 64, BK = 64;

  for (int i0 = 0; i0 < M; i0 += BM) {
    int imax = (i0 + BM > M) ? M : i0 + BM;
    for (int j0 = 0; j0 < N; j0 += BN) {
      int jmax = (j0 + BN > N) ? N : j0 + BN;
      for (int k0 = 0; k0 < K; k0 += BK) {
        int kmax = (k0 + BK > K) ? K : k0 + BK;

        for (int i = i0; i < imax; i++) {
          int j = j0;
          for (; j <= jmax - 8; j += 8) {
            __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
            for (int k = k0; k < kmax; k++) {
              __m256 a_vec = _mm256_set1_ps(A[i * K + k]);
              __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
              c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            }
            _mm256_storeu_ps(&C[i * N + j], c_vec);
          }
          for (; j < jmax; j++) {
            float c_val = C[i * N + j];
            for (int k = k0; k < kmax; k++) {
              c_val += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = c_val;
          }
        }
      }
    }
  }
}

void rmsnorm(float *out, const float *x, const float *weight, int dim,
             float eps) {
  float ss = 0.0f;
  for (int j = 0; j < dim; j++) {
    ss += x[j] * x[j];
  }
  ss /= dim;
  ss += eps;
  ss = 1.0f / sqrtf(ss); // 1 / rms
  for (int j = 0; j < dim; j++) {
    out[j] = (weight ? weight[j] : 1.0f) * (x[j] * ss);
  }
}

void softmax(float *x, int size) {
  float max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val)
      max_val = x[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  if (sum < 1e-6f)
    sum = 1e-6f; // Prevent div by zero if nearly all masked
  for (int i = 0; i < size; i++) {
    x[i] /= sum;
  }
}

void silu_inplace(float *x, int size) {
  for (int i = 0; i < size; i++) {
    x[i] = x[i] / (1.0f + expf(-x[i]));
  }
}

void residual_add(float *out, const float *a, const float *b, int size) {
  for (int i = 0; i < size; i++) {
    out[i] = a[i] + b[i];
  }
}

void rope(float *q, float *k, int head_dim, int pos, int n_heads,
          int n_kv_heads) {
  for (int i = 0; i < head_dim; i += 2) {
    float freq = 1.0f / powf(10000.0f, (float)i / (float)head_dim);
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    for (int h = 0; h < n_heads; h++) {
      float *q_head = q + h * head_dim;
      float q0 = q_head[i];
      float q1 = q_head[i + 1];
      q_head[i] = q0 * cos_val - q1 * sin_val;
      q_head[i + 1] = q0 * sin_val + q1 * cos_val;
    }

    for (int h = 0; h < n_kv_heads; h++) {
      float *k_head = k + h * head_dim;
      float k0 = k_head[i];
      float k1 = k_head[i + 1];
      k_head[i] = k0 * cos_val - k1 * sin_val;
      k_head[i + 1] = k0 * sin_val + k1 * cos_val;
    }
  }
}

void matmul(float *out, const float *x, const float *w, int n, int d) {
  // w is [n x d], x is [d], out is [n]
  for (int i = 0; i < n; i++) {
    __m256 sum256 = _mm256_setzero_ps();
    int j = 0;

    // Prefetch next row
    if (i + 1 < n) {
      _mm_prefetch((const char *)&w[(i + 1) * d], _MM_HINT_T0);
    }

    // Process 8 elements at a time
    for (; j <= d - 8; j += 8) {
      __m256 vx = _mm256_loadu_ps(&x[j]);
      __m256 vw = _mm256_loadu_ps(&w[i * d + j]);
      sum256 = _mm256_fmadd_ps(vx, vw, sum256);
    }

    // Horizontal sum
    __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256),
                               _mm256_extractf128_ps(sum256, 1));
    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 0x55));
    float sum = _mm_cvtss_f32(sum128);

    // Remainder
    for (; j < d; j++) {
      sum += x[j] * w[i * d + j];
    }

    out[i] = sum;
  }
}
