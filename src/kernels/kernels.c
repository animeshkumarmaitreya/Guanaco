#include "kernels.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>

/* ---- Person A's raw AVX2 kernels ---- */

void gemm_f32(const Tensor* A, const Tensor* B, Tensor* C) {
    int M = A->shape[0];
    int K = A->shape[1];
    int N = B->shape[1];
    
    float* a_data = (float*)A->data;
    float* b_data = (float*)B->data;
    float* c_data = (float*)C->data;

    memset(c_data, 0, (size_t)M * N * sizeof(float));

    if (M == 1) {
        // Matvec decode path optimization (M=1)
        // A is [1xK], B is [KxN] -> C is [1xN]
        // But his matmul expects w to be [N, K]. B is [K, N].
        // So B is column-major if we want inner dot product, but it's row-major in memory.
        // Let's just do a naive transpose-avoiding matvec or standard block matvec.
        for (int k = 0; k < K; k++) {
            float a_val = a_data[k];
            int j = 0;
            for (; j <= N - 8; j += 8) {
                __m256 c_vec = _mm256_loadu_ps(&c_data[j]);
                __m256 b_vec = _mm256_loadu_ps(&b_data[k * N + j]);
                __m256 a_m256 = _mm256_set1_ps(a_val);
                c_vec = _mm256_fmadd_ps(a_m256, b_vec, c_vec);
                _mm256_storeu_ps(&c_data[j], c_vec);
            }
            for (; j < N; j++) {
                c_data[j] += a_val * b_data[k * N + j];
            }
        }
        return;
    }

    // Standard block GEMM
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
                        __m256 c_vec = _mm256_loadu_ps(&c_data[i * N + j]);
                        for (int k = k0; k < kmax; k++) {
                            __m256 a_vec = _mm256_set1_ps(a_data[i * K + k]);
                            __m256 b_vec = _mm256_loadu_ps(&b_data[k * N + j]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                        }
                        _mm256_storeu_ps(&c_data[i * N + j], c_vec);
                    }
                    for (; j < jmax; j++) {
                        float c_val = c_data[i * N + j];
                        for (int k = k0; k < kmax; k++) {
                            c_val += a_data[i * K + k] * b_data[k * N + j];
                        }
                        c_data[i * N + j] = c_val;
                    }
                }
            }
        }
    }
}

void rmsnorm(const Tensor* input, const Tensor* weight, Tensor* output, float eps) {
    int seq_len = input->shape[0];
    int dim = input->shape[1];
    
    float* in_data = (float*)input->data;
    float* out_data = (float*)output->data;
    float* w_data = weight ? (float*)weight->data : NULL;

    for (int i = 0; i < seq_len; i++) {
        float* x = in_data + i * dim;
        float* out = out_data + i * dim;
        
        float ss = 0.0f;
        for (int j = 0; j < dim; j++) ss += x[j] * x[j];
        ss /= dim;
        ss += eps;
        ss = 1.0f / sqrtf(ss); // 1 / rms
        
        for (int j = 0; j < dim; j++) {
            out[j] = (w_data ? w_data[j] : 1.0f) * (x[j] * ss);
        }
    }
}

void softmax_inplace(Tensor* scores, int seq_len) {
    int total = tensor_numel(scores);
    int rows = total / seq_len;
    float* data = (float*)scores->data;

    for (int r = 0; r < rows; r++) {
        float* x = data + r * seq_len;
        
        float max_val = x[0];
        for (int i = 1; i < seq_len; i++) {
            if (x[i] > max_val) max_val = x[i];
        }
        
        /* Handle all -INFINITY case */
        if (isinf(max_val) && max_val < 0) {
            for (int i = 0; i < seq_len; i++) x[i] = 0.0f;
            continue;
        }

        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            x[i] = expf(x[i] - max_val);
            sum += x[i];
        }
        if (sum < 1e-10f) sum = 1e-10f; // Prevent div by zero if nearly all masked
        for (int i = 0; i < seq_len; i++) {
            x[i] /= sum;
        }
    }
}

void silu_inplace(Tensor* x) {
    int numel = tensor_numel(x);
    float* data = (float*)x->data;
    for (int i = 0; i < numel; i++) {
        data[i] = data[i] / (1.0f + expf(-data[i]));
    }
}

void residual_add(Tensor* x, const Tensor* residual) {
    int numel = tensor_numel(x);
    float* x_data = (float*)x->data;
    float* res_data = (float*)residual->data;
    for (int i = 0; i < numel; i++) {
        x_data[i] += res_data[i];
    }
}

void elemwise_mul(Tensor* a, const Tensor* b) {
    int numel = tensor_numel(a);
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    for (int i = 0; i < numel; i++) {
        a_data[i] *= b_data[i];
    }
}

void rope(Tensor* q, Tensor* k, int pos, int head_dim) {
    int n_heads = q->shape[0];
    int n_kv_heads = k->shape[0];
    
    float* q_data = (float*)q->data;
    float* k_data = (float*)k->data;

    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(10000.0f, (float)i / (float)head_dim);
        float angle = pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        for (int h = 0; h < n_heads; h++) {
            float* q_head = q_data + h * head_dim;
            float q0 = q_head[i];
            float q1 = q_head[i + 1];
            q_head[i] = q0 * cos_val - q1 * sin_val;
            q_head[i + 1] = q0 * sin_val + q1 * cos_val;
        }

        for (int h = 0; h < n_kv_heads; h++) {
            float* k_head = k_data + h * head_dim;
            float k0 = k_head[i];
            float k1 = k_head[i + 1];
            k_head[i] = k0 * cos_val - k1 * sin_val;
            k_head[i + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
}
