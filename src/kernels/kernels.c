#include "kernels.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>

/* ---- Person A's raw AVX2 kernels ---- */

void gemm_f32(const Tensor* A, const Tensor* B, Tensor* C) {
    int M = A->shape[0];
    int K = A->shape[1];
    /* B is stored as (out_features, in_features) which is (N, K) */
    int N = B->shape[0];
    
    float* a_data = (float*)A->data;
    float* b_data = (float*)B->data;
    float* c_data = (float*)C->data;

    // C(i, j) = dot(A[i, :], B[j, :])
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            int k = 0;
            
            // AVX2 vectorized dot product
            __m256 sum_vec = _mm256_setzero_ps();
            for (; k <= K - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&a_data[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&b_data[j * K + k]);
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            
            // Horizontal sum of the AVX2 accumulator
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            sum += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            
            // Remainder loop
            for (; k < K; k++) {
                sum += a_data[i * K + k] * b_data[j * K + k];
            }
            
            c_data[i * N + j] = sum;
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
