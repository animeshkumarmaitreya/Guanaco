#include "kernels.h"
#include <immintrin.h>
#include <assert.h>
#include <math.h>
#include <string.h>

/* ---- Person A's raw AVX2 kernels ---- */

void gemm_f32(const Tensor* A, const Tensor* B, Tensor* C) {
#ifndef NDEBUG
    assert(A != NULL && B != NULL && C != NULL);
    assert(A->data != NULL && B->data != NULL && C->data != NULL);
    assert(A->ndim == 2);
    assert(B->ndim == 2);
    assert(C->ndim == 2);
    assert(A->dtype == DTYPE_F32);
    assert(B->dtype == DTYPE_F32);
    assert(C->dtype == DTYPE_F32);

    // Contract: C(M,N) = A(M,K) * B(N,K)^T, with all tensors contiguous row-major
    assert(A->shape[0] > 0 && A->shape[1] > 0);
    assert(B->shape[0] > 0 && B->shape[1] > 0);
    assert(B->shape[1] == A->shape[1]);
    assert(C->shape[0] == A->shape[0]);
    assert(C->shape[1] == B->shape[0]);

    // Current implementation assumes contiguous layout and ignores stride fields.
    assert(A->stride[0] == A->shape[1] && A->stride[1] == 1);
    assert(B->stride[0] == B->shape[1] && B->stride[1] == 1);
    assert(C->stride[0] == C->shape[1] && C->stride[1] == 1);
#endif
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
#ifndef NDEBUG
    assert(input != NULL && output != NULL);
    assert(input->data != NULL && output->data != NULL);
    assert(input->ndim == 2);
    assert(output->ndim == 2);
    assert(input->dtype == DTYPE_F32);
    assert(output->dtype == DTYPE_F32);
    assert(output->shape[0] == input->shape[0]);
    assert(output->shape[1] == input->shape[1]);
    if (weight) {
        assert(weight->data != NULL);
        assert(weight->ndim == 1);
        assert(weight->dtype == DTYPE_F32);
        assert(weight->shape[0] == input->shape[1]);
    }
#endif
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
#ifndef NDEBUG
    assert(scores != NULL);
    assert(scores->data != NULL);
    assert(scores->dtype == DTYPE_F32);
    assert(seq_len > 0);
#endif
    int total = tensor_numel(scores);
#ifndef NDEBUG
    assert(total % seq_len == 0);
#endif
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
#ifndef NDEBUG
    assert(x != NULL);
    assert(x->data != NULL);
    assert(x->dtype == DTYPE_F32);
#endif
    int numel = tensor_numel(x);
    float* data = (float*)x->data;
    for (int i = 0; i < numel; i++) {
        data[i] = data[i] / (1.0f + expf(-data[i]));
    }
}

void residual_add(Tensor* x, const Tensor* residual) {
#ifndef NDEBUG
    assert(x != NULL && residual != NULL);
    assert(x->data != NULL && residual->data != NULL);
    assert(x->dtype == DTYPE_F32);
    assert(residual->dtype == DTYPE_F32);
    assert(tensor_numel(x) == tensor_numel(residual));
#endif
    int numel = tensor_numel(x);
    float* x_data = (float*)x->data;
    float* res_data = (float*)residual->data;
    for (int i = 0; i < numel; i++) {
        x_data[i] += res_data[i];
    }
}

void elemwise_mul(Tensor* a, const Tensor* b) {
#ifndef NDEBUG
    assert(a != NULL && b != NULL);
    assert(a->data != NULL && b->data != NULL);
    assert(a->dtype == DTYPE_F32);
    assert(b->dtype == DTYPE_F32);
    assert(tensor_numel(a) == tensor_numel(b));
#endif
    int numel = tensor_numel(a);
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    for (int i = 0; i < numel; i++) {
        a_data[i] *= b_data[i];
    }
}

void rope(Tensor* q, Tensor* k, int pos, int head_dim) {
#ifndef NDEBUG
    assert(q != NULL && k != NULL);
    assert(q->data != NULL && k->data != NULL);
    assert(q->dtype == DTYPE_F32 && k->dtype == DTYPE_F32);
    assert(q->ndim == 2 && k->ndim == 2);
    assert(q->shape[1] == head_dim);
    assert(k->shape[1] == head_dim);
    assert(q->stride[1] == 1);
    assert(k->stride[1] == 1);
#endif
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
