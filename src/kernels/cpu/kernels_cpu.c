#include "kernels.h"
#include "kernels_ext.h"
#include <immintrin.h>
#include <assert.h>
#include <math.h>
#include <string.h>

/* ---- CPU kernels (AVX2/FMA) ----
 *
 * This file contains the baseline CPU implementations of the kernel API
 * declared in `src/include/kernels.h`.
 *
 * Rationale for location:
 * - CPU-specific code lives under `src/kernels/cpu/`.
 * - CUDA-specific code will live under `src/kernels/cuda/`.
 * - The backend abstraction selects which implementation to call.
 */

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

    /* Contract: C(M,N) = A(M,K) * B(N,K)^T, with all tensors contiguous row-major */
    assert(A->shape[0] > 0 && A->shape[1] > 0);
    assert(B->shape[0] > 0 && B->shape[1] > 0);
    assert(B->shape[1] == A->shape[1]);
    assert(C->shape[0] == A->shape[0]);
    assert(C->shape[1] == B->shape[0]);

    /* Current implementation assumes contiguous layout and effectively ignores stride fields. */
    assert(tensor_is_contiguous_row_major(A));
    assert(tensor_is_contiguous_row_major(B));
    assert(tensor_is_contiguous_row_major(C));
#endif
    const int M = A->shape[0];
    const int K = A->shape[1];
    /* B is stored as (out_features, in_features) which is (N, K) */
    const int N = B->shape[0];

    const float* a_data = (const float*)A->data;
    const float* b_data = (const float*)B->data;
    float* c_data = (float*)C->data;

    /* C(i, j) = dot(A[i, :], B[j, :]) */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            int k = 0;

            /* AVX2 vectorized dot product */
            __m256 sum_vec = _mm256_setzero_ps();
            for (; k <= K - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&a_data[(size_t)i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&b_data[(size_t)j * K + k]);
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }

            /* Horizontal sum of the AVX2 accumulator */
            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            sum += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

            /* Remainder loop */
            for (; k < K; k++) {
                sum += a_data[(size_t)i * K + k] * b_data[(size_t)j * K + k];
            }

            c_data[(size_t)i * N + j] = sum;
        }
    }
}

void gemm_f32_nn(const Tensor* A, const Tensor* B, Tensor* C) {
    const int M = A->shape[0];
    const int K = A->shape[1];
    const int BK = B->shape[0];
    const int N = B->shape[1];

#ifndef NDEBUG
    assert(A != NULL && B != NULL && C != NULL);
    assert(A->data != NULL && B->data != NULL && C->data != NULL);
    assert(A->ndim == 2);
    assert(B->ndim == 2);
    assert(C->ndim == 2);
    assert(A->dtype == DTYPE_F32);
    assert(B->dtype == DTYPE_F32);
    assert(C->dtype == DTYPE_F32);

    assert(M > 0 && K > 0);
    assert(BK == K);
    assert(C->shape[0] == M);
    assert(C->shape[1] == N);

    /* Current implementation assumes contiguous row-major tensors and ignores stride fields. */
    assert(tensor_is_contiguous_row_major(A));
    assert(tensor_is_contiguous_row_major(B));
    assert(tensor_is_contiguous_row_major(C));
#endif

    const float* a = (const float*)A->data;
    const float* b = (const float*)B->data;
    float* c = (float*)C->data;

    /* Vectorize across N (columns). */
    for (int i = 0; i < M; i++) {
        const float* a_row = a + (size_t)i * K;
        float* c_row = c + (size_t)i * N;

        int j = 0;
        for (; j <= N - 8; j += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                const __m256 b_vec = _mm256_loadu_ps(&b[(size_t)k * N + j]);
                const __m256 a_broadcast = _mm256_set1_ps(a_row[k]);
                acc = _mm256_fmadd_ps(a_broadcast, b_vec, acc);
            }
            _mm256_storeu_ps(&c_row[j], acc);
        }

        /* Remainder columns */
        for (; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += a_row[k] * b[(size_t)k * N + j];
            c_row[j] = sum;
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
    assert(tensor_is_contiguous_row_major(input));
    assert(tensor_is_contiguous_row_major(output));
    if (weight) {
        assert(weight->data != NULL);
        assert(weight->ndim == 1);
        assert(weight->dtype == DTYPE_F32);
        assert(weight->shape[0] == input->shape[1]);
        assert(tensor_is_contiguous_row_major(weight));
    }
#endif
    const int seq_len = input->shape[0];
    const int dim = input->shape[1];

    const float* in_data = (const float*)input->data;
    float* out_data = (float*)output->data;
    const float* w_data = weight ? (const float*)weight->data : NULL;

    for (int i = 0; i < seq_len; i++) {
        const float* x = in_data + (size_t)i * dim;
        float* out = out_data + (size_t)i * dim;

        float ss = 0.0f;
        for (int j = 0; j < dim; j++) ss += x[j] * x[j];
        ss /= dim;
        ss += eps;
        ss = 1.0f / sqrtf(ss); /* 1 / rms */

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
    assert(tensor_is_contiguous_row_major(scores));
#endif
    int total = tensor_numel(scores);
#ifndef NDEBUG
    assert(total % seq_len == 0);
#endif
    const int rows = total / seq_len;
    float* data = (float*)scores->data;

    for (int r = 0; r < rows; r++) {
        float* x = data + (size_t)r * seq_len;

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
        if (sum < 1e-10f) sum = 1e-10f; /* Prevent div by zero if nearly all masked */
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
    assert(tensor_is_contiguous_row_major(x));
#endif
    const int numel = tensor_numel(x);
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
    assert(tensor_is_contiguous_row_major(x));
    assert(tensor_is_contiguous_row_major(residual));
#endif
    const int numel = tensor_numel(x);
    float* x_data = (float*)x->data;
    const float* res_data = (const float*)residual->data;
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
    assert(tensor_is_contiguous_row_major(a));
    assert(tensor_is_contiguous_row_major(b));
#endif
    const int numel = tensor_numel(a);
    float* a_data = (float*)a->data;
    const float* b_data = (const float*)b->data;
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
    assert(tensor_is_contiguous_row_major(q));
    assert(tensor_is_contiguous_row_major(k));
#endif
    const int n_heads = q->shape[0];
    const int n_kv_heads = k->shape[0];

    float* q_data = (float*)q->data;
    float* k_data = (float*)k->data;

    for (int i = 0; i < head_dim; i += 2) {
        const float freq = 1.0f / powf(10000.0f, (float)i / (float)head_dim);
        const float angle = pos * freq;
        const float cos_val = cosf(angle);
        const float sin_val = sinf(angle);

        for (int h = 0; h < n_heads; h++) {
            float* q_head = q_data + (size_t)h * head_dim;
            const float q0 = q_head[i];
            const float q1 = q_head[i + 1];
            q_head[i] = q0 * cos_val - q1 * sin_val;
            q_head[i + 1] = q0 * sin_val + q1 * cos_val;
        }

        for (int h = 0; h < n_kv_heads; h++) {
            float* k_head = k_data + (size_t)h * head_dim;
            const float k0 = k_head[i];
            const float k1 = k_head[i + 1];
            k_head[i] = k0 * cos_val - k1 * sin_val;
            k_head[i + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
}
