/*============================================================================
 * Kernel Stubs — Person A replaces these with real implementations
 *
 * These stubs fill outputs with small deterministic values so the pipeline
 * doesn't NaN. They are NOT correct — they exist only so the integration
 * code can test wiring, shapes, and control flow.
 *============================================================================*/

#include "kernels.h"
#include <math.h>
#include <string.h>

void gemm_f32(const Tensor* A, const Tensor* B, Tensor* C) {
    (void)A; (void)B;
    /* Stub: fill C with small deterministic values */
    int numel = tensor_numel(C);
    float* out = (float*)C->data;
    for (int i = 0; i < numel; i++) {
        out[i] = 0.01f * (float)(i % 7);
    }
}

void rmsnorm(const Tensor* input, const Tensor* weight, Tensor* output, float eps) {
    /* Stub: actually do a real rmsnorm — this is cheap and prevents NaN */
    int T = input->shape[0];
    int H = input->shape[1];
    const float* x = (const float*)input->data;
    const float* w = (const float*)weight->data;
    float* o = (float*)output->data;

    for (int t = 0; t < T; t++) {
        float ss = 0.0f;
        for (int h = 0; h < H; h++) {
            ss += x[t * H + h] * x[t * H + h];
        }
        float rms = sqrtf(ss / H + eps);
        for (int h = 0; h < H; h++) {
            o[t * H + h] = (x[t * H + h] / rms) * w[h];
        }
    }
}

void softmax_inplace(Tensor* scores, int seq_len) {
    /* Stub: real softmax — needed for non-NaN pipeline output */
    float* s = (float*)scores->data;
    int total = tensor_numel(scores);
    int rows = total / seq_len;

    for (int r = 0; r < rows; r++) {
        float* row = s + r * seq_len;
        float max_val = row[0];
        for (int i = 1; i < seq_len; i++) {
            if (row[i] > max_val) max_val = row[i];
        }
        /* Handle all -INFINITY case */
        if (isinf(max_val) && max_val < 0) {
            for (int i = 0; i < seq_len; i++) row[i] = 0.0f;
            continue;
        }
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            row[i] = expf(row[i] - max_val);
            sum += row[i];
        }
        if (sum < 1e-10f) sum = 1e-10f;
        for (int i = 0; i < seq_len; i++) {
            row[i] /= sum;
        }
    }
}

void silu_inplace(Tensor* x) {
    /* Stub: real SiLU — cheap */
    float* d = (float*)x->data;
    int n = tensor_numel(x);
    for (int i = 0; i < n; i++) {
        d[i] = d[i] / (1.0f + expf(-d[i]));
    }
}

void rope(Tensor* q, Tensor* k, int pos, int head_dim) {
    /* Stub: no-op (identity rotation) */
    (void)q; (void)k; (void)pos; (void)head_dim;
}

void residual_add(Tensor* x, const Tensor* residual) {
    float* xd = (float*)x->data;
    const float* rd = (const float*)residual->data;
    int n = tensor_numel(x);
    for (int i = 0; i < n; i++) {
        xd[i] += rd[i];
    }
}

void elemwise_mul(Tensor* a, const Tensor* b) {
    float* ad = (float*)a->data;
    const float* bd = (const float*)b->data;
    int n = tensor_numel(a);
    for (int i = 0; i < n; i++) {
        ad[i] *= bd[i];
    }
}
