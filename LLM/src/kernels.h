#ifndef KERNELS_H
#define KERNELS_H

// Core compute primitives. Stateless. Never allocate.
void gemm_f32(const float* A, const float* B, float* C, int M, int N, int K);
void rmsnorm(float* out, const float* x, const float* weight, int dim, float eps);
void softmax(float* x, int size);
void silu_inplace(float* x, int size);
void residual_add(float* out, const float* a, const float* b, int size);
void rope(float* q, float* k, int head_dim, int pos, int n_heads, int n_kv_heads);
void matmul(float* out, const float* x, const float* w, int n, int d);
// matmul = specialized M=1 GEMM (matvec) for decode hot path

#endif // KERNELS_H
