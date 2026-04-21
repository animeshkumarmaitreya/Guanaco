/*============================================================================
 * cuda_layer_kernels.cu — GPU kernels for elementwise transformer ops
 *
 * These kernels keep activation data resident in VRAM, eliminating
 * Host↔Device transfers between consecutive operations within a layer.
 *============================================================================*/

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ---- RMSNorm kernel ---- 
 * output[i] = weight[i] * (input[i] / rms(input))
 * One block processes one row of dim elements. */
__global__ void cuda_rmsnorm_kernel(const float* __restrict__ input,
                                     const float* __restrict__ weight,
                                     float* __restrict__ output,
                                     int dim, float eps) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    /* Sum of squares via parallel reduction */
    float local_ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = input[i];
        local_ss += v * v;
    }
    sdata[tid] = local_ss;
    __syncthreads();
    
    /* Tree reduction in shared memory */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    float rms_inv = rsqrtf(sdata[0] / dim + eps);
    
    /* Scale and write output */
    for (int i = tid; i < dim; i += blockDim.x) {
        output[i] = weight[i] * (input[i] * rms_inv);
    }
}

/* ---- RoPE kernel ----
 * Applies rotary position embedding to Q and K heads.
 * Each thread handles one pair (i, i+1) of the head dimension. */
__global__ void cuda_rope_kernel(float* __restrict__ q, float* __restrict__ k,
                                  int n_q_heads, int n_kv_heads,
                                  int head_dim, int pos) {
    int pair = threadIdx.x;  /* which dimension pair (0..head_dim/2-1) */
    int i = pair * 2;
    if (i >= head_dim) return;
    
    float freq = 1.0f / powf(10000.0f, (float)i / (float)head_dim);
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);
    
    /* Apply to all Q heads */
    for (int h = 0; h < n_q_heads; h++) {
        float* qh = q + h * head_dim;
        float q0 = qh[i], q1 = qh[i + 1];
        qh[i]     = q0 * cos_val - q1 * sin_val;
        qh[i + 1] = q0 * sin_val + q1 * cos_val;
    }
    
    /* Apply to all KV heads */
    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k + h * head_dim;
        float k0 = kh[i], k1 = kh[i + 1];
        kh[i]     = k0 * cos_val - k1 * sin_val;
        kh[i + 1] = k0 * sin_val + k1 * cos_val;
    }
}

/* ---- Fused SiLU + Elementwise multiply kernel ----
 * gate[i] = silu(gate[i]) * up[i]
 * One kernel replaces two separate passes. */
__global__ void cuda_silu_elemwise_kernel(float* __restrict__ gate,
                                           const float* __restrict__ up,
                                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        gate[idx] = (g / (1.0f + expf(-g))) * up[idx];
    }
}

/* ---- Residual add kernel ----
 * x[i] += residual[i] */
__global__ void cuda_residual_add_kernel(float* __restrict__ x,
                                          const float* __restrict__ residual,
                                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += residual[idx];
    }
}

/* ---- Softmax kernel (single row) ----
 * For decode (T=1), processes one row of seq_len elements.
 * Uses parallel reduction for max and sum. */
__global__ void cuda_softmax_kernel(float* __restrict__ data, int seq_len,
                                     float scale, int causal_pos) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    /* Scale + causal mask */
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float v = data[i] * scale;
        if (i > causal_pos) v = -INFINITY;
        data[i] = v;
    }
    __syncthreads();
    
    /* Find max via reduction */
    float local_max = -INFINITY;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float v = data[i];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid])
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
    float max_val = sdata[0];
    
    /* Exp and sum */
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float e = expf(data[i] - max_val);
        data[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float sum = sdata[0];
    if (sum < 1e-10f) sum = 1e-10f;
    
    /* Normalize */
    for (int i = tid; i < seq_len; i += blockDim.x) {
        data[i] /= sum;
    }
}

/* ---- MatVec for attention: single row dot product ----
 * score = dot(q_head, k_row)  for one head 
 * This is a simple dot product kernel for Q×Kᵀ and attn×V. */
__global__ void cuda_dot_product_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ out,
                                         int K) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    
    float sum = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        sum += a[i] * b[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) *out = sdata[0];
}

/* ---- Weighted sum for attention context ----
 * ctx[d] = sum_over_s(attn[s] * V[s][d])  for d in head_dim */
__global__ void cuda_attn_weighted_sum_kernel(const float* __restrict__ attn_weights,
                                               const float* __restrict__ V,
                                               float* __restrict__ ctx,
                                               int seq_len, int head_dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < head_dim) {
        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            sum += attn_weights[s] * V[s * head_dim + d];
        }
        ctx[d] = sum;
    }
}

/* ---- Vector copy kernel ---- */
__global__ void cuda_memcpy_kernel(float* __restrict__ dst,
                                    const float* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}


/* ============================================================================
 * C wrapper: cuda_transformer_layer_gpu
 *
 * Runs an entire transformer layer on GPU for decode (T=1).
 * Activations stay in VRAM throughout — only one H→D upload at entry
 * and one D→H download at exit.
 * ========================================================================= */

#include <stdlib.h>

/* Block structure for Q4_K (must match matvec_q4k_cuda.cu) */
#pragma pack(push, 1)
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_k_cuda;
#pragma pack(pop)

/* Import the matvec kernel from matvec_q4k_cuda.cu */
extern __global__ void matvec_q4k_kernel(const block_q4_k_cuda* W, const float* x,
                                          float* y, int num_rows, int num_blocks);

/* Persistent GPU scratch buffers for layer activations */
static float* g_d_hidden = NULL;      /* H floats */
static float* g_d_residual = NULL;    /* H floats */
static float* g_d_normed = NULL;      /* H floats */
static float* g_d_Q = NULL;           /* H floats (n_heads * head_dim) */
static float* g_d_K = NULL;           /* kv_dim floats */
static float* g_d_V = NULL;           /* kv_dim floats */
static float* g_d_attn_out = NULL;    /* H floats */
static float* g_d_gate = NULL;        /* ff_dim floats */
static float* g_d_up = NULL;          /* ff_dim floats */
static float* g_d_down = NULL;        /* H floats */
static float* g_d_normed2 = NULL;     /* H floats */
static float* g_d_attn_proj = NULL;   /* H floats */
static float* g_d_q_head = NULL;      /* head_dim floats */
static float* g_d_scores = NULL;      /* max_seq_len floats */
static float* g_d_ctx = NULL;         /* head_dim floats */
static int g_scratch_H = 0;
static int g_scratch_ff = 0;
static int g_scratch_seq = 0;
static cudaStream_t g_layer_stream = 0;

static void ensure_layer_scratch(int H, int kv_dim, int ff_dim, int max_seq) {
    if (g_scratch_H >= H && g_scratch_ff >= ff_dim && g_scratch_seq >= max_seq) return;
    
    /* Free old buffers */
    if (g_d_hidden) cudaFree(g_d_hidden);
    if (g_d_residual) cudaFree(g_d_residual);
    if (g_d_normed) cudaFree(g_d_normed);
    if (g_d_Q) cudaFree(g_d_Q);
    if (g_d_K) cudaFree(g_d_K);
    if (g_d_V) cudaFree(g_d_V);
    if (g_d_attn_out) cudaFree(g_d_attn_out);
    if (g_d_gate) cudaFree(g_d_gate);
    if (g_d_up) cudaFree(g_d_up);
    if (g_d_down) cudaFree(g_d_down);
    if (g_d_normed2) cudaFree(g_d_normed2);
    if (g_d_attn_proj) cudaFree(g_d_attn_proj);
    if (g_d_q_head) cudaFree(g_d_q_head);
    if (g_d_scores) cudaFree(g_d_scores);
    if (g_d_ctx) cudaFree(g_d_ctx);
    
    cudaMalloc(&g_d_hidden, H * sizeof(float));
    cudaMalloc(&g_d_residual, H * sizeof(float));
    cudaMalloc(&g_d_normed, H * sizeof(float));
    cudaMalloc(&g_d_Q, H * sizeof(float));
    cudaMalloc(&g_d_K, kv_dim * sizeof(float));
    cudaMalloc(&g_d_V, kv_dim * sizeof(float));
    cudaMalloc(&g_d_attn_out, H * sizeof(float));
    cudaMalloc(&g_d_gate, ff_dim * sizeof(float));
    cudaMalloc(&g_d_up, ff_dim * sizeof(float));
    cudaMalloc(&g_d_down, H * sizeof(float));
    cudaMalloc(&g_d_normed2, H * sizeof(float));
    cudaMalloc(&g_d_attn_proj, H * sizeof(float));
    cudaMalloc(&g_d_q_head, 128 * sizeof(float)); /* max head_dim */
    cudaMalloc(&g_d_scores, max_seq * sizeof(float));
    cudaMalloc(&g_d_ctx, 128 * sizeof(float));
    
    if (!g_layer_stream) cudaStreamCreate(&g_layer_stream);
    
    g_scratch_H = H;
    g_scratch_ff = ff_dim;
    g_scratch_seq = max_seq;
}

/* Q4_K matvec on device-resident data: W is already in VRAM, x and y in VRAM */
static void gpu_matvec_q4k(const void* d_W, const float* d_x, float* d_y,
                             int rows, int cols) {
    int num_blocks = cols / 256;
    int threadsPerBlock = 256;
    int warpsPerBlock = threadsPerBlock / 32;
    int blocksPerGrid = (rows + warpsPerBlock - 1) / warpsPerBlock;
    matvec_q4k_kernel<<<blocksPerGrid, threadsPerBlock, 0, g_layer_stream>>>(
        (const block_q4_k_cuda*)d_W, d_x, d_y, rows, num_blocks);
}

extern "C" int cuda_transformer_layer_gpu(
    /* Weight pointers (all in VRAM, Q4_K format) */
    const void* d_wq, int wq_rows, int wq_cols,
    const void* d_wk, int wk_rows, int wk_cols,
    const void* d_wv, int wv_rows, int wv_cols,
    const void* d_wo, int wo_rows, int wo_cols,
    const void* d_w_gate, int wgate_rows, int wgate_cols,
    const void* d_w_up, int wup_rows, int wup_cols,
    const void* d_w_down, int wdown_rows, int wdown_cols,
    /* Norm weights (small, uploaded each call — F32 on host) */
    const float* h_rms_att, const float* h_rms_ffn,
    /* Hidden state (H floats on host, modified in-place) */
    float* h_hidden,
    /* KV cache pointers (pinned host memory) */
    float* h_k_cache,  /* (n_kv_heads, max_seq, head_dim) */
    float* h_v_cache,  /* (n_kv_heads, max_seq, head_dim) */
    int k_cache_stride0, /* stride for head dimension */
    /* Config */
    int H, int kv_dim, int ff_dim, int head_dim,
    int n_heads, int n_kv_heads, int pos, int max_seq)
{
    int seq_len = pos + 1;
    int heads_per_kv = n_heads / n_kv_heads;
    
    ensure_layer_scratch(H, kv_dim, ff_dim, max_seq);
    
    /* Upload norm weights (small: H floats = 16KB each) */
    float *d_rms_att, *d_rms_ffn;
    cudaMalloc(&d_rms_att, H * sizeof(float));
    cudaMalloc(&d_rms_ffn, H * sizeof(float));
    cudaMemcpyAsync(d_rms_att, h_rms_att, H * sizeof(float), cudaMemcpyHostToDevice, g_layer_stream);
    cudaMemcpyAsync(d_rms_ffn, h_rms_ffn, H * sizeof(float), cudaMemcpyHostToDevice, g_layer_stream);
    
    /* ===== Upload hidden state (H floats = 16KB) ===== */
    cudaMemcpyAsync(g_d_hidden, h_hidden, H * sizeof(float), cudaMemcpyHostToDevice, g_layer_stream);
    
    /* Step 1: Save residual */
    cuda_memcpy_kernel<<<(H+255)/256, 256, 0, g_layer_stream>>>(g_d_residual, g_d_hidden, H);
    
    /* Step 2: RMSNorm (pre-attention) */
    int norm_threads = 256;
    cuda_rmsnorm_kernel<<<1, norm_threads, norm_threads * sizeof(float), g_layer_stream>>>(
        g_d_hidden, d_rms_att, g_d_normed, H, 1e-6f);
    
    /* Step 3: QKV projections (all in VRAM!) */
    gpu_matvec_q4k(d_wq, g_d_normed, g_d_Q, wq_rows, wq_cols);
    gpu_matvec_q4k(d_wk, g_d_normed, g_d_K, wk_rows, wk_cols);
    gpu_matvec_q4k(d_wv, g_d_normed, g_d_V, wv_rows, wv_cols);
    
    /* Step 4: RoPE on Q and K (in VRAM) */
    cuda_rope_kernel<<<1, head_dim/2, 0, g_layer_stream>>>(
        g_d_Q, g_d_K, n_heads, n_kv_heads, head_dim, pos);
    
    /* Step 5: Download Q, K, V for CPU attention (proven correct path).
     * Attention is ~1% of decode time; the matvec savings from GPU are the win. */
    float* h_Q = (float*)malloc(H * sizeof(float));
    float* h_K = (float*)malloc(kv_dim * sizeof(float));
    float* h_V = (float*)malloc(kv_dim * sizeof(float));
    cudaMemcpyAsync(h_Q, g_d_Q, H * sizeof(float), cudaMemcpyDeviceToHost, g_layer_stream);
    cudaMemcpyAsync(h_K, g_d_K, kv_dim * sizeof(float), cudaMemcpyDeviceToHost, g_layer_stream);
    cudaMemcpyAsync(h_V, g_d_V, kv_dim * sizeof(float), cudaMemcpyDeviceToHost, g_layer_stream);
    cudaStreamSynchronize(g_layer_stream);
    
    /* Append K,V to host KV cache */
    for (int h = 0; h < n_kv_heads; h++) {
        memcpy(h_k_cache + h * k_cache_stride0 + pos * head_dim, h_K + h * head_dim, head_dim * sizeof(float));
        memcpy(h_v_cache + h * k_cache_stride0 + pos * head_dim, h_V + h * head_dim, head_dim * sizeof(float));
    }
    
    /* CPU attention: exact same logic as engine.c's proven path */
    float* h_attn_out = (float*)calloc(H, sizeof(float));
    float scale = 1.0f / sqrtf((float)head_dim);
    float* scores = (float*)malloc(seq_len * sizeof(float));
    
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        float* q_head = h_Q + h * head_dim;
        float* k_cache_head = h_k_cache + kv_h * k_cache_stride0;
        float* v_cache_head = h_v_cache + kv_h * k_cache_stride0;
        
        /* Q × Kᵀ (dot product for each past position) */
        for (int s = 0; s < seq_len; s++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_head[d] * k_cache_head[s * head_dim + d];
            }
            scores[s] = dot * scale;
        }
        
        /* Softmax */
        float max_score = scores[0];
        for (int s = 1; s < seq_len; s++) {
            if (scores[s] > max_score) max_score = scores[s];
        }
        float sum_exp = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum_exp += scores[s];
        }
        for (int s = 0; s < seq_len; s++) scores[s] /= sum_exp;
        
        /* Weighted sum: attn_weights × V */
        for (int d = 0; d < head_dim; d++) {
            float val = 0.0f;
            for (int s = 0; s < seq_len; s++) {
                val += scores[s] * v_cache_head[s * head_dim + d];
            }
            h_attn_out[h * head_dim + d] = val;
        }
    }
    
    free(scores);
    free(h_Q);
    free(h_K);
    free(h_V);
    
    /* Upload attention output back to GPU for output projection */
    cudaMemcpyAsync(g_d_attn_out, h_attn_out, H * sizeof(float), cudaMemcpyHostToDevice, g_layer_stream);
    free(h_attn_out);
    
    /* Step 11: Output projection (in VRAM) */
    gpu_matvec_q4k(d_wo, g_d_attn_out, g_d_attn_proj, wo_rows, wo_cols);
    
    /* Step 12: Residual add */
    cuda_residual_add_kernel<<<(H+255)/256, 256, 0, g_layer_stream>>>(
        g_d_attn_proj, g_d_residual, H);
    
    /* Step 13: Save residual */
    cuda_memcpy_kernel<<<(H+255)/256, 256, 0, g_layer_stream>>>(g_d_residual, g_d_attn_proj, H);
    
    /* Step 14: RMSNorm (pre-MLP) */
    cuda_rmsnorm_kernel<<<1, norm_threads, norm_threads * sizeof(float), g_layer_stream>>>(
        g_d_attn_proj, d_rms_ffn, g_d_normed2, H, 1e-6f);
    
    /* Step 15-16: Gate and Up projections */
    gpu_matvec_q4k(d_w_gate, g_d_normed2, g_d_gate, wgate_rows, wgate_cols);
    gpu_matvec_q4k(d_w_up, g_d_normed2, g_d_up, wup_rows, wup_cols);
    
    /* Step 17-18: Fused SiLU + elemwise multiply */
    cuda_silu_elemwise_kernel<<<(ff_dim+255)/256, 256, 0, g_layer_stream>>>(
        g_d_gate, g_d_up, ff_dim);
    
    /* Step 19: Down projection */
    gpu_matvec_q4k(d_w_down, g_d_gate, g_d_down, wdown_rows, wdown_cols);
    
    /* Step 20: Residual add */
    cuda_residual_add_kernel<<<(H+255)/256, 256, 0, g_layer_stream>>>(
        g_d_down, g_d_residual, H);
    
    /* ===== Download result back to host ===== */
    cudaMemcpyAsync(h_hidden, g_d_down, H * sizeof(float), cudaMemcpyDeviceToHost, g_layer_stream);
    cudaStreamSynchronize(g_layer_stream);
    
    cudaFree(d_rms_att);
    cudaFree(d_rms_ffn);
    
    return 0;
}
