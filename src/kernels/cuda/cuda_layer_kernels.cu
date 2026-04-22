/*============================================================================
 * cuda_layer_kernels.cu — GPU kernels for elementwise transformer ops
 *
 * These kernels keep activation data resident in VRAM, eliminating
 * Host↔Device transfers between consecutive operations within a layer.
 *============================================================================*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* Block structure for Q6_K (must match quant_matvec_q6_k.c) */
#pragma pack(push, 1)
typedef struct {
    uint8_t ql[128];      // lower 4 bits
    uint8_t qh[64];       // upper 2 bits
    int8_t  scales[16];   // block scales
    uint16_t d;           // super-block scale
} block_q6_k_cuda;
#pragma pack(pop)

/* ---- Q6_K matvec kernel ---- */
__global__ void matvec_q6k_kernel(const block_q6_k_cuda* W, const float* x,
                                   float* y, int num_rows, int num_blocks) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= num_rows) return;

    int tid = threadIdx.x % 32;
    const block_q6_k_cuda* row_W = W + row * num_blocks;
    float sum = 0.0f;

    for (int b = 0; b < num_blocks; b++) {
        const block_q6_k_cuda* blk = &row_W[b];
        const float d_all = __half2float(*((const half*)&blk->d));
        const float* blk_x = x + b * 256;

        /* Q6_K has 256 elements in 2 super-blocks of 128 */
        /* Each thread handles 8 elements (256 / 32 = 8) */
        
        for (int i = 0; i < 2; i++) { // 2 super-blocks
            const uint8_t* ql = blk->ql + i * 64;
            const uint8_t* qh = blk->qh + i * 32;
            const int8_t* sc = blk->scales + i * 8;
            const float* px = blk_x + i * 128;

            int l = tid; 
            /* Indexing logic for 6-bit reconstruction */
            int is = l / 16;
            
            /* Formula: d_all * sc * (quant - 32) */
            /* l is 0..31. We need to handle 4 values per thread to cover 128 elements? 
             * No, 32 threads * 4 values = 128. 
             * In my CPU ref: l loop goes 0..31 and handles q1, q2, q3, q4 (4 values). */
            
            const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            sum += d_all * sc[is + 0] * q1 * px[l +  0];
            sum += d_all * sc[is + 2] * q2 * px[l + 32];
            sum += d_all * sc[is + 4] * q3 * px[l + 64];
            sum += d_all * sc[is + 6] * q4 * px[l + 96];
        }
    }

    /* Wrap up with warp reduction */
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (tid == 0) y[row] = sum;
}

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
    
    float theta = 500000.0f; /* Llama 3.1 theta */
    float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
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
static float* g_d_hidden = NULL;
static float* g_d_residual = NULL;
static float* g_d_normed = NULL;
static float* g_d_Q = NULL;
static float* g_d_K = NULL;
static float* g_d_V = NULL;
static float* g_d_attn_scores = NULL; // New for Phase 3
static float* g_d_attn_out = NULL;
static float* g_d_gate = NULL;
static float* g_d_up = NULL;
static float* g_d_down = NULL;
static float* g_d_normed2 = NULL;
static float* g_d_attn_proj = NULL;
static float* g_h_Q_scratch = NULL;   /* H floats (host) */
static float* g_h_K_scratch = NULL;   /* kv_dim floats (host) */
static float* g_h_V_scratch = NULL;   /* kv_dim floats (host) */
static float* g_h_scores_scratch = NULL; /* max_seq floats (host) */
static float* g_h_attn_out_scratch = NULL; /* H floats (host) */
static int g_scratch_H = 0;
static int g_scratch_ff = 0;
static int g_scratch_seq = 0;
static cudaStream_t g_layer_stream = 0;

#define MAX_GPU_LAYER_KV_SLOTS 64
typedef struct {
    float* host_k_base;
    float* host_v_base;
    float* d_k;
    float* d_v;
    int n_kv_heads;
    int head_dim;
    int max_seq;
    int stride0;
    int last_pos;
    int initialized;
} gpu_layer_kv_slot;

static gpu_layer_kv_slot g_layer_kv_slots[MAX_GPU_LAYER_KV_SLOTS];

static gpu_layer_kv_slot* get_or_create_layer_kv_slot(float* h_k_cache,
                                                       float* h_v_cache,
                                                       int n_kv_heads,
                                                       int head_dim,
                                                       int max_seq,
                                                       int stride0,
                                                       int pos) {
    int free_idx = -1;
    for (int i = 0; i < MAX_GPU_LAYER_KV_SLOTS; i++) {
        if (g_layer_kv_slots[i].initialized &&
            g_layer_kv_slots[i].host_k_base == h_k_cache &&
            g_layer_kv_slots[i].host_v_base == h_v_cache) {
            
            if (pos > 0 && g_layer_kv_slots[i].last_pos != -1 && pos != g_layer_kv_slots[i].last_pos + 1) {
                /* Re-sync if cache was modified externally */
                for (int h = 0; h < n_kv_heads; ++h) {
                    size_t off = (size_t)h * (size_t)stride0;
                    size_t bytes = (size_t)pos * (size_t)head_dim * sizeof(float);
                    cudaMemcpyAsync(g_layer_kv_slots[i].d_k + off, h_k_cache + off, bytes, cudaMemcpyHostToDevice, g_layer_stream);
                    cudaMemcpyAsync(g_layer_kv_slots[i].d_v + off, h_v_cache + off, bytes, cudaMemcpyHostToDevice, g_layer_stream);
                }
            }
            g_layer_kv_slots[i].last_pos = pos;
            return &g_layer_kv_slots[i];

        }
        if (!g_layer_kv_slots[i].initialized && free_idx < 0) {
            free_idx = i;
        }
    }

    if (free_idx < 0) return NULL;

    gpu_layer_kv_slot* slot = &g_layer_kv_slots[free_idx];
    memset(slot, 0, sizeof(*slot));
    slot->host_k_base = h_k_cache;
    slot->host_v_base = h_v_cache;
    slot->n_kv_heads = n_kv_heads;
    slot->head_dim = head_dim;
    slot->max_seq = max_seq;
    slot->stride0 = stride0;
    slot->last_pos = pos;

    size_t elems = (size_t)n_kv_heads * (size_t)max_seq * (size_t)head_dim;
    cudaMalloc(&slot->d_k, elems * sizeof(float));
    cudaMalloc(&slot->d_v, elems * sizeof(float));

    if (!slot->d_k || !slot->d_v) {
        if (slot->d_k) cudaFree(slot->d_k);
        if (slot->d_v) cudaFree(slot->d_v);
        memset(slot, 0, sizeof(*slot));
        return NULL;
    }

    if (pos > 0) {
        for (int h = 0; h < n_kv_heads; h++) {
            size_t src_off = (size_t)h * (size_t)stride0;
            size_t bytes = (size_t)pos * (size_t)head_dim * sizeof(float);
            cudaMemcpyAsync(slot->d_k + src_off, h_k_cache + src_off, bytes,
                            cudaMemcpyHostToDevice, g_layer_stream);
            cudaMemcpyAsync(slot->d_v + src_off, h_v_cache + src_off, bytes,
                            cudaMemcpyHostToDevice, g_layer_stream);
        }
    }

    slot->initialized = 1;
    return slot;
}

/* Persistent VRAM KV Cache pointers per layer */
static float* g_d_k_cache[64] = {NULL};
static float* g_d_v_cache[64] = {NULL};
static size_t g_k_cache_size = 0;

static int ensure_vram_kv_cache(int layer, int n_kv_heads, int head_dim, int max_seq) {
    /* Cap VRAM cache to 2048 tokens to stay within 4GB limit for 24 layers. 
     * Llama 3.1 131k context is impossible on RTX 3050. */
    int effective_max_seq = (max_seq > 2048) ? 2048 : max_seq;
    size_t needed = (size_t)n_kv_heads * (size_t)effective_max_seq * (size_t)head_dim * sizeof(float);
    
    if (g_d_k_cache[layer] && g_k_cache_size >= needed) return effective_max_seq;
    
    if (g_d_k_cache[layer]) cudaFree(g_d_k_cache[layer]);
    if (g_d_v_cache[layer]) cudaFree(g_d_v_cache[layer]);
    
    if (cudaMalloc(&g_d_k_cache[layer], needed) != cudaSuccess) g_d_k_cache[layer] = NULL;
    if (cudaMalloc(&g_d_v_cache[layer], needed) != cudaSuccess) g_d_v_cache[layer] = NULL;
    g_k_cache_size = needed;
    return effective_max_seq;
}

/* ---- GPU Attention Kernels (Optimized Phase 3) ---- */

__global__ void cuda_update_kv_cache_kernel(float* k_cache, float* v_cache,
                                          const float* k, const float* v,
                                          int n_kv_heads, int head_dim, int pos, int max_seq) {
    int h = blockIdx.x; // kv head index
    int i = threadIdx.x; // element index in head_dim
    if (h >= n_kv_heads || i >= head_dim) return;

    int cache_idx = (h * max_seq + pos) * head_dim + i;
    int src_idx = h * head_dim + i;

    k_cache[cache_idx] = k[src_idx];
    v_cache[cache_idx] = v[src_idx];
}

__global__ void cuda_attn_scores_kernel_gqa(const float* q, const float* k_cache, float* scores,
                                            int n_heads, int n_kv_heads, int head_dim,
                                            int pos, int max_seq, float scale) {
    int h = blockIdx.x; // q head index
    int s = blockIdx.y * blockDim.x + threadIdx.x; // sequence index
    if (h >= n_heads || s > pos) return;

    int kv_h = h / (n_heads / n_kv_heads);
    const float* head_q = q + h * head_dim;
    const float* head_k = k_cache + (kv_h * max_seq + s) * head_dim;

    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += head_q[d] * head_k[d];
    }
    
    scores[h * max_seq + s] = dot * scale;
}

__global__ void cuda_softmax_kernel(float* scores, int n_heads, int pos, int max_seq) {
    int h = blockIdx.x;
    if (h >= n_heads) return;

    float* head_scores = scores + h * max_seq;
    int seq_len = pos + 1;

    /* Parallel reduction for max */
    float max_val = -1e20f;
    for (int s = 0; s < seq_len; s++) {
        if (head_scores[s] > max_val) max_val = head_scores[s];
    }

    /* Parallel reduction for sum */
    float sum = 0.0f;
    for (int s = 0; s < seq_len; s++) {
        float e = expf(head_scores[s] - max_val);
        head_scores[s] = e;
        sum += e;
    }

    /* Normalize */
    float inv_sum = 1.0f / (sum + 1e-10f);
    for (int s = 0; s < seq_len; s++) {
        head_scores[s] *= inv_sum;
    }
}

__global__ void cuda_attn_context_kernel_gqa(const float* scores, const float* v_cache, float* output,
                                             int n_heads, int n_kv_heads, int head_dim,
                                             int pos, int max_seq) {
    int h = blockIdx.x; // q head index
    int d = threadIdx.x; // element index in head_dim
    if (h >= n_heads || d >= head_dim) return;

    int kv_h = h / (n_heads / n_kv_heads);
    const float* head_scores = scores + h * max_seq;
    const float* kv_v_base = v_cache + kv_h * max_seq * head_dim;

    float val = 0.0f;
    int seq_len = pos + 1;
    for (int s = 0; s < seq_len; s++) {
        val += head_scores[s] * kv_v_base[s * head_dim + d];
    }

    output[h * head_dim + d] = val;
}

static void ensure_layer_scratch(int H, int n_heads, int kv_dim, int ff_dim, int max_seq) {
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
    if (g_h_Q_scratch) free(g_h_Q_scratch);
    if (g_h_K_scratch) free(g_h_K_scratch);
    if (g_h_V_scratch) free(g_h_V_scratch);
    if (g_h_scores_scratch) free(g_h_scores_scratch);
    if (g_h_attn_out_scratch) free(g_h_attn_out_scratch);
    
    cudaMalloc(&g_d_hidden, (size_t)H * sizeof(float));
    cudaMalloc(&g_d_residual, (size_t)H * sizeof(float));
    cudaMalloc(&g_d_normed, (size_t)H * sizeof(float));
    cudaMalloc(&g_d_Q, (size_t)H * sizeof(float));
    cudaMalloc(&g_d_K, (size_t)kv_dim * sizeof(float));
    cudaMalloc(&g_d_V, (size_t)kv_dim * sizeof(float));
    cudaMalloc(&g_d_attn_scores, (size_t)n_heads * max_seq * sizeof(float));
    cudaMalloc(&g_d_attn_out, (size_t)H * sizeof(float));
    cudaMalloc(&g_d_gate, (size_t)ff_dim * sizeof(float));
    cudaMalloc(&g_d_up, (size_t)ff_dim * sizeof(float));
    cudaMalloc(&g_d_down, (size_t)H * sizeof(float));
    cudaMalloc(&g_d_normed2, (size_t)H * sizeof(float));
    cudaMalloc(&g_d_attn_proj, (size_t)H * sizeof(float));

    g_h_Q_scratch = (float*)malloc(H * sizeof(float));
    g_h_K_scratch = (float*)malloc(kv_dim * sizeof(float));
    g_h_V_scratch = (float*)malloc(kv_dim * sizeof(float));
    g_h_scores_scratch = (float*)malloc(max_seq * sizeof(float));
    g_h_attn_out_scratch = (float*)malloc(H * sizeof(float));
    
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
    int shared_mem_size = 256 * sizeof(float);
    matvec_q4k_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size, g_layer_stream>>>(
        (const block_q4_k_cuda*)d_W, d_x, d_y, rows, num_blocks);
}

static void gpu_matvec_q6k(const void* W, const float* x, float* y, int num_rows, int num_cols) {
    int num_blocks = num_cols / 256;
    int threads_per_block = 256;
    int blocks_per_grid = (num_rows + (threads_per_block / 32) - 1) / (threads_per_block / 32);
    matvec_q6k_kernel<<<blocks_per_grid, threads_per_block, 0, g_layer_stream>>>(
        (const block_q6_k_cuda*)W, x, y, num_rows, num_blocks);
}

static void gpu_matvec_any(const void* W, const float* x, float* y, int rows, int cols, int dtype) {
    if (dtype == 4) { // DTYPE_Q4_K
        gpu_matvec_q4k(W, x, y, rows, cols);
    } else if (dtype == 5) { // DTYPE_Q6_K
        gpu_matvec_q6k(W, x, y, rows, cols);
    } else {
        /* Fallback for others - should not happen in fused layer for now */
        printf("[GPU ERROR] Unsupported dtype %d in fused layer\n", dtype);
    }
}

extern "C" void cuda_sync_kv_cache(int layer, int n_kv_heads, int head_dim, int pos, int T, 
                                 const float* h_k_cache, const float* h_v_cache, int stride0, int max_seq) {
    int eff_seq = ensure_vram_kv_cache(layer, n_kv_heads, head_dim, max_seq);
    if (!g_d_k_cache[layer]) return;

    if (pos >= eff_seq) return; 
    int sync_T = (pos + T > eff_seq) ? (eff_seq - pos) : T;
    if (sync_T <= 0) return;
    
    /* Upload tokens from pos to pos+sync_T-1 */
    for (int h = 0; h < n_kv_heads; h++) {
        size_t h_off = (size_t)h * stride0 + (size_t)pos * head_dim;
        size_t d_off = (size_t)h * eff_seq * head_dim + (size_t)pos * head_dim;
        size_t bytes = (size_t)sync_T * head_dim * sizeof(float);
        
        cudaMemcpyAsync(g_d_k_cache[layer] + d_off, h_k_cache + h_off, bytes, cudaMemcpyHostToDevice, g_layer_stream);
        cudaMemcpyAsync(g_d_v_cache[layer] + d_off, h_v_cache + h_off, bytes, cudaMemcpyHostToDevice, g_layer_stream);
    }
}

extern "C" int cuda_transformer_layer_gpu(
    const void* d_wq, int wq_rows, int wq_cols, int wq_dtype,
    const void* d_wk, int wk_rows, int wk_cols, int wk_dtype,
    const void* d_wv, int wv_rows, int wv_cols, int wv_dtype,
    const void* d_wo, int wo_rows, int wo_cols, int wo_dtype,
    const void* d_w_gate, int wgate_rows, int wgate_cols, int wgate_dtype,
    const void* d_w_up, int wup_rows, int wup_cols, int wup_dtype,
    const void* d_w_down, int wdown_rows, int wdown_cols, int wdown_dtype,
    const float* h_rms_att, const float* h_rms_ffn,
    float* h_hidden,
    float* h_k_cache, float* h_v_cache, int k_cache_stride0,
    int H, int n_heads, int n_kv_heads, int head_dim,
    int pos, int layer, int max_seq_len) {

    int seq_len = pos + 1;
    int heads_per_kv = n_heads / n_kv_heads;
    int ff_dim = wgate_rows; // MLP intermediate dimension
    int kv_dim = n_kv_heads * head_dim;
    
    ensure_layer_scratch(H, n_heads, kv_dim, ff_dim, max_seq_len);
    
    /* ===== Step 1: Upload input hidden state and norm weights ===== */
    cudaMemcpyAsync(g_d_hidden, h_hidden, H * sizeof(float), cudaMemcpyHostToDevice, g_layer_stream);
    
    /* Reuse scratch buffers for norm weights (safe sequence) */
    float* d_rms_att_ptr = g_d_attn_proj; 
    float* d_rms_ffn_ptr = g_d_down;      
    cudaMemcpyAsync(d_rms_att_ptr, h_rms_att, H * sizeof(float), cudaMemcpyHostToDevice, g_layer_stream);
    cudaMemcpyAsync(d_rms_ffn_ptr, h_rms_ffn, H * sizeof(float), cudaMemcpyHostToDevice, g_layer_stream);
    
    /* Step 2: Save residual and compute RMSNorm (pre-attention) */
    cuda_memcpy_kernel<<<(H+255)/256, 256, 0, g_layer_stream>>>(g_d_residual, g_d_hidden, H);
    
    int norm_threads = 256;
    cuda_rmsnorm_kernel<<<1, norm_threads, norm_threads * sizeof(float), g_layer_stream>>>(
        g_d_hidden, d_rms_att_ptr, g_d_normed, H, 1e-6f);

    /* Step 3: QKV projections (Heterogeneous Dispatch) */
    gpu_matvec_any(d_wq, g_d_normed, g_d_Q, wq_rows, wq_cols, wq_dtype);
    gpu_matvec_any(d_wk, g_d_normed, g_d_K, wk_rows, wk_cols, wk_dtype);
    gpu_matvec_any(d_wv, g_d_normed, g_d_V, wv_rows, wv_cols, wv_dtype);
    
    /* Step 4: RoPE on Q and K (in VRAM) */
    cuda_rope_kernel<<<1, head_dim/2, 0, g_layer_stream>>>(
        g_d_Q, g_d_K, n_heads, n_kv_heads, head_dim, pos);
    
    /* Step 5: GPU Attention (Phase 3 Full GPU Residency) */
    int eff_seq = ensure_vram_kv_cache(layer, n_kv_heads, head_dim, max_seq_len);
    if (!g_d_k_cache[layer]) return -1; // VRAM OOM for cache
    
    /* Update VRAM cache with current token's K/V */
    cuda_update_kv_cache_kernel<<<n_kv_heads, head_dim, 0, g_layer_stream>>>(
        g_d_k_cache[layer], g_d_v_cache[layer], g_d_K, g_d_V,
        n_kv_heads, head_dim, pos, eff_seq);

    /* Compute scores: Q x K^T */
    float scale = 1.0f / sqrtf((float)head_dim);
    // Grid: (n_heads, (pos+1+31)/32), Block: 32
    dim3 grid_scores(n_heads, (pos + 32) / 32);
    cuda_attn_scores_kernel_gqa<<<grid_scores, 32, 0, g_layer_stream>>>(
        g_d_Q, g_d_k_cache[layer], g_d_attn_scores, 
        n_heads, n_kv_heads, head_dim, pos, eff_seq, scale);

    /* Softmax over scores */
    cuda_softmax_kernel<<<n_heads, 1, 0, g_layer_stream>>>(
        g_d_attn_scores, n_heads, pos, eff_seq);

    /* Compute context: scores x V */
    cuda_attn_context_kernel_gqa<<<n_heads, head_dim, 0, g_layer_stream>>>(
        g_d_attn_scores, g_d_v_cache[layer], g_d_attn_out,
        n_heads, n_kv_heads, head_dim, pos, eff_seq);

    /* Step 6: [DEPRECATED] upload no longer needed; g_d_attn_out is already on GPU */

    /* Step 7: Output projection */
    gpu_matvec_any(d_wo, g_d_attn_out, g_d_attn_proj, wo_rows, wo_cols, wo_dtype);

    
    /* Step 8: Add residual from Step 1 */
    cuda_residual_add_kernel<<<(H+255)/256, 256, 0, g_layer_stream>>>(
        g_d_attn_proj, g_d_residual, H);
    
    /* Step 9-10: MLP Norm (reuse g_d_residual for residual save) */
    cuda_memcpy_kernel<<<(H+255)/256, 256, 0, g_layer_stream>>>(g_d_residual, g_d_attn_proj, H);
    cuda_rmsnorm_kernel<<<1, norm_threads, norm_threads * sizeof(float), g_layer_stream>>>(
        g_d_attn_proj, d_rms_ffn_ptr, g_d_normed2, H, 1e-6f);
    
    /* Step 11: Gate and Up projections */
    gpu_matvec_any(d_w_gate, g_d_normed2, g_d_gate, wgate_rows, wgate_cols, wgate_dtype);
    gpu_matvec_any(d_w_up, g_d_normed2, g_d_up, wup_rows, wup_cols, wup_dtype);
    
    /* Step 13-14: Fused Activation (SwiGLU) */
    cuda_silu_elemwise_kernel<<<(ff_dim+255)/256, 256, 0, g_layer_stream>>>(
        g_d_gate, g_d_up, ff_dim);
    
    /* Step 15: MLP Down projection */
    gpu_matvec_any(d_w_down, g_d_gate, g_d_down, wdown_rows, wdown_cols, wdown_dtype);
    
    /* Step 16: Final residual add */
    cuda_residual_add_kernel<<<(H+255)/256, 256, 0, g_layer_stream>>>(
        g_d_down, g_d_residual, H);
    
    /* ===== Finalize: Download result back to host hidden state ===== */
    /* No stream synchronization here! We let the GPU pipeline run ahead. 
     * The host must call cuda_finish_layer() before using h_hidden. */
    cudaMemcpyAsync(h_hidden, g_d_down, H * sizeof(float), cudaMemcpyDeviceToHost, g_layer_stream);
    
    return 0;
}

extern "C" void cuda_finish_layer() {
    if (g_layer_stream) cudaStreamSynchronize(g_layer_stream);
}
