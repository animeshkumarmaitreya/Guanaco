#include "types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#pragma pack(push, 1)
typedef struct {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[128];
} block_q4_k_cuda;
#pragma pack(pop)

__device__ inline float f16_to_f32_cuda(uint16_t h) {
    return __half2float(*reinterpret_cast<const half*>(&h));
}

__device__ inline void get_scale_min_k4_cuda(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

__global__ void matvec_q4k_kernel(const block_q4_k_cuda *blocks, const float *x, float *y, int num_rows, int num_blocks) {
    int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    if (row >= num_rows) return;

    int lane_id = threadIdx.x % 32;
    float sum = 0.0f;
    const block_q4_k_cuda *row_blocks = blocks + row * num_blocks;

    for (int b = 0; b < num_blocks; b++) {
        const block_q4_k_cuda *blk = &row_blocks[b];
        const float d_all = f16_to_f32_cuda(blk->d);
        const float min_all = f16_to_f32_cuda(blk->dmin);

        const uint8_t *q_ptr = blk->qs;
        const float *px = x + b * 256;

        int is = 0;
        uint8_t sc, m;

        for (int j = 0; j < 256; j += 64) {
            get_scale_min_k4_cuda(is + 0, blk->scales, &sc, &m);
            float d1 = d_all * sc;
            float m1 = min_all * m;

            get_scale_min_k4_cuda(is + 1, blk->scales, &sc, &m);
            float d2 = d_all * sc;
            float m2 = min_all * m;

            uint8_t qval = q_ptr[lane_id];
            
            float w1 = d1 * (qval & 0xF) - m1;
            float w2 = d2 * (qval >> 4) - m2;

            sum += w1 * px[lane_id] + w2 * px[32 + lane_id];
            
            q_ptr += 32;
            px += 64;
            is += 2;
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        y[row] = sum;
    }
}

static float* g_d_x = NULL;
static float* g_d_y = NULL;
static int g_alloc_cols = 0;
static int g_alloc_rows = 0;
static cudaStream_t g_stream = 0;

extern "C" int cuda_matvec_q4k_f32_impl(const void *W_data, const float *x, float *y, int num_rows, int num_cols) {
    if (!g_stream) {
        cudaStreamCreate(&g_stream);
    }
    if (!g_d_x || g_alloc_cols < num_cols) {
        if (g_d_x) cudaFree(g_d_x);
        if (cudaMalloc(&g_d_x, num_cols * sizeof(float)) != cudaSuccess) return -1;
        g_alloc_cols = num_cols;
    }
    if (!g_d_y || g_alloc_rows < num_rows) {
        if (g_d_y) cudaFree(g_d_y);
        if (cudaMalloc(&g_d_y, num_rows * sizeof(float)) != cudaSuccess) return -1;
        g_alloc_rows = num_rows;
    }
    
    int num_blocks = num_cols / 256;
    cudaMemcpyAsync(g_d_x, x, num_cols * sizeof(float), cudaMemcpyHostToDevice, g_stream);
    
    int threadsPerBlock = 256; 
    int warpsPerBlock = threadsPerBlock / 32;
    int blocksPerGrid = (num_rows + warpsPerBlock - 1) / warpsPerBlock;
    
    int shared_mem_size = 256 * sizeof(float);
    matvec_q4k_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size, g_stream>>>((const block_q4_k_cuda *)W_data, g_d_x, g_d_y, num_rows, num_blocks);
    
    cudaMemcpyAsync(y, g_d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
    return 0;
}

extern "C" void* cuda_upload_weight(const void* host_ptr, size_t size) {
    void* d_ptr = NULL;
    if (cudaMalloc(&d_ptr, size) != cudaSuccess) return NULL;
    
    /* Stage through a heap buffer: CUDA DMA cannot safely read from
     * mmap'd MAP_PRIVATE file-backed pages (lazy page faults cause SIGSEGV).
     * The staging buffer is freed immediately after the upload. */
    void* staging = malloc(size);
    if (!staging) { cudaFree(d_ptr); return NULL; }
    memcpy(staging, host_ptr, size);
    
    cudaError_t err = cudaMemcpy(d_ptr, staging, size, cudaMemcpyHostToDevice);
    free(staging);
    if (err != cudaSuccess) {
        cudaFree(d_ptr);
        return NULL;
    }
    return d_ptr;
}
