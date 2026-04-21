#include "quant.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// FP16 bit representation for 1.0f
#define FP16_ONE 0x3c00

#pragma pack(push, 1)
typedef struct {
  uint16_t scale;
  int8_t quants[32];
} block_q8_0;

typedef struct {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[128];
} block_q4_k;
#pragma pack(pop)

static void test_q8_0_matvec() {
  Tensor W;
  W.ndim = 2;
  W.shape[0] = 1;
  W.shape[1] = 64; // 2 blocks
  W.dtype = DTYPE_Q8_0;

  block_q8_0 blocks[2];
  memset(blocks, 0, sizeof(blocks));

  // Block 0: scale = 1.0, quants = 1, 2, ..., 32
  blocks[0].scale = FP16_ONE;
  for (int i = 0; i < 32; i++)
    blocks[0].quants[i] = i + 1;

  // Block 1: scale = 2.0 (FP16: 0x4000), quants = -1, -2, ..., -32
  blocks[1].scale = 0x4000;
  for (int i = 0; i < 32; i++)
    blocks[1].quants[i] = -(i + 1);

  W.data = blocks;
  W.byte_size = sizeof(blocks);

  float x[64];
  for (int i = 0; i < 64; i++)
    x[i] = 1.0f; // dot product is sum

  float y[1] = {0};
  int ret = matvec_q8_0_f32(&W, x, y);
  assert(ret == 0);

  float expected = 0.0f;
  for (int i = 0; i < 32; i++)
    expected += (i + 1) * 1.0f * 1.0f;
  for (int i = 0; i < 32; i++)
    expected -= (i + 1) * 2.0f * 1.0f;

  if (fabsf(y[0] - expected) > 1e-4) {
    printf("  FAIL: Q8_0 matvec (got %f, expected %f)\n", y[0], expected);
    exit(1);
  }
  printf("  PASS: q8_0_matvec\n");
}

static void test_q8_0_byte_size_bounds(void) {
  Tensor W;
  W.ndim = 2;
  W.shape[0] = 1;
  W.shape[1] = 64; // 2 blocks
  W.dtype = DTYPE_Q8_0;

  block_q8_0 blocks[2];
  memset(blocks, 0, sizeof(blocks));
  W.data = blocks;

  float x[64];
  memset(x, 0, sizeof(x));
  float y[1] = {0};

  W.byte_size = sizeof(blocks) - 1; // too small
  int ret = matvec_q8_0_f32(&W, x, y);
  if (ret == 0) {
    printf("  FAIL: q8_0_byte_size_bounds (expected failure)\n");
    exit(1);
  }
  printf("  PASS: q8_0_byte_size_bounds\n");
}

static void test_q4_k_matvec() {
  Tensor W;
  W.ndim = 2;
  W.shape[0] = 1;
  W.shape[1] = 256; // 1 block
  W.dtype = DTYPE_Q4_K;

  block_q4_k block;
  memset(&block, 0, sizeof(block));

  block.d = FP16_ONE; // 1.0
  block.dmin = 0;     // 0.0

  memset(block.scales, 1, 12);
  memset(block.qs, 0x12, 128);

  W.data = &block;
  W.byte_size = sizeof(block);

  float x[256];
  for (int i = 0; i < 256; i++)
    x[i] = 1.0f;

  float y[1] = {0};
  int ret = matvec_q4k_f32(&W, x, y);
  assert(ret == 0);

  float expected = 0;
  for (int i = 0; i < 256; i += 64) {
    float sc1 = 1.0f;
    float m1 = (i < 128) ? 1.0f : 0.0f;
    float sc2 = sc1;
    float m2 = m1;

    for (int l = 0; l < 32; l++)
      expected += (1.0f * sc1 * 2 - 0.0f * m1); // lower 4 bits of 0x12 are 2
    for (int l = 0; l < 32; l++)
      expected += (1.0f * sc2 * 1 - 0.0f * m2); // upper 4 bits of 0x12 are 1
  }

  if (fabsf(y[0] - expected) > 1e-4) {
    printf("  FAIL: Q4_K matvec (got %f, expected %f)\n", y[0], expected);
    exit(1);
  }
  printf("  PASS: q4_k_matvec\n");
}

static uint8_t expected_scale_u6(const uint8_t *scales, int g) {
  if (g < 4) {
    return (uint8_t)(scales[g] & 0x3fu);
  }
  uint8_t low4 = (uint8_t)(scales[g + 4] & 0x0fu);
  uint8_t hi2 = (uint8_t)((scales[g - 4] >> 6) & 0x03u);
  return (uint8_t)(low4 | (hi2 << 4));
}

static uint8_t expected_min_u6(const uint8_t *scales, int g) {
  if (g < 4) {
    return (uint8_t)(scales[g + 4] & 0x3fu);
  }
  uint8_t low4 = (uint8_t)((scales[g + 4] >> 4) & 0x0fu);
  uint8_t hi2 = (uint8_t)((scales[g] >> 6) & 0x03u);
  return (uint8_t)(low4 | (hi2 << 4));
}

static void test_q4_k_scales_and_nibbles_mapping(void) {
  Tensor W;
  W.ndim = 2;
  W.shape[0] = 1;
  W.shape[1] = 256;
  W.dtype = DTYPE_Q4_K;

  block_q4_k block;
  memset(&block, 0, sizeof(block));
  block.d = FP16_ONE;    // d_all = 1.0
  block.dmin = FP16_ONE; // min_all = 1.0

  /* Craft scales with non-trivial high bits so g>=4 depends on cross-byte packing. */
  uint8_t sc[12] = {
      0xC1, 0x42, 0x85, 0x0F, /* 0..3 */
      0xF0, 0x3C, 0x83, 0x55, /* 4..7 */
      0x12, 0x34, 0x56, 0x78  /* 8..11 */
  };
  memcpy(block.scales, sc, sizeof(sc));

  /* Put distinct q values at the first element of each 32-weight sub-block.
   * Indices: 0,32,64,96,128,160,192,224 */
  block.qs[0] = (uint8_t)((2u << 4) | 1u);
  block.qs[32] = (uint8_t)((4u << 4) | 3u);
  block.qs[64] = (uint8_t)((6u << 4) | 5u);
  block.qs[96] = (uint8_t)((8u << 4) | 7u);

  W.data = &block;
  W.byte_size = sizeof(block);

  float x[256];
  memset(x, 0, sizeof(x));
  x[0] = 1.0f;
  x[32] = 1.0f;
  x[64] = 1.0f;
  x[96] = 1.0f;
  x[128] = 1.0f;
  x[160] = 1.0f;
  x[192] = 1.0f;
  x[224] = 1.0f;

  float y[1] = {0};
  int ret = matvec_q4k_f32(&W, x, y);
  assert(ret == 0);

  int qvals[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float expected = 0.0f;
  for (int g = 0; g < 8; g++) {
    uint8_t su6 = expected_scale_u6(sc, g);
    uint8_t mu6 = expected_min_u6(sc, g);
    expected += (float)su6 * (float)qvals[g] - (float)mu6;
  }

  if (fabsf(y[0] - expected) > 1e-4f) {
    printf("  FAIL: q4_k_scales_and_nibbles_mapping (got %f, expected %f)\n",
           y[0], expected);
    exit(1);
  }
  printf("  PASS: q4_k_scales_and_nibbles_mapping\n");
}

static void test_q4_k_byte_size_bounds(void) {
  Tensor W;
  W.ndim = 2;
  W.shape[0] = 1;
  W.shape[1] = 256;
  W.dtype = DTYPE_Q4_K;

  block_q4_k block;
  memset(&block, 0, sizeof(block));
  W.data = &block;

  float x[256];
  memset(x, 0, sizeof(x));
  float y[1] = {0};

  W.byte_size = sizeof(block) - 1; // too small
  int ret = matvec_q4k_f32(&W, x, y);
  if (ret == 0) {
    printf("  FAIL: q4_k_byte_size_bounds (expected failure)\n");
    exit(1);
  }
  printf("  PASS: q4_k_byte_size_bounds\n");
}

#ifdef USE_CUDA
extern int cuda_matvec_q4k_f32_impl(const void *W_data, const float *x, float *y, int num_rows, int num_cols);
extern void* cuda_upload_weight(const void* host_ptr, size_t size);
extern void cudaFree(void* devPtr);

static void test_cuda_parity_q4k(void) {
  Tensor W;
  W.ndim = 2;
  W.shape[0] = 32;
  W.shape[1] = 1024;
  W.dtype = DTYPE_Q4_K;

  int num_blocks = 32 * (1024 / 256);
  block_q4_k* blocks = (block_q4_k*)malloc(sizeof(block_q4_k) * num_blocks);
  memset(blocks, 0, sizeof(block_q4_k) * num_blocks);
  for (int i = 0; i < num_blocks; i++) {
    blocks[i].d = FP16_ONE;
    blocks[i].dmin = FP16_ONE;
    for(int j=0; j<12; j++) blocks[i].scales[j] = (uint8_t)(j % 8);
    for(int j=0; j<128; j++) blocks[i].qs[j] = (uint8_t)(j % 256);
  }
  W.data = blocks;
  W.byte_size = sizeof(block_q4_k) * num_blocks;

  float* x = (float*)malloc(sizeof(float) * 1024);
  for (int i = 0; i < 1024; i++) x[i] = (float)i / 1024.0f;

  float y_cpu[32] = {0};
  float y_gpu[32] = {0};

  // CPU
  int ret_cpu = matvec_q4k_f32(&W, x, y_cpu);
  assert(ret_cpu == 0);

  // GPU
  void* d_W = cuda_upload_weight(blocks, W.byte_size);
  assert(d_W != NULL);
  int ret_gpu = cuda_matvec_q4k_f32_impl(d_W, x, y_gpu, 32, 1024);
  assert(ret_gpu == 0);

  // Compare
  float max_err = 0.0f;
  for (int i = 0; i < 32; i++) {
    float err = fabsf(y_cpu[i] - y_gpu[i]);
    if (err > max_err) max_err = err;
  }

  // Release
  // Not using exact cuda namespace here, but if cudaFree works:
  // Wait, I can't guarantee `cudaFree` is accessible if I don't include cuda_runtime.h.
  // We can leave it allocated for the test, or just skip free.
  free(blocks);
  free(x);

  if (max_err > 1e-4f) {
    printf("  FAIL: cuda_parity_q4k (max err %f)\n", max_err);
    exit(1);
  }
  printf("  PASS: cuda_parity_q4k (max err %f)\n", max_err);
}
#endif

int main(void) {
  printf("=== Quantization Test Suite ===\n");
  test_q8_0_matvec();
  test_q8_0_byte_size_bounds();
  test_q4_k_matvec();
  test_q4_k_scales_and_nibbles_mapping();
  test_q4_k_byte_size_bounds();
#ifdef USE_CUDA
  test_cuda_parity_q4k();
#endif
  printf("\nAll tests PASSED ✓\n");
  return 0;
}
