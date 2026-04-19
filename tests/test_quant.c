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

int main(void) {
  printf("=== Quantization Test Suite ===\n");
  test_q8_0_matvec();
  test_q4_k_matvec();
  printf("\nAll tests PASSED ✓\n");
  return 0;
}
