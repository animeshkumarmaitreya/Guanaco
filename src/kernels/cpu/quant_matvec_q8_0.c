#include "quant.h"

#include <string.h>

#pragma pack(push, 1)
typedef struct {
  uint16_t scale;
  int8_t quants[32];
} block_q8_0;
#pragma pack(pop)

static inline float extract_f16_to_f32(uint16_t h) {
  if (h == 0)
    return 0.0f;
  uint32_t sign = (h & 0x8000) << 16;
  uint32_t exp = (h & 0x7c00) >> 10;
  uint32_t frac = (h & 0x03ff);
  uint32_t float_exp = exp + (127 - 15);
  uint32_t r = sign | (float_exp << 23) | (frac << 13);
  float f;
  memcpy(&f, &r, sizeof(float));
  return f;
}

int matvec_q8_0_f32(const Tensor *W_q8_0, const float *x, float *y) {
  if (W_q8_0->dtype != DTYPE_Q8_0 || W_q8_0->ndim != 2)
    return -1;

  int num_rows = W_q8_0->shape[0];
  int num_cols = W_q8_0->shape[1];
  if (num_cols % 32 != 0)
    return -2;

  int num_blocks = num_cols / 32;
  const block_q8_0 *blocks = (const block_q8_0 *)W_q8_0->data;

  for (int r = 0; r < num_rows; r++) {
    float sum = 0.0f;
    const block_q8_0 *row_blocks = blocks + r * num_blocks;
    for (int b = 0; b < num_blocks; b++) {
      float scale = extract_f16_to_f32(row_blocks[b].scale);
      const int8_t *qs = row_blocks[b].quants;
      const float *px = x + b * 32;

      float block_sum = 0.0f;
      for (int i = 0; i < 32; i++) {
        block_sum += qs[i] * px[i];
      }
      sum += block_sum * scale;
    }
    y[r] = sum;
  }

  return 0;
}
