#include "quant.h"

#include <string.h>

#pragma pack(push, 1)
typedef struct {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[128];
} block_q4_k;
#pragma pack(pop)

static inline float extract_f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t frac = h & 0x03ffu;

  uint32_t bits;
  if (exp == 0) {
    if (frac == 0) {
      bits = sign;
    } else {
      uint32_t e = 127u - 15u + 1u;
      while ((frac & 0x0400u) == 0) {
        frac <<= 1;
        e--;
      }
      frac &= 0x03ffu;
      bits = sign | (e << 23) | (frac << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (frac << 13);
  } else {
    bits = sign | ((exp + (127u - 15u)) << 23) | (frac << 13);
  }

  float f;
  memcpy(&f, &bits, sizeof(float));
  return f;
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d,
                                    uint8_t *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

int matvec_q4k_f32(const Tensor *W_q4k, const float *x, float *y) {
  if (W_q4k == NULL || x == NULL || y == NULL)
    return -1;
  if (W_q4k->dtype != DTYPE_Q4_K || W_q4k->ndim != 2)
    return -1;

  int num_rows = W_q4k->shape[0];
  int num_cols = W_q4k->shape[1];
  if (num_cols % 256 != 0)
    return -2;

  int num_blocks = num_cols / 256;
  if (W_q4k->byte_size != 0) {
    size_t need = (size_t)num_rows * (size_t)num_blocks * sizeof(block_q4_k);
    if (need > W_q4k->byte_size)
      return -3;
  }
  const block_q4_k *blocks = (const block_q4_k *)W_q4k->data;

  for (int r = 0; r < num_rows; r++) {
    float sum = 0.0f;
    const block_q4_k *row_blocks = blocks + r * num_blocks;

    for (int b = 0; b < num_blocks; b++) {
      const block_q4_k *blk = &row_blocks[b];
      const float d_all = extract_f16_to_f32(blk->d);
      const float min_all = extract_f16_to_f32(blk->dmin);

      const uint8_t *q = blk->qs;
      const float *px = x + b * 256;

      int is = 0;
      uint8_t sc, m;

      // 4 super-iterations, each processing 64 weights
      for (int j = 0; j < 256; j += 64) {
        // First 32 weights
        get_scale_min_k4(is + 0, blk->scales, &sc, &m);
        const float d1 = d_all * sc;
        const float m1 = min_all * m;

        // Second 32 weights
        get_scale_min_k4(is + 1, blk->scales, &sc, &m);
        const float d2 = d_all * sc;
        const float m2 = min_all * m;

        float sum_sub1 = 0.0f;
        float sum_sub2 = 0.0f;

        // Dequantize and dot with x
        for (int l = 0; l < 32; ++l) {
          float w1 = d1 * (q[l] & 0xF) - m1;
          sum_sub1 += w1 * px[l];
        }
        for (int l = 0; l < 32; ++l) {
          float w2 = d2 * (q[l] >> 4) - m2;
          sum_sub2 += w2 * px[32 + l];
        }

        sum += sum_sub1 + sum_sub2;

        q += 32;
        px += 64;
        is += 2;
      }
    }
    y[r] = sum;
  }

  return 0;
}
