#include "quant.h"

#include <string.h>

#pragma pack(push, 1)
typedef struct {
    uint8_t ql[128];      // lower 4 bits
    uint8_t qh[64];       // upper 2 bits
    int8_t  scales[16];   // block scales
    uint16_t d;           // super-block scale
} block_q6_k;
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

int matvec_q6k_f32(const Tensor *W_q6k, const float *x, float *y) {
  if (W_q6k == NULL || x == NULL || y == NULL)
    return -1;
  if (W_q6k->dtype != DTYPE_Q6_K || W_q6k->ndim != 2)
    return -1;

  int num_rows = W_q6k->shape[0];
  int num_cols = W_q6k->shape[1];
  if (num_cols % 256 != 0)
    return -2;

  int num_blocks = num_cols / 256;
  if (W_q6k->byte_size != 0) {
    size_t need = (size_t)num_rows * (size_t)num_blocks * sizeof(block_q6_k);
    if (need > W_q6k->byte_size)
      return -3;
  }
  const block_q6_k *blocks = (const block_q6_k *)W_q6k->data;

  for (int r = 0; r < num_rows; r++) {
    float sum = 0.0f;
    const block_q6_k *row_blocks = blocks + r * num_blocks;

    for (int b = 0; b < num_blocks; b++) {
      const block_q6_k *blk = &row_blocks[b];
      const float d_all = extract_f16_to_f32(blk->d);

      const uint8_t *ql = blk->ql;
      const uint8_t *qh = blk->qh;
      const int8_t  *sc = blk->scales;
      const float *px = x + b * 256;

      for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

            sum += d_all * sc[is + 0] * q1 * px[l +  0];
            sum += d_all * sc[is + 2] * q2 * px[l + 32];
            sum += d_all * sc[is + 4] * q3 * px[l + 64];
            sum += d_all * sc[is + 6] * q4 * px[l + 96];
        }
        px += 128;
        ql += 64;
        qh += 32;
        sc += 8;
      }
    }
    y[r] = sum;
  }

  return 0;
}
