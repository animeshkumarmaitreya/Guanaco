#include "quant.h"
#include "threadpool.h"

#include <string.h>
#include <immintrin.h>

extern ThreadPool* g_threadpool;

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

typedef struct {
  const Tensor *W_q4k;
  const float *x;
  float *y;
  int num_blocks;
} MatvecQ4KCtx;

static void matvec_q4k_cols(int start_row, int end_row, void *vctx) {
  MatvecQ4KCtx *ctx = (MatvecQ4KCtx *)vctx;
  const block_q4_k *blocks = (const block_q4_k *)ctx->W_q4k->data;
  const float *x = ctx->x;
  float *y = ctx->y;
  int num_blocks = ctx->num_blocks;

  for (int r = start_row; r < end_row; r++) {
    float sum = 0.0f;
    const block_q4_k *row_blocks = blocks + r * num_blocks;

    for (int b = 0; b < num_blocks; b++) {
      const block_q4_k *blk = &row_blocks[b];
      /* Prefetch next block to L1 cache */
      if (b + 1 < num_blocks)
        _mm_prefetch((const char*)&row_blocks[b + 1], _MM_HINT_T0);
      const float d_all = extract_f16_to_f32(blk->d);
      const float min_all = extract_f16_to_f32(blk->dmin);

      const uint8_t *q = blk->qs;
      const float *px = x + b * 256;

      int is = 0;
      uint8_t sc, m;

      __m256 sum_vec = _mm256_setzero_ps();

      // 4 super-iterations, each processing 64 weights
      for (int j = 0; j < 256; j += 64) {
        get_scale_min_k4(is + 0, blk->scales, &sc, &m);
        const float d1 = d_all * sc;
        const float m1 = min_all * m;

        get_scale_min_k4(is + 1, blk->scales, &sc, &m);
        const float d2 = d_all * sc;
        const float m2 = min_all * m;

        __m256 vd1 = _mm256_set1_ps(d1);
        __m256 vm1 = _mm256_set1_ps(m1);
        __m256 vd2 = _mm256_set1_ps(d2);
        __m256 vm2 = _mm256_set1_ps(m2);

        // Dequantize and dot with x using AVX2 (8 floats per step)
        for (int step = 0; step < 4; ++step) {
          __m128i q8 = _mm_loadl_epi64((const __m128i*)(q + step * 8));
          __m256i q32 = _mm256_cvtepu8_epi32(q8);
          
          __m256i mask = _mm256_set1_epi32(0xF);
          __m256i q1_32 = _mm256_and_si256(q32, mask);
          __m256i q2_32 = _mm256_srli_epi32(q32, 4);

          __m256 f1 = _mm256_cvtepi32_ps(q1_32);
          __m256 f2 = _mm256_cvtepi32_ps(q2_32);

          __m256 w1 = _mm256_fmsub_ps(vd1, f1, vm1);
          __m256 w2 = _mm256_fmsub_ps(vd2, f2, vm2);

          __m256 vx1 = _mm256_loadu_ps(px + step * 8);
          __m256 vx2 = _mm256_loadu_ps(px + 32 + step * 8);

          sum_vec = _mm256_fmadd_ps(w1, vx1, sum_vec);
          sum_vec = _mm256_fmadd_ps(w2, vx2, sum_vec);
        }

        q += 32;
        px += 64;
        is += 2;
      }
      
      float temp[8];
      _mm256_storeu_ps(temp, sum_vec);
      sum += temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    }
    y[r] = sum;
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

  MatvecQ4KCtx ctx = { W_q4k, x, y, num_blocks };
  
  if (g_threadpool != NULL && threadpool_num_threads(g_threadpool) > 1 && num_rows >= 32) {
      int threads = threadpool_num_threads(g_threadpool);
      int chunk_size = num_rows / threads;
      if (chunk_size < 8) chunk_size = 8;
      
      threadpool_parallel_for(g_threadpool, 0, num_rows, chunk_size, matvec_q4k_cols, &ctx);
  } else {
      matvec_q4k_cols(0, num_rows, &ctx);
  }

  return 0;
}
