/*============================================================================
 * Transformer Engine — Forward Pass
 *
 * Implements:
 *   - Embedding lookup
 *   - transformer_layer():  attention + MLP with residual connections
 *   - forward():           embed → N layers → final norm → logits
 *
 * Uses kernel API (kernels.h) and memory API (memory.h) — works with
 * both stubs and real implementations.
 *============================================================================*/

#include "engine.h"
#include "memory.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Tensor helpers ---------- */

/* Allocate a scratch tensor with given shape (FP32) */
static Tensor *scratch_tensor(Scratch *scr, int ndim, int d0, int d1, int d2,
                              int d3) {
  Tensor *t = (Tensor *)scratch_alloc(scr, sizeof(Tensor), 64);
  t->ndim = ndim;
  t->shape[0] = d0;
  t->shape[1] = d1;
  t->shape[2] = d2;
  t->shape[3] = d3;
  t->dtype = DTYPE_F32;
  t->byte_size = 0;

  /* Compute strides (row-major) */
  int dims[4] = {d0, d1, d2, d3};
  t->stride[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    t->stride[i] = t->stride[i + 1] * dims[i + 1];
  }

  /* Allocate data */
  int numel = tensor_numel(t);
  t->data = scratch_alloc(scr, numel * sizeof(float), 64);
  t->byte_size = (size_t)numel * sizeof(float);
  return t;
}

/* Create a tensor view (no data copy, shares underlying buffer) */
static Tensor make_view(void *data, int ndim, int d0, int d1, int d2) {
  Tensor t;
  t.data = data;
  t.ndim = ndim;
  t.dtype = DTYPE_F32;
  t.byte_size = 0;
  t.shape[0] = d0;
  t.shape[1] = d1;
  t.shape[2] = d2;
  t.shape[3] = 0;

  if (ndim == 1) {
    t.stride[0] = 1;
  } else if (ndim == 2) {
    t.stride[0] = d1;
    t.stride[1] = 1;
  } else if (ndim == 3) {
    t.stride[0] = d1 * d2;
    t.stride[1] = d2;
    t.stride[2] = 1;
  } else {
    assert(!"make_view only supports ndim 1..3");
  }
  t.stride[3] = 0;
  t.byte_size = (size_t)tensor_numel(&t) * sizeof(float);
  return t;
}

/* ---------- Linear dispatch (single integration point for Phase B quant)
 * ---------- */

static void linear_dispatch(const Tensor *X, const Tensor *W, Tensor *Y,
                            const KernelVTable *k) {
  assert(X != NULL && W != NULL && Y != NULL);
  assert(k != NULL);

#ifndef NDEBUG
  assert(X->data != NULL && W->data != NULL && Y->data != NULL);
  assert(X->ndim == 2);
  assert(W->ndim == 2);
  assert(Y->ndim == 2);
  assert(X->dtype == DTYPE_F32);
  assert(Y->dtype == DTYPE_F32);
  assert(tensor_is_contiguous_row_major(X));
  assert(tensor_is_contiguous_row_major(W));
  assert(tensor_is_contiguous_row_major(Y));
  assert(X->shape[1] == W->shape[1]);
  assert(Y->shape[0] == X->shape[0]);
  assert(Y->shape[1] == W->shape[0]);
#endif

  /* Today: loader enforces F32-only weights. This switch is here so Phase B
   * can integrate quant matvec at a single call site. */
  if (W->dtype == DTYPE_F32) {
    assert(k->gemm_f32 != NULL);
    k->gemm_f32(X, W, Y);
    return;
  }

  int T = X->shape[0];
  int in_dim = W->shape[1];
  int out_dim = W->shape[0];

  for (int t = 0; t < T; t++) {
    float *x_t = (float *)X->data + t * in_dim;
    float *y_t = (float *)Y->data + t * out_dim;

    if (W->dtype == DTYPE_Q8_0) {
      assert(k->matvec_q8_0_f32 != NULL);
      int ret = k->matvec_q8_0_f32(W, x_t, y_t);
      if (ret != 0) {
        fprintf(stderr, "linear_dispatch: matvec_q8_0_f32 failed (ret=%d)\n",
                ret);
        abort();
      }
    } else if (W->dtype == DTYPE_Q4_K) {
      assert(k->matvec_q4k_f32 != NULL);
      int ret = k->matvec_q4k_f32(W, x_t, y_t);
      if (ret != 0) {
        fprintf(stderr, "linear_dispatch: matvec_q4k_f32 failed (ret=%d)\n",
                ret);
        abort();
      }
    } else if (W->dtype == DTYPE_Q6_K) {
      assert(k->matvec_q6k_f32 != NULL);
      int ret = k->matvec_q6k_f32(W, x_t, y_t);
      if (ret != 0) {
        fprintf(stderr, "linear_dispatch: matvec_q6k_f32 failed (ret=%d)\n",
                ret);
        abort();
      }
    } else {
      fprintf(stderr,
              "linear_dispatch: quant weights not supported yet (dtype=%d)\n",
              (int)W->dtype);
      abort();
    }
  }
}

/* ---------- Embedding lookup ---------- */

#pragma pack(push, 1)
typedef struct {
  uint16_t scale;
  int8_t quants[32];
} block_q8_0_embd;

typedef struct {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[128];
} block_q4_k_embd;
#pragma pack(pop)

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

static inline float extract_f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t frac = h & 0x03ffu;

  uint32_t bits;
  if (exp == 0) {
    if (frac == 0) {
      bits = sign;
    } else {
      /* Subnormal half: normalize the mantissa. */
      uint32_t e = 127u - 15u + 1u;
      while ((frac & 0x0400u) == 0) {
        frac <<= 1;
        e--;
      }
      frac &= 0x03ffu;
      bits = sign | (e << 23) | (frac << 13);
    }
  } else if (exp == 31) {
    /* Inf/NaN */
    bits = sign | 0x7f800000u | (frac << 13);
  } else {
    bits = sign | ((exp + (127u - 15u)) << 23) | (frac << 13);
  }

  float f;
  memcpy(&f, &bits, sizeof(float));
  return f;
}

static void embedding_lookup(const Tensor *embedding, int *token_ids,
                             int n_tokens, Tensor *output) {
  int H = embedding->shape[1];
  float *out_data = (float *)output->data;

  for (int t = 0; t < n_tokens; t++) {
    int tok = token_ids[t];
    assert(tok >= 0 && tok < embedding->shape[0]);

    if (embedding->dtype == DTYPE_F32) {
      float *emb_data = (float *)embedding->data;
      memcpy(out_data + t * H, emb_data + tok * H, H * sizeof(float));
    } else if (embedding->dtype == DTYPE_Q8_0) {
      /* Dequantize Q8_0 embedding row on the fly */
      block_q8_0_embd *emb_data = (block_q8_0_embd *)embedding->data;
      int num_blocks = H / 32;
      float *out_row = out_data + t * H;
      block_q8_0_embd *row_blocks = emb_data + tok * num_blocks;
      for (int b = 0; b < num_blocks; b++) {
        float scale = extract_f16_to_f32(row_blocks[b].scale);
        for (int i = 0; i < 32; i++) {
          out_row[b * 32 + i] = row_blocks[b].quants[i] * scale;
        }
      }
    } else if (embedding->dtype == DTYPE_Q4_K) {
      /* Dequantize Q4_K embedding row on the fly */
      block_q4_k_embd *emb_data = (block_q4_k_embd *)embedding->data;
      int num_blocks = H / 256;
      float *out_row = out_data + t * H;
      block_q4_k_embd *row_blocks = emb_data + tok * num_blocks;

      for (int b = 0; b < num_blocks; b++) {
        const float d_all = extract_f16_to_f32(row_blocks[b].d);
        const float min_all = extract_f16_to_f32(row_blocks[b].dmin);
        const uint8_t *q = row_blocks[b].qs;
        int is = 0;
        uint8_t sc, m;

        float *px = out_row + b * 256;

        for (int j = 0; j < 256; j += 64) {
          get_scale_min_k4(is + 0, row_blocks[b].scales, &sc, &m);
          const float d1 = d_all * sc;
          const float m1 = min_all * m;

          get_scale_min_k4(is + 1, row_blocks[b].scales, &sc, &m);
          const float d2 = d_all * sc;
          const float m2 = min_all * m;

          for (int l = 0; l < 32; ++l) {
            px[l] = d1 * (q[l] & 0xF) - m1;
          }
          for (int l = 0; l < 32; ++l) {
            px[32 + l] = d2 * (q[l] >> 4) - m2;
          }
          q += 32;
          px += 64;
          is += 2;
        }
      }
    } else {
      fprintf(stderr, "embedding_lookup: unsupported embedding dtype %d\n",
              embedding->dtype);
      abort();
    }
  }
}

/* ---------- Transformer Layer ---------- */

void transformer_layer(Tensor *hidden, LayerWeights *weights, KVCache *kv,
                       int layer, int pos, Scratch *scr, ModelConfig *cfg,
                       const KernelVTable *k) {
  assert(k != NULL);
  assert(k->gemm_f32 != NULL);
  assert(k->gemm_f32_nn != NULL);
  assert(k->rmsnorm != NULL);
  assert(k->softmax_inplace != NULL);
  assert(k->silu_inplace != NULL);
  assert(k->rope != NULL);
  assert(k->residual_add != NULL);
  assert(k->elemwise_mul != NULL);

  int T = hidden->shape[0]; /* number of tokens (1 for decode, T for prefill) */
  int H = cfg->hidden_dim;
  int n_heads = cfg->n_heads;
  int n_kv_heads = cfg->n_kv_heads;
  int head_dim = cfg->head_dim;
  int kv_dim = n_kv_heads * head_dim; /* total KV projection size */
  assert(n_kv_heads > 0);
  assert(n_heads % n_kv_heads == 0);
  int heads_per_kv = n_heads / n_kv_heads; /* for GQA */

  /* ---- Step 1: Save residual ---- */
  Tensor *residual = scratch_tensor(scr, 2, T, H, 0, 0);
  memcpy(residual->data, hidden->data, T * H * sizeof(float));

  /* ---- Step 2: Pre-attention RMSNorm ---- */
  Tensor *normed = scratch_tensor(scr, 2, T, H, 0, 0);
  k->rmsnorm(hidden, weights->rms_att, normed, 1e-6f);

  /* ---- Step 3: QKV projections ---- */
  Tensor *Q = scratch_tensor(scr, 2, T, H, 0, 0);
  Tensor *K = scratch_tensor(scr, 2, T, kv_dim, 0, 0);
  Tensor *V = scratch_tensor(scr, 2, T, kv_dim, 0, 0);

  linear_dispatch(normed, weights->wq, Q, k); /* (T, H) × (H, H) = (T, H) */
  linear_dispatch(normed, weights->wk, K,
                  k); /* (T, H) × (H, kv_dim) = (T, kv_dim) */
  linear_dispatch(normed, weights->wv, V,
                  k); /* (T, H) × (H, kv_dim) = (T, kv_dim) */

  /* ---- Step 4: RoPE on Q and K ---- */
  for (int t = 0; t < T; t++) {
    float *q_t = (float *)Q->data + t * H;
    float *k_t = (float *)K->data + t * kv_dim;

    Tensor q_view = make_view(q_t, 2, n_heads, head_dim, 0);
    Tensor k_view = make_view(k_t, 2, n_kv_heads, head_dim, 0);

    k->rope(&q_view, &k_view, pos + t, head_dim);
  }

  /* ---- Step 5: Append K, V to cache ---- */
  for (int t = 0; t < T; t++) {
    float *k_t = (float *)K->data + t * kv_dim;
    float *v_t = (float *)V->data + t * kv_dim;

    Tensor k_tok = make_view(k_t, 2, n_kv_heads, head_dim, 0);
    Tensor v_tok = make_view(v_t, 2, n_kv_heads, head_dim, 0);

    kv_cache_append(kv, layer, &k_tok, &v_tok, pos + t);
  }

  /* ---- Step 6: Get cached K, V ---- */
  int seq_len = pos + T; /* total sequence length so far */
  Tensor *K_cached =
      kv_cache_get_k(kv, layer, seq_len); /* (n_kv_heads, seq_len, head_dim) */
  Tensor *V_cached =
      kv_cache_get_v(kv, layer, seq_len); /* (n_kv_heads, seq_len, head_dim) */

#ifndef NDEBUG
  assert(K_cached != NULL && V_cached != NULL);
  assert(K_cached->dtype == DTYPE_F32);
  assert(V_cached->dtype == DTYPE_F32);
  assert(K_cached->ndim == 3);
  assert(V_cached->ndim == 3);
  /* Expect head-major contiguous cache: (n_kv_heads, seq_len, head_dim) */
  assert(K_cached->shape[0] == n_kv_heads);
  assert(K_cached->shape[1] == seq_len);
  assert(K_cached->shape[2] == head_dim);
  assert(V_cached->shape[0] == n_kv_heads);
  assert(V_cached->shape[1] == seq_len);
  assert(V_cached->shape[2] == head_dim);
  assert(K_cached->stride[2] == 1);
  assert(V_cached->stride[2] == 1);
  assert(K_cached->stride[1] == head_dim);
  assert(V_cached->stride[1] == head_dim);
#endif

  /* ---- Step 7-10: Multi-head attention ---- */
  Tensor *attn_output = scratch_tensor(scr, 2, T, H, 0, 0);

  float scale = 1.0f / sqrtf((float)head_dim);

  /* Reusable temporaries to avoid per-head scratch blow-up.
   * We overwrite these buffers for each head. */
  Tensor *q_pack = scratch_tensor(scr, 2, T, head_dim, 0, 0);
  Tensor *scores = scratch_tensor(scr, 2, T, seq_len, 0, 0);
  Tensor *ctx = scratch_tensor(scr, 2, T, head_dim, 0, 0);

#ifndef NDEBUG
  assert(tensor_is_contiguous_row_major(q_pack));
  assert(tensor_is_contiguous_row_major(scores));
  assert(tensor_is_contiguous_row_major(ctx));
  assert(tensor_is_contiguous_row_major(attn_output));
  assert(tensor_is_contiguous_row_major(Q));
#endif

  float *q_pack_data = (float *)q_pack->data;
  float *scores_data = (float *)scores->data;
  float *ctx_data = (float *)ctx->data;
  float *attn_data = (float *)attn_output->data;
  float *Q_data = (float *)Q->data;

  for (int h = 0; h < n_heads; h++) {
    const int kv_h =
        h / heads_per_kv; /* GQA: which KV head this Q head maps to */

    /* Pack Q_h into a contiguous (T, head_dim) matrix.
     * Q is stored as (T, H) so the per-head slice is strided by H per row. */
    for (int tq = 0; tq < T; tq++) {
      memcpy(q_pack_data + (size_t)tq * head_dim,
             Q_data + (size_t)tq * H + (size_t)h * head_dim,
             (size_t)head_dim * sizeof(float));
    }

    /* K/V for this KV head from cache: underlying layout is head-major. */
    float *k_h = (float *)K_cached->data + (size_t)kv_h * K_cached->stride[0];
    float *v_h = (float *)V_cached->data + (size_t)kv_h * V_cached->stride[0];

    Tensor K_head =
        make_view(k_h, 2, seq_len, head_dim, 0); /* (N=seq_len, K=head_dim) */
    Tensor V_head =
        make_view(v_h, 2, seq_len, head_dim, 0); /* (K=seq_len, N=head_dim) */

    /* Step 7: scores = Q × Kᵀ */
    k->gemm_f32(
        q_pack, &K_head,
        scores); /* (T, head_dim) × (seq_len, head_dim) => (T, seq_len) */

    /* Step 7b/8: scale + causal mask */
    for (int tq = 0; tq < T; tq++) {
      const int query_pos = pos + tq;
      float *row = scores_data + (size_t)tq * seq_len;
      for (int s = 0; s < seq_len; s++) {
        float v = row[s] * scale;
        if (s > query_pos)
          v = -INFINITY;
        row[s] = v;
      }
    }

    /* Step 9: softmax over last dim for each row */
    k->softmax_inplace(scores, seq_len);

    /* Step 10: ctx = scores × V  (standard GEMM: (T, seq_len) × (seq_len,
     * head_dim)) */
    k->gemm_f32_nn(scores, &V_head, ctx);

    /* Scatter ctx into the packed (T, H) attention output buffer */
    for (int tq = 0; tq < T; tq++) {
      memcpy(attn_data + (size_t)tq * H + (size_t)h * head_dim,
             ctx_data + (size_t)tq * head_dim,
             (size_t)head_dim * sizeof(float));
    }
  }

  /* ---- Step 11: Output projection ---- */
  Tensor *attn_proj = scratch_tensor(scr, 2, T, H, 0, 0);
  linear_dispatch(attn_output, weights->wo, attn_proj,
                  k); /* (T, H) × (H, H) = (T, H) */

  /* ---- Step 12: Residual add ---- */
  k->residual_add(attn_proj, residual);

  /* ---- Step 13: Save residual ---- */
  memcpy(residual->data, attn_proj->data, T * H * sizeof(float));

  /* ---- Step 14: Pre-MLP RMSNorm ---- */
  Tensor *normed2 = scratch_tensor(scr, 2, T, H, 0, 0);
  k->rmsnorm(attn_proj, weights->rms_ffn, normed2, 1e-6f);

  /* ---- Step 15-19: SwiGLU MLP ---- */
  int ff_dim = cfg->ff_dim;
  Tensor *gate = scratch_tensor(scr, 2, T, ff_dim, 0, 0);
  Tensor *up = scratch_tensor(scr, 2, T, ff_dim, 0, 0);

  /* Step 15: gate = x × W_gate */
  linear_dispatch(normed2, weights->w_gate, gate,
                  k); /* (T, H) × (H, ff) = (T, ff) */

  /* Step 16: up = x × W_up */
  linear_dispatch(normed2, weights->w_up, up,
                  k); /* (T, H) × (H, ff) = (T, ff) */

  /* Step 17: SiLU(gate) */
  k->silu_inplace(gate);

  /* Step 18: gate = gate ⊙ up */
  k->elemwise_mul(gate, up);

  /* Step 19: down = gate × W_down */
  Tensor *down = scratch_tensor(scr, 2, T, H, 0, 0);
  linear_dispatch(gate, weights->w_down, down,
                  k); /* (T, ff) × (ff, H) = (T, H) */

  /* ---- Step 20: Residual add ---- */
  k->residual_add(down, residual);

  /* Write back to hidden */
  memcpy(hidden->data, down->data, T * H * sizeof(float));
}

/* ---------- Forward Pass ---------- */

Tensor *forward(ModelWeights *model, KVCache *kv, Scratch *scr, int *token_ids,
                int n_tokens, int pos, const KernelVTable *k) {
  assert(k != NULL);
  assert(k->gemm_f32 != NULL);
  assert(k->rmsnorm != NULL);

  ModelConfig *cfg = &model->config;
  int H = cfg->hidden_dim;

  /* Reset scratch for this forward pass */
  scratch_reset(scr);

  /* Step 1: Embedding lookup */
  Tensor *hidden = scratch_tensor(scr, 2, n_tokens, H, 0, 0);
  embedding_lookup(model->embedding, token_ids, n_tokens, hidden);

  /* Step 2: Run all transformer layers */
  for (int l = 0; l < cfg->n_layers; l++) {
    transformer_layer(hidden, &model->layers[l], kv, l, pos, scr, cfg, k);
  }

  /* Step 3: Final RMSNorm */
  Tensor *normed = scratch_tensor(scr, 2, n_tokens, H, 0, 0);
  k->rmsnorm(hidden, model->rms_final, normed, 1e-6f);

  /* Step 4: logits = normed[-1] × lm_head
   * We only need the last token's logits for generation */
  float *last_hidden = (float *)normed->data + (n_tokens - 1) * H;
  Tensor last_view = make_view(last_hidden, 2, 1, H, 0);

  Tensor *logits = scratch_tensor(scr, 2, 1, cfg->vocab_size, 0, 0);
  linear_dispatch(&last_view, model->lm_head, logits,
                  k); /* (1, H) × (H, vocab_size) = (1, V) */

  return logits;
}
