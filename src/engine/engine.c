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
#include "kernels.h"
#include "memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* ---------- Tensor helpers ---------- */

/* Allocate a scratch tensor with given shape (FP32) */
static Tensor* scratch_tensor(Scratch* scr, int ndim, int d0, int d1, int d2, int d3) {
    Tensor* t = (Tensor*)scratch_alloc(scr, sizeof(Tensor), 64);
    t->ndim = ndim;
    t->shape[0] = d0; t->shape[1] = d1; t->shape[2] = d2; t->shape[3] = d3;
    t->dtype = DTYPE_F32;
    t->byte_size = 0;

    /* Compute strides (row-major) */
    int dims[4] = {d0, d1, d2, d3};
    t->stride[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        t->stride[i] = t->stride[i+1] * dims[i+1];
    }

    /* Allocate data */
    int numel = tensor_numel(t);
    t->data = scratch_alloc(scr, numel * sizeof(float), 64);
    t->byte_size = (size_t)numel * sizeof(float);
    return t;
}

/* Create a tensor view (no data copy, shares underlying buffer) */
static Tensor make_view(void* data, int ndim, int d0, int d1, int d2) {
    Tensor t;
    t.data = data;
    t.ndim = ndim;
    t.dtype = DTYPE_F32;
    t.byte_size = 0;
    t.shape[0] = d0; t.shape[1] = d1; t.shape[2] = d2; t.shape[3] = 0;

    if (ndim == 1) { t.stride[0] = 1; }
    else if (ndim == 2) { t.stride[0] = d1; t.stride[1] = 1; }
    else if (ndim == 3) { t.stride[0] = d1 * d2; t.stride[1] = d2; t.stride[2] = 1; }
    else { assert(!"make_view only supports ndim 1..3"); }
    t.stride[3] = 0;
    t.byte_size = (size_t)tensor_numel(&t) * sizeof(float);
    return t;
}

/* ---------- Embedding lookup ---------- */

static void embedding_lookup(const Tensor* embedding, int* token_ids, int n_tokens,
                              Tensor* output) {
    int H = embedding->shape[1];
    float* emb_data = (float*)embedding->data;
    float* out_data = (float*)output->data;

    for (int t = 0; t < n_tokens; t++) {
        int tok = token_ids[t];
        assert(tok >= 0 && tok < embedding->shape[0]);
        memcpy(out_data + t * H, emb_data + tok * H, H * sizeof(float));
    }
}

/* ---------- Transformer Layer ---------- */

void transformer_layer(Tensor* hidden, LayerWeights* weights, KVCache* kv,
                       int layer, int pos, Scratch* scr, ModelConfig* cfg) {
    int T = hidden->shape[0];   /* number of tokens (1 for decode, T for prefill) */
    int H = cfg->hidden_dim;
    int n_heads = cfg->n_heads;
    int n_kv_heads = cfg->n_kv_heads;
    int head_dim = cfg->head_dim;
    int kv_dim = n_kv_heads * head_dim;  /* total KV projection size */
    assert(n_kv_heads > 0);
    assert(n_heads % n_kv_heads == 0);
    int heads_per_kv = n_heads / n_kv_heads;  /* for GQA */

    /* ---- Step 1: Save residual ---- */
    Tensor* residual = scratch_tensor(scr, 2, T, H, 0, 0);
    memcpy(residual->data, hidden->data, T * H * sizeof(float));

    /* ---- Step 2: Pre-attention RMSNorm ---- */
    Tensor* normed = scratch_tensor(scr, 2, T, H, 0, 0);
    rmsnorm(hidden, weights->rms_att, normed, 1e-6f);

    /* ---- Step 3: QKV projections ---- */
    Tensor* Q = scratch_tensor(scr, 2, T, H, 0, 0);
    Tensor* K = scratch_tensor(scr, 2, T, kv_dim, 0, 0);
    Tensor* V = scratch_tensor(scr, 2, T, kv_dim, 0, 0);

    gemm_f32(normed, weights->wq, Q);   /* (T, H) × (H, H) = (T, H) */
    gemm_f32(normed, weights->wk, K);   /* (T, H) × (H, kv_dim) = (T, kv_dim) */
    gemm_f32(normed, weights->wv, V);   /* (T, H) × (H, kv_dim) = (T, kv_dim) */

    /* ---- Step 4: RoPE on Q and K ---- */
    for (int t = 0; t < T; t++) {
        float* q_t = (float*)Q->data + t * H;
        float* k_t = (float*)K->data + t * kv_dim;

        Tensor q_view = make_view(q_t, 2, n_heads, head_dim, 0);
        Tensor k_view = make_view(k_t, 2, n_kv_heads, head_dim, 0);

        rope(&q_view, &k_view, pos + t, head_dim);
    }

    /* ---- Step 5: Append K, V to cache ---- */
    for (int t = 0; t < T; t++) {
        float* k_t = (float*)K->data + t * kv_dim;
        float* v_t = (float*)V->data + t * kv_dim;

        Tensor k_tok = make_view(k_t, 2, n_kv_heads, head_dim, 0);
        Tensor v_tok = make_view(v_t, 2, n_kv_heads, head_dim, 0);

        kv_cache_append(kv, layer, &k_tok, &v_tok, pos + t);
    }

    /* ---- Step 6: Get cached K, V ---- */
    int seq_len = pos + T;  /* total sequence length so far */
    Tensor* K_cached = kv_cache_get_k(kv, layer, seq_len);  /* (n_kv_heads, seq_len, head_dim) */
    Tensor* V_cached = kv_cache_get_v(kv, layer, seq_len);  /* (n_kv_heads, seq_len, head_dim) */

    /* ---- Step 7-10: Multi-head attention ---- */
    Tensor* attn_output = scratch_tensor(scr, 2, T, H, 0, 0);

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;  /* GQA: which KV head this Q head maps to */

        /* Q for this head: (T, head_dim) */
        float* q_h = (float*)Q->data + h * head_dim;  /* stride = H per row */
        /* K for this head from cache: (seq_len, head_dim) */
        float* k_h = (float*)K_cached->data + kv_h * K_cached->stride[0] * (int)sizeof(float) / (int)sizeof(float);
        /* Actually, K_cached stride is in elements, so: */
        k_h = (float*)K_cached->data + (size_t)kv_h * K_cached->stride[0];

        /* V for this head from cache: (seq_len, head_dim) */
        float* v_h = (float*)V_cached->data + (size_t)kv_h * V_cached->stride[0];

        /* Step 7: scores = Q × Kᵀ / sqrt(d_k)
         * For each query token t, compute scores against all seq_len positions */
        float* scores = (float*)scratch_alloc(scr, T * seq_len * sizeof(float), 64);

        for (int tq = 0; tq < T; tq++) {
            float* q_row = q_h + (size_t)tq * H;  /* Q is (T, H), stride H between rows */
            for (int s = 0; s < seq_len; s++) {
                float* k_row = k_h + (size_t)s * head_dim;  /* K is (seq_len, head_dim) */
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += q_row[d] * k_row[d];
                }
                scores[tq * seq_len + s] = dot * scale;
            }
        }

        /* Step 8: Causal mask — mask out future positions */
        for (int tq = 0; tq < T; tq++) {
            int query_pos = pos + tq;
            for (int s = query_pos + 1; s < seq_len; s++) {
                scores[tq * seq_len + s] = -INFINITY;
            }
        }

        /* Step 9: Softmax over each query row */
        for (int tq = 0; tq < T; tq++) {
            Tensor score_row = make_view(scores + tq * seq_len, 1, seq_len, 0, 0);
            softmax_inplace(&score_row, seq_len);
        }

        /* Step 10: context = scores × V — (T, seq_len) × (seq_len, head_dim) = (T, head_dim) */
        for (int tq = 0; tq < T; tq++) {
            float* out_h = (float*)attn_output->data + tq * H + h * head_dim;
            float* score_row = scores + tq * seq_len;

            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int s = 0; s < seq_len; s++) {
                    sum += score_row[s] * v_h[s * head_dim + d];
                }
                out_h[d] = sum;
            }
        }
    }

    /* ---- Step 11: Output projection ---- */
    Tensor* attn_proj = scratch_tensor(scr, 2, T, H, 0, 0);
    gemm_f32(attn_output, weights->wo, attn_proj);  /* (T, H) × (H, H) = (T, H) */

    /* ---- Step 12: Residual add ---- */
    residual_add(attn_proj, residual);

    /* ---- Step 13: Save residual ---- */
    memcpy(residual->data, attn_proj->data, T * H * sizeof(float));

    /* ---- Step 14: Pre-MLP RMSNorm ---- */
    Tensor* normed2 = scratch_tensor(scr, 2, T, H, 0, 0);
    rmsnorm(attn_proj, weights->rms_ffn, normed2, 1e-6f);

    /* ---- Step 15-19: SwiGLU MLP ---- */
    int ff_dim = cfg->ff_dim;
    Tensor* gate = scratch_tensor(scr, 2, T, ff_dim, 0, 0);
    Tensor* up   = scratch_tensor(scr, 2, T, ff_dim, 0, 0);

    /* Step 15: gate = x × W_gate */
    gemm_f32(normed2, weights->w_gate, gate);  /* (T, H) × (H, ff) = (T, ff) */

    /* Step 16: up = x × W_up */
    gemm_f32(normed2, weights->w_up, up);      /* (T, H) × (H, ff) = (T, ff) */

    /* Step 17: SiLU(gate) */
    silu_inplace(gate);

    /* Step 18: gate = gate ⊙ up */
    elemwise_mul(gate, up);

    /* Step 19: down = gate × W_down */
    Tensor* down = scratch_tensor(scr, 2, T, H, 0, 0);
    gemm_f32(gate, weights->w_down, down);     /* (T, ff) × (ff, H) = (T, H) */

    /* ---- Step 20: Residual add ---- */
    residual_add(down, residual);

    /* Write back to hidden */
    memcpy(hidden->data, down->data, T * H * sizeof(float));
}

/* ---------- Forward Pass ---------- */

Tensor* forward(ModelWeights* model, KVCache* kv, Scratch* scr,
                int* token_ids, int n_tokens, int pos) {
    ModelConfig* cfg = &model->config;
    int H = cfg->hidden_dim;

    /* Reset scratch for this forward pass */
    scratch_reset(scr);

    /* Step 1: Embedding lookup */
    Tensor* hidden = scratch_tensor(scr, 2, n_tokens, H, 0, 0);
    embedding_lookup(model->embedding, token_ids, n_tokens, hidden);

    /* Step 2: Run all transformer layers */
    for (int l = 0; l < cfg->n_layers; l++) {
        transformer_layer(hidden, &model->layers[l], kv, l, pos, scr, cfg);
    }

    /* Step 3: Final RMSNorm */
    Tensor* normed = scratch_tensor(scr, 2, n_tokens, H, 0, 0);
    rmsnorm(hidden, model->rms_final, normed, 1e-6f);

    /* Step 4: logits = normed[-1] × lm_head
     * We only need the last token's logits for generation */
    float* last_hidden = (float*)normed->data + (n_tokens - 1) * H;
    Tensor last_view = make_view(last_hidden, 2, 1, H, 0);

    Tensor* logits = scratch_tensor(scr, 2, 1, cfg->vocab_size, 0, 0);
    gemm_f32(&last_view, model->lm_head, logits);  /* (1, H) × (H, vocab_size) = (1, V) */

    return logits;
}
