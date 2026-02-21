/*============================================================================
 * KV Cache — Person B's real implementation
 *
 * Head-major layout: K[layer][kv_head][token_pos][dim_within_head]
 * Backed by Arena memory (session lifetime).
 *
 * API matches memory.h:
 *   kv_cache_create  — allocate from arena
 *   kv_cache_append  — write K/V Tensors at a position for one layer
 *   kv_cache_get_k   — return Tensor view into K cache for a layer
 *   kv_cache_get_v   — return Tensor view into V cache for a layer
 *   kv_cache_destroy — no-op (arena owns the memory)
 *
 * TinyLlama sizing:
 *   2 × 22 layers × 4 kv_heads × 2048 max_seq × 64 head_dim × 4 bytes
 *   = 92,274,688 bytes (~88 MiB)
 *============================================================================*/

#include "memory.h"
#include <string.h>
#include <stdio.h>
#include <stdint.h>

struct KVCache {
    float*       k_data;     /* all K: [n_layers][n_kv_heads][max_seq][head_dim] */
    float*       v_data;     /* all V: same layout */
    ModelConfig  cfg;
    int          max_seq;
    Tensor       k_view;     /* reusable tensor view for get_k */
    Tensor       v_view;     /* reusable tensor view for get_v */
};

KVCache* kv_cache_create(Arena* arena, ModelConfig* cfg, int max_seq_len) {
    if (arena == NULL || cfg == NULL) return NULL;

    /* Allocate the KVCache struct from the arena */
    KVCache* kv = (KVCache*)arena_alloc(arena, sizeof(KVCache), 64);
    if (kv == NULL) {
        fprintf(stderr, "[kv] failed to allocate KVCache struct\n");
        return NULL;
    }

    kv->cfg     = *cfg;
    kv->max_seq = max_seq_len;
    memset(&kv->k_view, 0, sizeof(Tensor));
    memset(&kv->v_view, 0, sizeof(Tensor));

    /* Total floats per cache (K or V):
     *   n_layers × n_kv_heads × max_seq_len × head_dim */
    size_t total = (size_t)cfg->n_layers * cfg->n_kv_heads
                   * max_seq_len * cfg->head_dim;
    size_t bytes = total * sizeof(float);

    kv->k_data = (float*)arena_alloc(arena, bytes, 64);
    kv->v_data = (float*)arena_alloc(arena, bytes, 64);

    if (kv->k_data == NULL || kv->v_data == NULL) {
        fprintf(stderr, "[kv] arena_alloc failed: need 2 × %zu bytes (%.1f MiB)\n",
                bytes, (2.0 * bytes) / (1024.0 * 1024.0));
        return NULL;
    }

    /* Zero-initialize for clean state */
    memset(kv->k_data, 0, bytes);
    memset(kv->v_data, 0, bytes);

    return kv;
}

/*
 * Offset into the flat cache array:
 *   ((layer * n_kv_heads + head) * max_seq + pos) * head_dim
 */
static inline size_t kv_offset(const KVCache* kv, int layer, int head, int pos) {
    return ((size_t)((size_t)layer * kv->cfg.n_kv_heads + head)
            * kv->max_seq + pos) * kv->cfg.head_dim;
}

int kv_cache_append(KVCache* kv, int layer, const Tensor* k, const Tensor* v, int pos) {
    if (kv == NULL || k == NULL || v == NULL) return -1;
    if (pos < 0 || pos >= kv->max_seq) return -1;
    if (layer < 0 || layer >= kv->cfg.n_layers) return -1;

    int n_kv = kv->cfg.n_kv_heads;
    int d    = kv->cfg.head_dim;
    const float* kd = (const float*)k->data;
    const float* vd = (const float*)v->data;

    /* k and v are shaped (n_kv_heads, head_dim).
     * Scatter each head's slice into the correct cache location. */
    for (int h = 0; h < n_kv; h++) {
        size_t off = kv_offset(kv, layer, h, pos);
        memcpy(kv->k_data + off, kd + h * d, d * sizeof(float));
        memcpy(kv->v_data + off, vd + h * d, d * sizeof(float));
    }

    return 0;
}

Tensor* kv_cache_get_k(KVCache* kv, int layer, int up_to_pos) {
    if (kv == NULL) return NULL;

    int n_kv = kv->cfg.n_kv_heads;
    int d    = kv->cfg.head_dim;
    int seq  = kv->max_seq;

    /* Point into the start of this layer's K data */
    kv->k_view.data      = kv->k_data + (size_t)layer * n_kv * seq * d;
    kv->k_view.ndim      = 3;
    kv->k_view.shape[0]  = n_kv;
    kv->k_view.shape[1]  = up_to_pos;
    kv->k_view.shape[2]  = d;
    kv->k_view.stride[0] = seq * d;   /* stride between heads */
    kv->k_view.stride[1] = d;         /* stride between positions */
    kv->k_view.stride[2] = 1;         /* stride between dims */
    kv->k_view.dtype     = DTYPE_F32;

    return &kv->k_view;
}

Tensor* kv_cache_get_v(KVCache* kv, int layer, int up_to_pos) {
    if (kv == NULL) return NULL;

    int n_kv = kv->cfg.n_kv_heads;
    int d    = kv->cfg.head_dim;
    int seq  = kv->max_seq;

    kv->v_view.data      = kv->v_data + (size_t)layer * n_kv * seq * d;
    kv->v_view.ndim      = 3;
    kv->v_view.shape[0]  = n_kv;
    kv->v_view.shape[1]  = up_to_pos;
    kv->v_view.shape[2]  = d;
    kv->v_view.stride[0] = seq * d;
    kv->v_view.stride[1] = d;
    kv->v_view.stride[2] = 1;
    kv->v_view.dtype     = DTYPE_F32;

    return &kv->v_view;
}

void kv_cache_destroy(KVCache* kv) {
    /* Memory is arena-managed — nothing to free individually.
     * This function exists for API completeness; the arena_destroy()
     * call at session end releases everything. */
    (void)kv;
}
