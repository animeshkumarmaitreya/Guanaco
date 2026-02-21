/*============================================================================
 * KV Cache — Person C
 *
 * Per-layer Key and Value storage for autoregressive decoding.
 *
 * Memory layout: Head-Major (Layout B from research doc)
 *   K[layer][kv_head][token_pos][dim_within_head]
 *   V[layer][kv_head][token_pos][dim_within_head]
 *
 * Each head's data for a layer is contiguous in memory, which gives perfect
 * sequential access during Q × Kᵀ and attn × V (the hot-path reads).
 *
 * Offset formula:
 *   offset(layer, head, pos, dim)
 *       = ((layer * n_kv_heads + head) * max_seq + pos) * head_dim + dim
 *
 * Backing memory is allocated from the supplied Arena so its lifetime is
 * tied to the session and there is zero malloc on the hot path.
 *============================================================================*/

#include "memory.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---------- Internal struct ---------- */

struct KVCache {
    float*      k_data;      /* all K data: layers × kv_heads × max_seq × head_dim  */
    float*      v_data;      /* all V data: same layout                              */
    int         n_layers;
    int         n_kv_heads;
    int         head_dim;
    int         max_seq;
    int         seq_len;     /* current number of tokens stored (append increments)  */

    /* Reusable Tensor views returned by kv_cache_get_k / kv_cache_get_v.
     * Two views so the caller can hold both k and v simultaneously.         */
    Tensor      k_view;
    Tensor      v_view;
};

/* ---------- Helpers ---------- */

/* Total floats in ONE of the K or V blocks. */
static inline size_t kv_total_floats(const KVCache* kv) {
    return (size_t)kv->n_layers * kv->n_kv_heads * kv->max_seq * kv->head_dim;
}

/* Flat index into k_data / v_data for (layer, head, pos, 0). */
static inline size_t kv_offset(const KVCache* kv, int layer, int head, int pos) {
    return ((size_t)((size_t)layer * kv->n_kv_heads + head) * kv->max_seq + pos)
           * kv->head_dim;
}

/* ---------- Public API ---------- */

KVCache* kv_cache_create(Arena* arena, ModelConfig* cfg, int max_seq_len) {
    if (!arena || !cfg || max_seq_len <= 0) return NULL;

    /* Allocate the KVCache struct itself from the arena. */
    KVCache* kv = (KVCache*)arena_alloc(arena, sizeof(KVCache), 64);
    if (!kv) return NULL;

    kv->n_layers   = cfg->n_layers;
    kv->n_kv_heads = cfg->n_kv_heads;
    kv->head_dim   = cfg->head_dim;
    kv->max_seq    = max_seq_len;
    kv->seq_len    = 0;

    size_t num_floats = (size_t)cfg->n_layers * cfg->n_kv_heads
                        * max_seq_len * cfg->head_dim;
    size_t bytes = num_floats * sizeof(float);

    kv->k_data = (float*)arena_alloc(arena, bytes, 64);
    kv->v_data = (float*)arena_alloc(arena, bytes, 64);
    if (!kv->k_data || !kv->v_data) return NULL;

    /* Zero-initialize so stale reads return 0.0f. */
    memset(kv->k_data, 0, bytes);
    memset(kv->v_data, 0, bytes);

    memset(&kv->k_view, 0, sizeof(Tensor));
    memset(&kv->v_view, 0, sizeof(Tensor));

    return kv;
}

int kv_cache_append(KVCache* kv, int layer, const Tensor* k, const Tensor* v, int pos) {
    if (!kv || !k || !v) return -1;
    if (layer < 0 || layer >= kv->n_layers) return -1;
    if (pos < 0 || pos >= kv->max_seq)      return -1;

    const float* kd = (const float*)k->data;
    const float* vd = (const float*)v->data;

    int n_kv = kv->n_kv_heads;
    int d    = kv->head_dim;

    /* For each KV head, copy head_dim floats into the right slot.
     * Source layout: k->data is (n_kv_heads, head_dim) row-major. */
    for (int h = 0; h < n_kv; h++) {
        size_t dst_off = kv_offset(kv, layer, h, pos);
        memcpy(kv->k_data + dst_off, kd + (size_t)h * d, d * sizeof(float));
        memcpy(kv->v_data + dst_off, vd + (size_t)h * d, d * sizeof(float));
    }

    /* Track the high-water mark of tokens stored. */
    if (pos + 1 > kv->seq_len) {
        kv->seq_len = pos + 1;
    }

    return 0;
}

Tensor* kv_cache_get_k(KVCache* kv, int layer, int up_to_pos) {
    if (!kv) return NULL;
    if (layer < 0 || layer >= kv->n_layers) return NULL;
    if (up_to_pos <= 0 || up_to_pos > kv->max_seq) return NULL;

    int n_kv = kv->n_kv_heads;
    int d    = kv->head_dim;

    /* Return a 3-D view: (n_kv_heads, up_to_pos, head_dim).
     * stride[0] = max_seq * head_dim  (jump between heads)
     * stride[1] = head_dim            (jump between tokens)
     * stride[2] = 1                   (contiguous dim)            */
    kv->k_view.data      = kv->k_data + kv_offset(kv, layer, 0, 0);
    kv->k_view.ndim      = 3;
    kv->k_view.shape[0]  = n_kv;
    kv->k_view.shape[1]  = up_to_pos;
    kv->k_view.shape[2]  = d;
    kv->k_view.stride[0] = kv->max_seq * d;
    kv->k_view.stride[1] = d;
    kv->k_view.stride[2] = 1;
    kv->k_view.dtype     = DTYPE_F32;
    return &kv->k_view;
}

Tensor* kv_cache_get_v(KVCache* kv, int layer, int up_to_pos) {
    if (!kv) return NULL;
    if (layer < 0 || layer >= kv->n_layers) return NULL;
    if (up_to_pos <= 0 || up_to_pos > kv->max_seq) return NULL;

    int n_kv = kv->n_kv_heads;
    int d    = kv->head_dim;

    kv->v_view.data      = kv->v_data + kv_offset(kv, layer, 0, 0);
    kv->v_view.ndim      = 3;
    kv->v_view.shape[0]  = n_kv;
    kv->v_view.shape[1]  = up_to_pos;
    kv->v_view.shape[2]  = d;
    kv->v_view.stride[0] = kv->max_seq * d;
    kv->v_view.stride[1] = d;
    kv->v_view.stride[2] = 1;
    kv->v_view.dtype     = DTYPE_F32;
    return &kv->v_view;
}

void kv_cache_destroy(KVCache* kv) {
    /* KV cache memory is owned by the Arena — nothing to free here.
     * This function exists for the API contract. If the Arena is reset or
     * destroyed, all KV cache memory goes with it.                       */
    (void)kv;
}
