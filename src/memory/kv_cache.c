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
#include <assert.h>

struct KVCache {
    float*       k_data;     /* all K: [n_layers][n_kv_heads][max_seq][head_dim] */
    float*       v_data;     /* all V: same layout */
    ModelConfig  cfg;
    int          max_seq;
    Tensor*      k_views;    /* per-layer tensor views for get_k [n_layers] */
    Tensor*      v_views;    /* per-layer tensor views for get_v [n_layers] */
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

    /* Allocate per-layer Tensor views from the arena */
    size_t views_bytes = (size_t)cfg->n_layers * sizeof(Tensor);
    kv->k_views = (Tensor*)arena_alloc(arena, views_bytes, 64);
    kv->v_views = (Tensor*)arena_alloc(arena, views_bytes, 64);
    if (kv->k_views == NULL || kv->v_views == NULL) {
        fprintf(stderr, "[kv] failed to allocate tensor views\n");
        return NULL;
    }
    memset(kv->k_views, 0, views_bytes);
    memset(kv->v_views, 0, views_bytes);

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

#ifdef USE_CUDA
    extern int cudaHostRegister(void *ptr, size_t size, unsigned int flags);
    cudaHostRegister(kv->k_data, bytes, 0);
    cudaHostRegister(kv->v_data, bytes, 0);
#endif

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

#ifndef NDEBUG
    assert(k->dtype == DTYPE_F32 && v->dtype == DTYPE_F32);
    assert(k->ndim == 2 && v->ndim == 2);
    assert(k->shape[0] == kv->cfg.n_kv_heads);
    assert(v->shape[0] == kv->cfg.n_kv_heads);
    assert(k->shape[1] == kv->cfg.head_dim);
    assert(v->shape[1] == kv->cfg.head_dim);
    assert(k->stride[0] == kv->cfg.head_dim && k->stride[1] == 1);
    assert(v->stride[0] == kv->cfg.head_dim && v->stride[1] == 1);
#endif

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
    if (layer < 0 || layer >= kv->cfg.n_layers) return NULL;

    int n_kv = kv->cfg.n_kv_heads;
    int d    = kv->cfg.head_dim;
    int seq  = kv->max_seq;

    Tensor* view = &kv->k_views[layer];
    view->data      = kv->k_data + (size_t)layer * n_kv * seq * d;
    view->ndim      = 3;
    view->shape[0]  = n_kv;
    view->shape[1]  = up_to_pos;
    view->shape[2]  = d;
    view->stride[0] = seq * d;   /* stride between heads */
    view->stride[1] = d;         /* stride between positions */
    view->stride[2] = 1;         /* stride between dims */
    view->dtype     = DTYPE_F32;

    return view;
}

Tensor* kv_cache_get_v(KVCache* kv, int layer, int up_to_pos) {
    if (kv == NULL) return NULL;
    if (layer < 0 || layer >= kv->cfg.n_layers) return NULL;

    int n_kv = kv->cfg.n_kv_heads;
    int d    = kv->cfg.head_dim;
    int seq  = kv->max_seq;

    Tensor* view = &kv->v_views[layer];
    view->data      = kv->v_data + (size_t)layer * n_kv * seq * d;
    view->ndim      = 3;
    view->shape[0]  = n_kv;
    view->shape[1]  = up_to_pos;
    view->shape[2]  = d;
    view->stride[0] = seq * d;
    view->stride[1] = d;
    view->stride[2] = 1;
    view->dtype     = DTYPE_F32;

    return view;
}

void kv_cache_destroy(KVCache* kv) {
    if (!kv) return;
#ifdef USE_CUDA
    extern int cudaHostUnregister(void *ptr);
    if (kv->k_data) cudaHostUnregister(kv->k_data);
    if (kv->v_data) cudaHostUnregister(kv->v_data);
#endif
    /* Memory is arena-managed — nothing to free individually.
     * This function exists for API completeness; the arena_destroy()
     * call at session end releases everything. */
}
