/*============================================================================
 * Memory & KV Cache Stubs — Person B replaces these
 *
 * Arena/Scratch stubs use plain malloc. KV Cache stub is minimal but
 * functional enough for integration testing.
 *============================================================================*/

#define _DEFAULT_SOURCE  /* for posix_memalign */

#include "memory.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- Arena stub (wraps malloc) ---- */

struct Arena {
    void*  base;
    size_t capacity;
    size_t used;
};

Arena* arena_create(size_t size) {
    Arena* a = (Arena*)malloc(sizeof(Arena));
    if (!a) return NULL;
    /* Align base to 64 bytes so bump-alloc can satisfy alignment requests */
    if (posix_memalign(&a->base, 64, size) != 0) { free(a); return NULL; }
    a->capacity = size;
    a->used = 0;
    return a;
}

void* arena_alloc(Arena* arena, size_t size, size_t alignment) {
    /* Align the current offset */
    size_t offset = arena->used;
    size_t aligned = (offset + alignment - 1) & ~(alignment - 1);
    if (aligned + size > arena->capacity) return NULL;
    arena->used = aligned + size;
    return (char*)arena->base + aligned;
}

void arena_reset(Arena* arena) {
    arena->used = 0;
}

void arena_destroy(Arena* arena) {
    if (arena) {
        free(arena->base);
        free(arena);
    }
}

/* ---- Scratch stub (identical to arena) ---- */

struct Scratch {
    void*  base;
    size_t capacity;
    size_t used;
};

Scratch* scratch_create(size_t size) {
    Scratch* s = (Scratch*)malloc(sizeof(Scratch));
    if (!s) return NULL;
    s->base = malloc(size);
    if (!s->base) { free(s); return NULL; }
    s->capacity = size;
    s->used = 0;
    return s;
}

void* scratch_alloc(Scratch* scr, size_t size, size_t alignment) {
    size_t offset = scr->used;
    size_t aligned = (offset + alignment - 1) & ~(alignment - 1);
    if (aligned + size > scr->capacity) return NULL;
    scr->used = aligned + size;
    return (char*)scr->base + aligned;
}

void scratch_reset(Scratch* scr) {
    scr->used = 0;
}

void scratch_destroy(Scratch* scr) {
    if (scr) {
        free(scr->base);
        free(scr);
    }
}

/* ---- KV Cache stub (minimal) ---- */

struct KVCache {
    float*       k_data;    /* all K data: [n_layers][n_kv_heads][max_seq][head_dim] */
    float*       v_data;    /* all V data: same layout */
    ModelConfig  cfg;
    int          max_seq;
    Tensor       k_view;    /* reusable tensor view */
    Tensor       v_view;
};

KVCache* kv_cache_create(Arena* arena, ModelConfig* cfg, int max_seq_len) {
    KVCache* kv = (KVCache*)malloc(sizeof(KVCache));
    if (!kv) return NULL;

    kv->cfg = *cfg;
    kv->max_seq = max_seq_len;

    size_t per_kv = (size_t)cfg->n_layers * cfg->n_kv_heads * max_seq_len * cfg->head_dim;
    kv->k_data = (float*)calloc(per_kv, sizeof(float));
    kv->v_data = (float*)calloc(per_kv, sizeof(float));

    if (!kv->k_data || !kv->v_data) {
        free(kv->k_data);
        free(kv->v_data);
        free(kv);
        return NULL;
    }

    memset(&kv->k_view, 0, sizeof(Tensor));
    memset(&kv->v_view, 0, sizeof(Tensor));
    (void)arena; /* stub ignores arena, uses malloc */
    return kv;
}

int kv_cache_append(KVCache* kv, int layer, const Tensor* k, const Tensor* v, int pos) {
    if (pos >= kv->max_seq) return -1;

    int n_kv  = kv->cfg.n_kv_heads;
    int d     = kv->cfg.head_dim;
    int seq   = kv->max_seq;
    const float* kd = (const float*)k->data;
    const float* vd = (const float*)v->data;

    for (int h = 0; h < n_kv; h++) {
        size_t offset = ((size_t)layer * n_kv + h) * seq * d + (size_t)pos * d;
        memcpy(kv->k_data + offset, kd + h * d, d * sizeof(float));
        memcpy(kv->v_data + offset, vd + h * d, d * sizeof(float));
    }
    return 0;
}

Tensor* kv_cache_get_k(KVCache* kv, int layer, int up_to_pos) {
    int n_kv = kv->cfg.n_kv_heads;
    int d    = kv->cfg.head_dim;
    int seq  = kv->max_seq;

    kv->k_view.data = kv->k_data + (size_t)layer * n_kv * seq * d;
    kv->k_view.ndim = 3;
    kv->k_view.shape[0] = n_kv;
    kv->k_view.shape[1] = up_to_pos;
    kv->k_view.shape[2] = d;
    kv->k_view.stride[0] = seq * d;
    kv->k_view.stride[1] = d;
    kv->k_view.stride[2] = 1;
    kv->k_view.dtype = DTYPE_F32;
    return &kv->k_view;
}

Tensor* kv_cache_get_v(KVCache* kv, int layer, int up_to_pos) {
    int n_kv = kv->cfg.n_kv_heads;
    int d    = kv->cfg.head_dim;
    int seq  = kv->max_seq;

    kv->v_view.data = kv->v_data + (size_t)layer * n_kv * seq * d;
    kv->v_view.ndim = 3;
    kv->v_view.shape[0] = n_kv;
    kv->v_view.shape[1] = up_to_pos;
    kv->v_view.shape[2] = d;
    kv->v_view.stride[0] = seq * d;
    kv->v_view.stride[1] = d;
    kv->v_view.stride[2] = 1;
    kv->v_view.dtype = DTYPE_F32;
    return &kv->v_view;
}

void kv_cache_destroy(KVCache* kv) {
    if (kv) {
        free(kv->k_data);
        free(kv->v_data);
        free(kv);
    }
}
