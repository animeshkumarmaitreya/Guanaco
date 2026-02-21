#ifndef LLMRT_MEMORY_H
#define LLMRT_MEMORY_H

#include "types.h"

/*============================================================================
 * Memory & KV Cache API — Person B owns the implementation
 *
 * Three allocation tiers:
 *   Arena   — session lifetime (KV cache, token buffer)
 *   Scratch — per-forward-pass lifetime (activations, intermediates)
 *   KVCache — per-layer K/V storage, head-major layout
 *============================================================================*/

/* ---- Arena allocator ---- */

/* Create an arena of `size` bytes. Returns NULL on failure. */
Arena* arena_create(size_t size);

/* Bump-allocate `size` bytes from the arena, aligned to `alignment`.
 * Returns NULL if arena is full. alignment must be a power of 2. */
void* arena_alloc(Arena* arena, size_t size, size_t alignment);

/* Reset arena — all previous allocations are invalidated. O(1). */
void arena_reset(Arena* arena);

/* Destroy arena and free backing memory. */
void arena_destroy(Arena* arena);

/* ---- Scratch allocator ---- */

/* Create a scratch pool of `size` bytes. Returns NULL on failure. */
Scratch* scratch_create(size_t size);

/* Bump-allocate from scratch pool. Same semantics as arena_alloc. */
void* scratch_alloc(Scratch* scr, size_t size, size_t alignment);

/* Reset scratch pool. Called once per forward pass. O(1). */
void scratch_reset(Scratch* scr);

/* Destroy scratch pool. */
void scratch_destroy(Scratch* scr);

/* ---- KV Cache ---- */

/* Create KV cache for all layers. Uses arena for backing memory.
 * Layout: head-major — K[layer][kv_head][token][dim] 
 * Pre-allocates for max_seq_len tokens. */
KVCache* kv_cache_create(Arena* arena, ModelConfig* cfg, int max_seq_len);

/* Append new K and V vectors for one token at `pos` across one layer.
 * k: (n_kv_heads, head_dim), v: (n_kv_heads, head_dim) */
int kv_cache_append(KVCache* kv, int layer, const Tensor* k, const Tensor* v, int pos);

/* Get K cache for a layer, from token 0 up to `up_to_pos` (exclusive).
 * Returns a Tensor view: (n_kv_heads, up_to_pos, head_dim).
 * The returned tensor points into cache memory — do not free. */
Tensor* kv_cache_get_k(KVCache* kv, int layer, int up_to_pos);

/* Get V cache for a layer, same semantics as kv_cache_get_k. */
Tensor* kv_cache_get_v(KVCache* kv, int layer, int up_to_pos);

/* Destroy KV cache and free backing memory. */
void kv_cache_destroy(KVCache* kv);

#endif /* LLMRT_MEMORY_H */
