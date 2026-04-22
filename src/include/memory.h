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

/* ---- Raw pointer accessors (for GPU fused layers / session I/O) ---- */
float* kv_cache_raw_k(KVCache* kv, int layer);
float* kv_cache_raw_v(KVCache* kv, int layer);
int    kv_cache_head_stride(KVCache* kv);

/* ---- Session Persistence (runs OUTSIDE decode loop) ---- */

/* Save full KV cache state + token history to a binary .lctx file.
 * Called after the decode loop finishes. Zero hot-path cost. */
int kv_cache_save(KVCache* kv, const ModelConfig* cfg, int max_seq,
                  int current_pos, const int* tokens, int n_tokens,
                  const char* path);

/* Load KV cache state from a .lctx file, restoring position and tokens.
 * Called before prefill — if successful, prefill is skipped entirely.
 * Returns 0 on success, -1 on failure (file missing = -1, caller continues normally). */
int kv_cache_load(KVCache* kv, const ModelConfig* cfg, int max_seq,
                  int* out_pos, int** out_tokens, int* out_n_tokens,
                  const char* path);

/* ---- Prompt Cache (frozen prefix, runs OUTSIDE decode loop) ---- */

/* Save the KV cache for the first prefix_len positions (system prompt).
 * Subsequent runs can load this to skip prefilling the system prompt. */
int prompt_cache_save(KVCache* kv, const ModelConfig* cfg, int max_seq,
                      int prefix_len, const char* path);

/* Load a previously saved prompt cache. Sets *out_prefix_len to the
 * number of valid positions, so the caller starts prefill from there. */
int prompt_cache_load(KVCache* kv, const ModelConfig* cfg, int max_seq,
                      int* out_prefix_len, const char* path);

#endif /* LLMRT_MEMORY_H */
