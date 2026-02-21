# Person C — Assumptions & Design Decisions

## Arena & Scratch Allocators

- **mmap-backed**: Both arena and scratch use `MAP_PRIVATE | MAP_ANONYMOUS` mmap. This avoids `malloc`/`free` on the hot path and gives the OS a clean region to reclaim.
- **Logical vs physical capacity**: mmap rounds up to page size (4096). The allocators enforce the *user-requested* capacity, not the mmap'd size, so overflow detection works at the exact boundary the caller specified.
- **64-byte alignment**: All allocations honor the caller's `alignment` parameter (typically 64 for AVX-512/cache-line). This matches the invariant in the research doc (Section 1.5 #6).
- **Scratch alignment parameter**: The header declares `scratch_alloc(Scratch*, size_t, size_t alignment)` (3 args). The team_plan pseudocode showed 2 args. I followed the header since that's the compiled contract.

## KV Cache

- **Layout B (Head-Major)**: `K[layer][kv_head][token_pos][dim]` as recommended in the research doc (Section 2.2). This gives contiguous reads for Q×Kᵀ per head.
- **Arena-backed**: The KVCache struct and its K/V data blocks are all allocated from the supplied Arena. `kv_cache_destroy()` is a no-op since memory lifetime is tied to the Arena.
- **FP32 only**: KV cache stores `float` (FP32). Quantized KV (FP16/Q8) is not implemented — this is consistent with the research doc's activation precision (FP32 for activations/intermediates).
- **Tensor views share internal buffers**: `kv_cache_get_k` / `kv_cache_get_v` return pointers into the single `k_view` / `v_view` field. This means two successive calls to `kv_cache_get_k` for different layers overwrite the previous view's metadata. The caller must consume or copy the Tensor before calling again for a different layer. The forward pass in `engine.c` processes one layer at a time, so this is safe.
- **seq_len tracking**: `kv_cache_append` tracks a high-water mark `seq_len`. This is a convenience — the forward pass currently passes `pos` explicitly and the engine controls sequencing.
- **No sliding window**: When `pos >= max_seq`, append returns -1. Sliding window / ring-buffer eviction is not implemented (marked as future work in the research doc Section 1.2.5).
