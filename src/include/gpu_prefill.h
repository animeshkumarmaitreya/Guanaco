#ifndef LLMRT_GPU_PREFILL_H
#define LLMRT_GPU_PREFILL_H

#include "types.h"
#include "backend.h"

/*============================================================================
 * GPU prefill hook (MUST Phase A) — scaffolding
 *
 * Prefill (n_tokens > 1) runs on GPU when backend is CUDA; decode remains CPU.
 *============================================================================*/

/* Returns 0 on success, non-zero on failure/unavailable. */
int gpu_prefill(const Backend* backend,
                ModelWeights* model,
                KVCache* kv,
                Scratch* scr,
                const int* token_ids,
                int n_tokens,
                int pos);

#endif /* LLMRT_GPU_PREFILL_H */
