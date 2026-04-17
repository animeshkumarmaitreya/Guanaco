#include "gpu_prefill.h"

int gpu_prefill(const Backend* backend,
                ModelWeights* model,
                KVCache* kv,
                Scratch* scr,
                const int* token_ids,
                int n_tokens,
                int pos) {
    (void)backend;
    (void)model;
    (void)kv;
    (void)scr;
    (void)token_ids;
    (void)n_tokens;
    (void)pos;

    /* Stub: Phase A implementation will live here. */
    return -1;
}
