/*============================================================================
 * Engine Stubs — Animesh replaces these
 *============================================================================*/

#include "engine.h"
#include "memory.h"
#include "kernels.h"
#include "tokenizer.h"
#include <stdlib.h>
#include <stdio.h>

ModelWeights* load_model(const char* path, Arena* arena) {
    (void)path; (void)arena;
    fprintf(stderr, "[STUB] load_model: not implemented\n");
    return NULL;
}

void free_model(ModelWeights* model) {
    (void)model;
}

void transformer_layer(Tensor* hidden, LayerWeights* weights, KVCache* kv,
                       int layer, int pos, Scratch* scr, ModelConfig* cfg) {
    (void)hidden; (void)weights; (void)kv;
    (void)layer; (void)pos; (void)scr; (void)cfg;
    fprintf(stderr, "[STUB] transformer_layer: not implemented\n");
}

Tensor* forward(ModelWeights* model, KVCache* kv, Scratch* scr,
                int* token_ids, int n_tokens, int pos) {
    (void)model; (void)kv; (void)scr;
    (void)token_ids; (void)n_tokens; (void)pos;
    fprintf(stderr, "[STUB] forward: not implemented\n");
    return NULL;
}

void generate(const char* model_path, const char* prompt, int max_tokens,
              float temperature, int top_k, float top_p) {
    (void)model_path; (void)prompt; (void)max_tokens;
    (void)temperature; (void)top_k; (void)top_p;
    fprintf(stderr, "[STUB] generate: not implemented\n");
}
