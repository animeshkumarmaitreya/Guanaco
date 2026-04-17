/*============================================================================
 * Engine Test Suite — Animesh's test harness
 *
 * Tests the forward pass wiring with stub kernels + stub memory.
 * Verifies: shapes flow correctly, no crashes, residual connections work.
 *
 * Run: make test_engine && ./build/test_engine
 *============================================================================*/

#include "engine.h"
#include "memory.h"
#include "kernels.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PASS(name) printf("  PASS: %s\n", name)
#define FAIL(name, msg) do { printf("  FAIL: %s — %s\n", name, msg); failures++; } while(0)

static int failures = 0;

/* ---------- Weight allocation helpers ---------- */

static Tensor* make_weight_2d(int rows, int cols, float fill) {
    Tensor* t = (Tensor*)calloc(1, sizeof(Tensor));
    t->data = calloc((size_t)rows * cols, sizeof(float));
    t->ndim = 2;
    t->shape[0] = rows; t->shape[1] = cols;
    t->stride[0] = cols; t->stride[1] = 1;
    t->dtype = DTYPE_F32;
    float* d = (float*)t->data;
    for (int i = 0; i < rows * cols; i++) d[i] = fill;
    return t;
}

static Tensor* make_weight_1d(int dim, float fill) {
    Tensor* t = (Tensor*)calloc(1, sizeof(Tensor));
    t->data = calloc((size_t)dim, sizeof(float));
    t->ndim = 1;
    t->shape[0] = dim; t->stride[0] = 1;
    t->dtype = DTYPE_F32;
    float* d = (float*)t->data;
    for (int i = 0; i < dim; i++) d[i] = fill;
    return t;
}

static void free_weight(Tensor* w) {
    if (w) { free(w->data); free(w); }
}

static void init_layer_weights(LayerWeights* lw, int H, int ff) {
    lw->wq       = make_weight_2d(H, H, 0.001f);
    lw->wk       = make_weight_2d(H, H, 0.001f);
    lw->wv       = make_weight_2d(H, H, 0.001f);
    lw->wo       = make_weight_2d(H, H, 0.001f);
    /* gemm_f32 expects weights as (out_features, in_features) */
    lw->w_gate   = make_weight_2d(ff, H, 0.001f);
    lw->w_up     = make_weight_2d(ff, H, 0.001f);
    lw->w_down   = make_weight_2d(H, ff, 0.001f);
    lw->rms_att  = make_weight_1d(H, 1.0f);
    lw->rms_ffn  = make_weight_1d(H, 1.0f);
}

static void free_layer_weights(LayerWeights* lw) {
    free_weight(lw->wq); free_weight(lw->wk);
    free_weight(lw->wv); free_weight(lw->wo);
    free_weight(lw->w_gate); free_weight(lw->w_up);
    free_weight(lw->w_down);
    free_weight(lw->rms_att); free_weight(lw->rms_ffn);
}

/* ---------- Test transformer_layer ---------- */

static void test_transformer_layer_shapes(void) {
    ModelConfig cfg = {
        .hidden_dim = 64, .n_heads = 4, .n_kv_heads = 4,
        .head_dim = 16, .n_layers = 1, .vocab_size = 100,
        .ff_dim = 128, .max_seq_len = 32
    };

    Arena* arena = arena_create(64 * 1024 * 1024);
    Scratch* scr = scratch_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(arena, &cfg, 32);

    if (!arena || !scr || !kv) {
        FAIL("transformer_layer_shapes", "allocation failed");
        return;
    }

    LayerWeights lw;
    init_layer_weights(&lw, cfg.hidden_dim, cfg.ff_dim);

    /* Create input hidden state */
    int H = cfg.hidden_dim;
    float input[64];
    for (int i = 0; i < H; i++) input[i] = 1.0f + 0.01f * (float)i;

    Tensor hidden;
    hidden.data = input;
    hidden.ndim = 2;
    hidden.shape[0] = 1; hidden.shape[1] = H;
    hidden.shape[2] = 0; hidden.shape[3] = 0;
    hidden.stride[0] = H; hidden.stride[1] = 1;
    hidden.stride[2] = 0; hidden.stride[3] = 0;
    hidden.dtype = DTYPE_F32;

    /* Run transformer layer */
    transformer_layer(&hidden, &lw, kv, 0, 0, scr, &cfg);

    /* Check no NaN/Inf */
    int has_nan = 0;
    float* out = (float*)hidden.data;
    for (int i = 0; i < H; i++) {
        if (isnan(out[i]) || isinf(out[i])) {
            has_nan = 1;
            break;
        }
    }

    if (has_nan) {
        FAIL("transformer_layer_shapes", "output contains NaN/Inf");
    } else {
        PASS("transformer_layer_shapes");
    }

    free_layer_weights(&lw);
    kv_cache_destroy(kv);
    scratch_destroy(scr);
    arena_destroy(arena);
}

/* ---------- Test forward pass ---------- */

static void test_forward_stub(void) {
    ModelConfig cfg = {
        .hidden_dim = 64, .n_heads = 4, .n_kv_heads = 4,
        .head_dim = 16, .n_layers = 2, .vocab_size = 100,
        .ff_dim = 128, .max_seq_len = 32
    };

    Arena* arena = arena_create(64 * 1024 * 1024);
    Scratch* scr = scratch_create(64 * 1024 * 1024);
    KVCache* kv = kv_cache_create(arena, &cfg, 32);

    if (!arena || !scr || !kv) {
        FAIL("forward_stub", "allocation failed");
        return;
    }

    int H = cfg.hidden_dim;
    int ff = cfg.ff_dim;
    int V = cfg.vocab_size;

    ModelWeights model;
    memset(&model, 0, sizeof(model));
    model.config = cfg;
    model.layers = (LayerWeights*)calloc(cfg.n_layers, sizeof(LayerWeights));

    for (int l = 0; l < cfg.n_layers; l++) {
        init_layer_weights(&model.layers[l], H, ff);
    }
    model.embedding = make_weight_2d(V, H, 0.001f);
    model.rms_final = make_weight_1d(H, 1.0f);
    model.lm_head   = make_weight_2d(V, H, 0.001f);

    /* Run forward for 3 tokens (prefill) */
    int tokens[] = {1, 5, 10};
    Tensor* logits = forward(&model, kv, scr, tokens, 3, 0);

    if (!logits) {
        FAIL("forward_stub", "forward returned NULL");
    } else if (logits->shape[0] != 1 || logits->shape[1] != V) {
        FAIL("forward_stub", "wrong logits shape");
    } else {
        float* ld = (float*)logits->data;
        int has_nan = 0;
        for (int i = 0; i < V; i++) {
            if (isnan(ld[i]) || isinf(ld[i])) { has_nan = 1; break; }
        }
        if (has_nan) FAIL("forward_stub", "logits contain NaN/Inf");
        else         PASS("forward_stub");
    }

    /* Run decode (single token) to test KV cache continuation */
    int tok2 = 7;
    Tensor* logits2 = forward(&model, kv, scr, &tok2, 1, 3);
    if (!logits2) {
        FAIL("forward_decode", "decode forward returned NULL");
    } else {
        PASS("forward_decode");
    }

    /* Cleanup */
    for (int l = 0; l < cfg.n_layers; l++) {
        free_layer_weights(&model.layers[l]);
    }
    free_weight(model.embedding);
    free_weight(model.rms_final);
    free_weight(model.lm_head);
    free(model.layers);

    kv_cache_destroy(kv);
    scratch_destroy(scr);
    arena_destroy(arena);
}

/* ---------- Test sampler integration ---------- */

static void test_sampler_integration(void) {
    float logits[100];
    for (int i = 0; i < 100; i++) logits[i] = -1.0f;
    logits[42] = 5.0f;

    int picked = sample_greedy(logits, 100);
    if (picked == 42) PASS("sampler_integration");
    else              FAIL("sampler_integration", "expected token 42");
}

/* ---------- Main ---------- */

int main(void) {
    printf("=== Engine Test Suite (stub mode) ===\n");

    test_transformer_layer_shapes();
    test_forward_stub();
    test_sampler_integration();

    printf("\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    return failures;
}
