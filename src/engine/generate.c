/*============================================================================
 * Generate Loop
 *
 * Orchestrates the full inference pipeline:
 *   tokenize → prefill → decode loop → detokenize → print
 *
 * Also provides timing output (TTFT, tok/s).
 *============================================================================*/

#define _POSIX_C_SOURCE 199309L  /* for clock_gettime, CLOCK_MONOTONIC */

#include "engine.h"
#include "backend.h"
#include "memory.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------- Timing helpers ---------- */

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static const char* backend_kind_str(BackendKind k) {
    switch (k) {
        case BACKEND_CPU:  return "cpu";
        case BACKEND_CUDA: return "cuda";
        case BACKEND_AUTO:
        default:           return "auto";
    }
}

/* ---------- Generate ---------- */

void generate(const char* model_path, const char* prompt, int max_tokens,
              float temperature, int top_k, float top_p,
              const BackendConfig* backend_cfg) {

    BackendConfig cfg_local = {0};
    if (backend_cfg) cfg_local = *backend_cfg;
    else {
        cfg_local.kind = BACKEND_CPU;
        cfg_local.threads = 1;
        cfg_local.device_id = 0;
    }

    /* Backend selection semantics:
     * - BACKEND_CUDA: required, error if unavailable
     * - BACKEND_CPU: forced
     * - (Caller can implement AUTO by attempting CUDA then falling back to CPU)
     */
    BackendKind requested_kind = cfg_local.kind;
    Backend* backend = backend_create(&cfg_local);
    if (!backend && cfg_local.kind == BACKEND_CUDA) {
        fprintf(stderr, "Error: CUDA backend requested but unavailable (build with USE_CUDA=1, and ensure a supported GPU/runtime)\n");
        return;
    }
    if (!backend) {
        /* Fallback to CPU for non-CUDA forced cases. */
        cfg_local.kind = BACKEND_CPU;
        backend = backend_create(&cfg_local);
        if (!backend) {
            fprintf(stderr, "Error: failed to create CPU backend\n");
            return;
        }
    }
    const KernelVTable* k = backend_kernels(backend);
    if (!k) {
        fprintf(stderr, "Error: backend returned NULL kernel vtable\n");
        backend_destroy(backend);
        return;
    }

    BackendKind selected = backend_kind(backend);
    if (requested_kind == BACKEND_AUTO) {
        printf("Selected backend: %s\n", backend_kind_str(selected));
    } else {
        printf("Backend: %s\n", backend_kind_str(selected));
    }

    printf("Loading model: %s\n", model_path);
    double t0 = time_ms();

    /* ---- Load model ---- */
    ModelWeights* model = load_model(model_path, NULL);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return;
    }

    double load_time = time_ms() - t0;
    printf("Model loaded in %.1f ms\n", load_time);

    ModelConfig* cfg = &model->config;
    const int eos_token_id = (cfg->eos_token_id > 0) ? cfg->eos_token_id : 2;

    /* ---- Create allocators ---- */
    /* Arena for KV cache — size calculation:
     * 2 (K+V) × n_layers × n_kv_heads × max_seq_len × head_dim × sizeof(float) */
    size_t kv_size = 2UL * cfg->n_layers * cfg->n_kv_heads * cfg->max_seq_len
                     * cfg->head_dim * sizeof(float);
    /* Add some headroom */
    size_t arena_size = kv_size + 64 * 1024 * 1024;  /* + 64 MiB overhead */
    Arena* arena = arena_create(arena_size);
    if (!arena) {
        fprintf(stderr, "Failed to create arena (%zu bytes)\n", arena_size);
        free_model(model);
        return;
    }

    /* Scratch for per-forward-pass temporaries — generous size */
    size_t scratch_size = 512 * 1024 * 1024;  /* 512 MiB for LLaMA 3.2 massive FF_DIMs */
    Scratch* scr = scratch_create(scratch_size);
    if (!scr) {
        fprintf(stderr, "Failed to create scratch (%zu bytes)\n", scratch_size);
        arena_destroy(arena);
        free_model(model);
        return;
    }

    /* ---- Create KV cache ---- */
    KVCache* kv = kv_cache_create(arena, cfg, cfg->max_seq_len);
    if (!kv) {
        fprintf(stderr, "Failed to create KV cache\n");
        scratch_destroy(scr);
        arena_destroy(arena);
        free_model(model);
        return;
    }

    Tokenizer* tok = tokenizer_create(cfg);
    if (!tok) {
        fprintf(stderr, "Failed to create tokenizer\n");
        scratch_destroy(scr);
        arena_destroy(arena);
        free_model(model);
        return;
    }

    int prompt_len = 0;
    int* prompt_tokens = tokenize(tok, prompt, &prompt_len);
    if (!prompt_tokens || prompt_len == 0) {
        fprintf(stderr, "Tokenization failed or empty prompt\n");
        tokenizer_destroy(tok);
        scratch_destroy(scr);
        arena_destroy(arena);
        free_model(model);
        return;
    }

    printf("Prompt: \"%s\" (%d tokens)\n", prompt, prompt_len);
    printf("Generating %d tokens...\n\n", max_tokens);

    /* ---- Prefill ---- */
    double t_prefill_start = time_ms();

    Tensor* logits = forward(model, kv, scr, prompt_tokens, prompt_len, 0, k);
    if (!logits) {
        fprintf(stderr, "Prefill forward pass failed\n");
        goto cleanup;
    }

    double ttft = time_ms() - t_prefill_start;

    /* Sample first token from prefill logits */
    int next_token;
    float* logit_data = (float*)logits->data;

    if (temperature <= 0.0f) {
        next_token = sample_greedy(logit_data, cfg->vocab_size);
    } else if (top_k > 0 && top_k < cfg->vocab_size) {
        next_token = sample_top_k(logit_data, cfg->vocab_size, top_k, temperature);
    } else if (top_p < 1.0f) {
        next_token = sample_top_p(logit_data, cfg->vocab_size, top_p, temperature);
    } else {
        next_token = sample_temperature(logit_data, cfg->vocab_size, temperature);
    }

    /* Print first generated token */
    const char* token_str = detokenize(tok, next_token);
    printf("%s", token_str);
    fflush(stdout);

    /* ---- Decode loop ---- */
    double t_decode_start = time_ms();
    int tokens_generated = 1;

    for (int i = 1; i < max_tokens; i++) {
        int pos = prompt_len + i - 1;

        /* Check sequence length limit */
        if (pos >= cfg->max_seq_len - 1) {
            printf("\n[max sequence length reached]\n");
            break;
        }

        /* Forward pass for single token */
        logits = forward(model, kv, scr, &next_token, 1, pos, k);
        if (!logits) {
            fprintf(stderr, "\nForward pass failed at token %d\n", i);
            break;
        }

        logit_data = (float*)logits->data;

        /* Sample next token */
        if (temperature <= 0.0f) {
            next_token = sample_greedy(logit_data, cfg->vocab_size);
        } else if (top_k > 0 && top_k < cfg->vocab_size) {
            next_token = sample_top_k(logit_data, cfg->vocab_size, top_k, temperature);
        } else if (top_p < 1.0f) {
            next_token = sample_top_p(logit_data, cfg->vocab_size, top_p, temperature);
        } else {
            next_token = sample_temperature(logit_data, cfg->vocab_size, temperature);
        }

        tokens_generated++;

        /* Check for EOS token (from GGUF; fallback preserves prior behavior) */
        if (next_token == eos_token_id) {
            printf("\n");
            break;
        }

        /* Print decoded token */
        token_str = detokenize(tok, next_token);
        printf("%s", token_str);
        fflush(stdout);
    }

    double decode_time = time_ms() - t_decode_start;

    /* ---- Print timing stats ---- */
    printf("\n\n--- Stats ---\n");
    printf("TTFT (time to first token): %.1f ms\n", ttft);
    printf("Tokens generated: %d\n", tokens_generated);
    if (decode_time > 0 && tokens_generated > 1) {
        double tok_per_sec = (tokens_generated - 1) / (decode_time / 1000.0);
        printf("Decode speed: %.1f tok/s (%.1f ms/tok)\n",
               tok_per_sec, decode_time / (tokens_generated - 1));
    }
    printf("Total time: %.1f ms\n", time_ms() - t0);

cleanup:
    free(prompt_tokens);
    tokenizer_destroy(tok);
    scratch_destroy(scr);
    arena_destroy(arena);
    free_model(model);
    backend_destroy(backend);
}
