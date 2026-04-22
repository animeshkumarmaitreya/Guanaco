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
#include <unistd.h>

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
              float temperature, int top_k, float top_p, int ctx_len,
              const BackendConfig* backend_cfg,
              const char* session_path, const char* prompt_cache_path) {

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
    ModelWeights* model = load_model(model_path, cfg_local.n_gpu_layers, NULL);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return;
    }

    double load_time = time_ms() - t0;
    printf("Model loaded in %.1f ms", load_time);
    if (cfg_local.n_gpu_layers > 0) {
        printf(" (%d/%d layers on GPU)", cfg_local.n_gpu_layers, model->config.n_layers);
    }
    printf("\n");

    ModelConfig* cfg = &model->config;
    const int eos_token_id = (cfg->eos_token_id > 0) ? cfg->eos_token_id : 2;

    /* Clamp context length to prevent OOM */
    if (ctx_len <= 0) {
        ctx_len = 8192; /* Safe default */
    }
    if (cfg->max_seq_len > ctx_len) {
        cfg->max_seq_len = ctx_len;
    }

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

    /* ---- Session / Prompt Cache Load (BEFORE prefill — zero hot-path cost) ---- */
    int session_loaded = 0;
    int session_pos = 0;
    int* session_tokens = NULL;
    int session_n_tokens = 0;

    /* Try loading a full session first */
    if (session_path) {
        if (kv_cache_load(kv, cfg, cfg->max_seq_len,
                          &session_pos, &session_tokens, &session_n_tokens,
                          session_path) == 0) {
            session_loaded = 1;
            printf("Session restored (pos=%d) — skipping prefill\n", session_pos);
        }
    }

    /* Try loading a prompt cache (frozen prefix) if no session was loaded */
    int prompt_cache_prefix_len = 0;
    if (!session_loaded && prompt_cache_path) {
        if (prompt_cache_load(kv, cfg, cfg->max_seq_len,
                              &prompt_cache_prefix_len, prompt_cache_path) == 0) {
            printf("Prompt cache loaded (prefix_len=%d)\n", prompt_cache_prefix_len);
        }
    }

    printf("Prompt: \"%s\" (%d tokens)\n", prompt, prompt_len);
    printf("Generating %d tokens...\n\n", max_tokens);

    /* ---- Prefill ---- */
    double t_prefill_start = time_ms();
    int prefill_start_pos = 0;
    Tensor* logits = NULL;

    if (session_loaded) {
        /* Session restored — prefill the NEW prompt tokens starting at session_pos.
         * The KV cache already contains the full previous session context.
         * We need to inject the new user prompt into the KV cache so the model
         * can attend to both the old session AND the new prompt. */
        logits = forward(model, kv, scr, prompt_tokens, prompt_len, session_pos, k);
    } else if (prompt_cache_prefix_len > 0 && prompt_cache_prefix_len <= prompt_len) {
        /* Prompt cache loaded — skip prefilling the cached prefix portion.
         * Only prefill the NEW tokens (from prefix_len onward). */
        int remaining = prompt_len - prompt_cache_prefix_len;
        if (remaining > 0) {
            logits = forward(model, kv, scr,
                             prompt_tokens + prompt_cache_prefix_len,
                             remaining,
                             prompt_cache_prefix_len, k);
        } else {
            /* Entire prompt was cached — just do a single-token forward */
            int last_tok = prompt_tokens[prompt_len - 1];
            logits = forward(model, kv, scr, &last_tok, 1, prompt_cache_prefix_len - 1, k);
        }
    } else {
        /* Normal prefill — no cache */
        logits = forward(model, kv, scr, prompt_tokens, prompt_len, 0, k);
    }

    if (!logits) {
        fprintf(stderr, "Prefill forward pass failed\n");
        goto cleanup;
    }

    /* Save prompt cache if flag is set and file doesn't exist yet */
    if (!session_loaded && prompt_cache_path && prompt_cache_prefix_len == 0) {
        prompt_cache_save(kv, cfg, cfg->max_seq_len, prompt_len, prompt_cache_path);
    }

    double ttft = time_ms() - t_prefill_start;

    /* Effective starting position for decode */
    int decode_start_pos = session_loaded ? session_pos : prompt_len;

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
        int pos = decode_start_pos + i - 1;

        /* Check sequence length limit */
        if (pos >= cfg->max_seq_len - 1) {
            printf("\n[max sequence length reached]\n");
            break;
        }

        /* Thermal safety — sample every 10 tokens to avoid sysfs overhead */
        if (i % 10 == 0) {
            extern int cpu_get_temperature(void);
            int cpu_t = cpu_get_temperature();
            if (cpu_t >= 85) {
                usleep(100000); /* 100ms CPU throttle */
            } else if (cpu_t >= 80) {
                usleep(20000);  /* 20ms gentle backoff */
            }
#ifdef USE_CUDA
            if (cfg_local.kind == BACKEND_CUDA) {
                extern int cuda_get_temperature(void);
                int gpu_t = cuda_get_temperature();
                if (gpu_t >= 82) {
                    usleep(50000); /* 50ms GPU throttle */
                } else if (gpu_t >= 78) {
                    usleep(10000); /* 10ms gentle backoff */
                }
            }
#endif
        }
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

    /* ---- Session Save (AFTER decode loop — zero hot-path cost) ---- */
    if (session_path) {
        int final_pos = decode_start_pos + tokens_generated;
        kv_cache_save(kv, cfg, cfg->max_seq_len,
                      final_pos, prompt_tokens, prompt_len, session_path);
    }

cleanup:
    free(prompt_tokens);
    tokenizer_destroy(tok);
    scratch_destroy(scr);
    arena_destroy(arena);
    free_model(model);
    backend_destroy(backend);
}
