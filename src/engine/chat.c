#include "chat.h"

#include "engine.h"
#include "memory.h"
#include "tokenizer.h"
#include "backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHAT_MAX_INPUT_LEN 2048
#define CHAT_MAX_OUTPUT_TOKENS 512

/* Timing helper */
static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int chat_repl(const BackendConfig* backend_cfg,
              const char* model_path,
              int max_tokens,
              float temperature,
              int top_k,
              float top_p) {
    
    if (!backend_cfg || !model_path) {
        fprintf(stderr, "Error: invalid arguments to chat_repl\n");
        return -1;
    }

    /* ---- Setup backend ---- */
    BackendConfig cfg_local = *backend_cfg;
    Backend* backend = backend_create(&cfg_local);
    if (!backend) {
        fprintf(stderr, "Error: failed to create backend\n");
        return -1;
    }
    const KernelVTable* k = backend_kernels(backend);
    if (!k) {
        fprintf(stderr, "Error: backend returned NULL kernel vtable\n");
        backend_destroy(backend);
        return -1;
    }

    /* ---- Load model ---- */
    printf("Loading model: %s\n", model_path);
    double t0 = time_ms();
    
    ModelWeights* model = load_model(model_path, NULL);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        backend_destroy(backend);
        return -1;
    }

    double load_time = time_ms() - t0;
    printf("Model loaded in %.1f ms\n", load_time);

    ModelConfig* cfg = &model->config;
    const int eos_token_id = (cfg->eos_token_id > 0) ? cfg->eos_token_id : 2;

    /* ---- Create allocators ---- */
    size_t kv_size = 2UL * cfg->n_layers * cfg->n_kv_heads * cfg->max_seq_len
                     * cfg->head_dim * sizeof(float);
    size_t arena_size = kv_size + 64 * 1024 * 1024;
    Arena* arena = arena_create(arena_size);
    if (!arena) {
        fprintf(stderr, "Failed to create arena (%zu bytes)\n", arena_size);
        free_model(model);
        backend_destroy(backend);
        return -1;
    }

    size_t scratch_size = 512 * 1024 * 1024;
    Scratch* scr = scratch_create(scratch_size);
    if (!scr) {
        fprintf(stderr, "Failed to create scratch (%zu bytes)\n", scratch_size);
        arena_destroy(arena);
        free_model(model);
        backend_destroy(backend);
        return -1;
    }

    /* ---- Create KV cache ---- */
    KVCache* kv = kv_cache_create(arena, cfg, cfg->max_seq_len);
    if (!kv) {
        fprintf(stderr, "Failed to create KV cache\n");
        scratch_destroy(scr);
        arena_destroy(arena);
        free_model(model);
        backend_destroy(backend);
        return -1;
    }

    /* ---- Create tokenizer ---- */
    Tokenizer* tok = tokenizer_create(cfg);
    if (!tok) {
        fprintf(stderr, "Failed to create tokenizer\n");
        scratch_destroy(scr);
        arena_destroy(arena);
        free_model(model);
        backend_destroy(backend);
        return -1;
    }

    /* ---- Initialize conversation context ---- */
    int current_pos = 0;  /* Position counter for KV cache */
    char user_input[CHAT_MAX_INPUT_LEN];

    printf("\n=== Chat Mode ===\n");
    printf("Type your message and press Enter. Type 'exit' to quit.\n");
    printf("(Use Ctrl-C to force quit)\n");
    printf("-\n");

    /* Interactive chat loop */
    while (1) {
        printf("\nYou: ");
        fflush(stdout);

        /* Read user input */
        if (!fgets(user_input, sizeof(user_input), stdin)) {
            /* EOF or read error */
            break;
        }

        /* Remove trailing newline if present */
        int input_len = strlen(user_input);
        if (input_len > 0 && user_input[input_len - 1] == '\n') {
            user_input[input_len - 1] = '\0';
            input_len--;
        }

        /* Check for exit command */
        if (strcmp(user_input, "exit") == 0) {
            printf("\nGoodbye!\n");
            break;
        }

        /* Skip empty input */
        if (input_len == 0) {
            continue;
        }

        /* ---- Tokenize user input ----
         * Inject BOS exactly once at the beginning of each conversation.
         * Subsequent turns must not inject BOS (KV reuse correctness). */
        int user_tokens_len = 0;
        int* user_tokens = (current_pos == 0)
                               ? tokenize(tok, user_input, &user_tokens_len)
                               : tokenize_no_bos(tok, user_input, &user_tokens_len);
        if (!user_tokens || user_tokens_len == 0) {
            fprintf(stderr, "Tokenization failed\n");
            continue;
        }

        /* ---- Check sequence length limit ---- */
        if (current_pos + user_tokens_len >= cfg->max_seq_len - 1) {
            printf("Assistant: [Context window full - restarting conversation]\n");
            /* Reset KV cache for new conversation */
            current_pos = 0;
            scratch_reset(scr);

            /* Re-tokenize with BOS for the new conversation context. */
            free(user_tokens);
            user_tokens_len = 0;
            user_tokens = tokenize(tok, user_input, &user_tokens_len);
            if (!user_tokens || user_tokens_len == 0) {
                fprintf(stderr, "Tokenization failed\n");
                continue;
            }

            if (user_tokens_len >= cfg->max_seq_len - 1) {
                printf("Assistant: [Input too long for context window]\n");
                free(user_tokens);
                continue;
            }
        }

        /* ---- Forward pass on user input ---- */
        Tensor* logits = forward(model, kv, scr, user_tokens, user_tokens_len, 
                                  current_pos, k);
        free(user_tokens);

        if (!logits) {
            fprintf(stderr, "Forward pass failed\n");
            continue;
        }

        /* Update position after user input */
        current_pos += user_tokens_len;

        /* ---- Sample and generate assistant response ---- */
        printf("Assistant: ");
        fflush(stdout);

        float* logit_data = (float*)logits->data;
        int next_token;

        /* Sample first token from user input response */
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

        int tokens_generated = 1;

        /* ---- Decode loop for assistant response ---- */
        for (int i = 1; i < max_tokens && tokens_generated < CHAT_MAX_OUTPUT_TOKENS; i++) {
            int pos = current_pos + i - 1;

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

            /* Check for EOS token */
            if (next_token == eos_token_id) {
                break;
            }

            /* Print token */
            token_str = detokenize(tok, next_token);
            printf("%s", token_str);
            fflush(stdout);
        }

        /* Update position after assistant response */
        current_pos += tokens_generated;

        printf("\n");
    }

    /* ---- Cleanup ---- */
    tokenizer_destroy(tok);
    scratch_destroy(scr);
    arena_destroy(arena);
    free_model(model);
    backend_destroy(backend);

    return 0;
}
