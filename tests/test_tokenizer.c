/*============================================================================
 * Tokenizer & Sampler Test Suite — Person C's test harness
 *
 * Run: make test_tokenizer && ./build/test_tokenizer
 *============================================================================*/

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#define PASS(name) printf("  PASS: %s\n", name)
#define FAIL(name, msg) do { printf("  FAIL: %s — %s\n", name, msg); failures++; } while(0)

static int failures = 0;

/* ---- Sampler tests (no tokenizer dependency) ---- */

static void test_greedy_basic(void) {
    float logits[] = {1.0f, 5.0f, 2.0f, 3.0f};
    int result = sample_greedy(logits, 4);
    if (result == 1) PASS("greedy_basic");
    else             FAIL("greedy_basic", "expected index 1");
}

static void test_greedy_first_max(void) {
    /* When multiple maxima, should return first */
    float logits[] = {3.0f, 1.0f, 3.0f, 2.0f};
    int result = sample_greedy(logits, 4);
    if (result == 0) PASS("greedy_first_max");
    else             FAIL("greedy_first_max", "expected index 0 (first max)");
}

static void test_greedy_negative(void) {
    float logits[] = {-5.0f, -2.0f, -10.0f, -1.0f};
    int result = sample_greedy(logits, 4);
    if (result == 3) PASS("greedy_negative");
    else             FAIL("greedy_negative", "expected index 3");
}

static void test_greedy_single(void) {
    float logits[] = {42.0f};
    int result = sample_greedy(logits, 1);
    if (result == 0) PASS("greedy_single");
    else             FAIL("greedy_single", "expected index 0");
}

static void test_temperature_zero(void) {
    float logits[] = {1.0f, 5.0f, 2.0f, 3.0f};
    int result = sample_temperature(logits, 4, 0.0f);
    if (result == 1) PASS("temp_zero (greedy)");
    else             FAIL("temp_zero (greedy)", "expected index 1");
}

static void test_temperature_deterministic(void) {
    /* With very low temperature, should almost always pick argmax */
    float logits[] = {1.0f, 10.0f, 2.0f};
    int count_1 = 0;
    for (int trial = 0; trial < 100; trial++) {
        int result = sample_temperature(logits, 3, 0.01f);
        if (result == 1) count_1++;
    }
    /* With stub (greedy fallback), this always picks 1 */
    if (count_1 >= 95) PASS("temp_deterministic");
    else               FAIL("temp_deterministic", "expected index 1 almost always");
}

static void test_topk_temp_zero_is_greedy(void) {
    float logits[] = {1.0f, 5.0f, 2.0f, 3.0f};
    int result = sample_top_k(logits, 4, 2, 0.0f);
    if (result == 1) PASS("topk_temp_zero_is_greedy");
    else             FAIL("topk_temp_zero_is_greedy", "expected argmax index 1");
}

static void test_topp_temp_zero_is_greedy(void) {
    float logits[] = {1.0f, 5.0f, 2.0f, 3.0f};
    int result = sample_top_p(logits, 4, 0.9f, 0.0f);
    if (result == 1) PASS("topp_temp_zero_is_greedy");
    else             FAIL("topp_temp_zero_is_greedy", "expected argmax index 1");
}

/* ---- Tokenizer tests ---- */

static void test_tokenizer_create(void) {
    char* dummy[1] = {"A"};
    ModelConfig cfg = { .vocab_size = 1, .vocab_strings = dummy };
    Tokenizer* tok = tokenizer_create(&cfg);
    if (tok) {
        PASS("tokenizer_create");
        tokenizer_destroy(tok);
    } else {
        FAIL("tokenizer_create", "returned NULL");
    }
}

static void test_tokenizer_basic(void) {
    char* dummy[2] = {"Hell", "o"};
    ModelConfig cfg = { .vocab_size = 2, .vocab_strings = dummy };
    Tokenizer* tok = tokenizer_create(&cfg);
    if (!tok) { FAIL("tokenizer_basic", "create failed"); return; }

    int len = 0;
    int* ids = tokenize(tok, "Hello", &len);

    if (len > 0 && ids) {
        PASS("tokenizer_basic");
    } else {
        FAIL("tokenizer_basic", "returned no tokens");
    }

    free(ids);
    tokenizer_destroy(tok);
}

static void test_tokenizer_empty(void) {
    char* dummy[1] = {"A"};
    ModelConfig cfg = { .vocab_size = 1, .vocab_strings = dummy };
    Tokenizer* tok = tokenizer_create(&cfg);
    if (!tok) { FAIL("tokenizer_empty", "create failed"); return; }

    int len = 0;
    int* ids = tokenize(tok, "", &len);

    if (len == 0) {
        PASS("tokenizer_empty");
    } else {
        FAIL("tokenizer_empty", "expected 0 tokens for empty string");
    }
    free(ids);
    tokenizer_destroy(tok);
}

static void test_tokenizer_no_bos(void) {
    char* dummy[3] = {"H", "<BOS>", "Hello"};
    ModelConfig cfg = { .vocab_size = 3, .vocab_strings = dummy, .bos_token_id = 1 };
    Tokenizer* tok = tokenizer_create(&cfg);
    if (!tok) { FAIL("tokenizer_no_bos", "create failed"); return; }

    int len_bos = 0;
    int* ids_bos = tokenize(tok, "Hello", &len_bos);

    int len_no_bos = 0;
    int* ids_no_bos = tokenize_no_bos(tok, "Hello", &len_no_bos);

    if (!ids_bos || !ids_no_bos) {
        FAIL("tokenizer_no_bos", "tokenize returned NULL");
    } else if (len_bos != len_no_bos + 1) {
        FAIL("tokenizer_no_bos", "expected tokenize() to add exactly one BOS token");
    } else if (ids_bos[0] != cfg.bos_token_id) {
        FAIL("tokenizer_no_bos", "expected first token to be BOS");
    } else if (ids_no_bos[0] == cfg.bos_token_id) {
        FAIL("tokenizer_no_bos", "expected tokenize_no_bos() to not inject BOS");
    } else {
        PASS("tokenizer_no_bos");
    }

    free(ids_bos);
    free(ids_no_bos);
    tokenizer_destroy(tok);
}

static void test_tokenizer_bos_id_zero(void) {
    char* dummy[3] = {"<BOS>", "Hello", "H"};
    ModelConfig cfg = { .vocab_size = 3, .vocab_strings = dummy, .bos_token_id = 0 };
    Tokenizer* tok = tokenizer_create(&cfg);
    if (!tok) { FAIL("tokenizer_bos_id_zero", "create failed"); return; }

    int len = 0;
    int* ids = tokenize(tok, "Hello", &len);
    if (!ids) {
        FAIL("tokenizer_bos_id_zero", "tokenize returned NULL");
    } else if (len < 2) {
        FAIL("tokenizer_bos_id_zero", "expected at least BOS + 1 token");
    } else if (ids[0] != 0) {
        FAIL("tokenizer_bos_id_zero", "expected BOS token id 0 to be used");
    } else {
        PASS("tokenizer_bos_id_zero");
    }

    free(ids);
    tokenizer_destroy(tok);
}

static void test_detokenize(void) {
    char* dummy[100] = {0}; dummy[65] = "A";
    ModelConfig cfg = { .vocab_size = 100, .vocab_strings = dummy };
    Tokenizer* tok = tokenizer_create(&cfg);
    if (!tok) { FAIL("detokenize", "create failed"); return; }

    const char* s = detokenize(tok, 65);  /* ASCII 'A' */
    if (s && s[0] == 'A') {
        PASS("detokenize");
    } else {
        FAIL("detokenize", "expected 'A'");
    }
    tokenizer_destroy(tok);
}

/* ---- CLI tests ---- */

static void test_cli_help(void) {
    char* argv[] = {"main", "--help"};
    CLIArgs args;
    int result = cli_parse(2, argv, &args);
    /* --help returns 1 (help printed) */
    if (result == 1) PASS("cli_help");
    else             FAIL("cli_help", "expected return 1 for --help");
}

static void test_cli_missing_model(void) {
    char* argv[] = {"main", "--prompt", "hello"};
    CLIArgs args;
    int result = cli_parse(3, argv, &args);
    if (result == -1) PASS("cli_missing_model");
    else              FAIL("cli_missing_model", "expected error for missing --model");
}

static void test_cli_full(void) {
    char* argv[] = {"main", "--model", "test.gguf", "--prompt", "Hello world",
                    "--max-tokens", "64", "--temperature", "0.5"};
    CLIArgs args;
    int result = cli_parse(9, argv, &args);

    if (result != 0) { FAIL("cli_full", "parse failed"); return; }
    if (strcmp(args.model_path, "test.gguf") != 0) { FAIL("cli_full", "wrong model path"); return; }
    if (strcmp(args.prompt, "Hello world") != 0)    { FAIL("cli_full", "wrong prompt"); return; }
    if (args.max_tokens != 64)                       { FAIL("cli_full", "wrong max_tokens"); return; }
    if (fabsf(args.temperature - 0.5f) > 0.001f)    { FAIL("cli_full", "wrong temperature"); return; }
    PASS("cli_full");
}

static void test_cli_device_cuda_errors(void) {
    char* argv[] = {"main", "--model", "test.gguf", "--device", "cuda"};
    CLIArgs args;
    int result = cli_parse(5, argv, &args);
#if defined(USE_CUDA) && (USE_CUDA == 1)
    if (result != 0) {
        FAIL("cli_device_cuda_errors", "expected --device cuda to parse in CUDA-enabled build");
    } else if (args.device != DEVICE_CUDA) {
        FAIL("cli_device_cuda_errors", "expected args.device=DEVICE_CUDA");
    } else if (args.n_gpu_layers <= 0) {
        FAIL("cli_device_cuda_errors", "expected auto default --n-gpu-layers > 0 for CUDA device");
    } else {
        PASS("cli_device_cuda_errors");
    }
#else
    if (result == -1) PASS("cli_device_cuda_errors");
    else              FAIL("cli_device_cuda_errors", "expected error for --device cuda when CUDA is disabled");
#endif
}

static void test_cli_chat_errors(void) {
    char* argv[] = {"main", "--model", "test.gguf", "--chat"};
    CLIArgs args;
    int result = cli_parse(4, argv, &args);
    if (result != 0) {
        FAIL("cli_chat_parses", "expected successful parse for --chat");
        return;
    }
    if (!args.chat) {
        FAIL("cli_chat_parses", "expected args.chat=1");
        return;
    }
    PASS("cli_chat_parses");
}

/* ---- Main ---- */

int main(void) {
    printf("=== Tokenizer & Sampler Test Suite ===\n");

    test_greedy_basic();
    test_greedy_first_max();
    test_greedy_negative();
    test_greedy_single();
    test_temperature_zero();
    test_temperature_deterministic();
    test_topk_temp_zero_is_greedy();
    test_topp_temp_zero_is_greedy();
    test_tokenizer_create();
    test_tokenizer_basic();
    test_tokenizer_empty();
    test_tokenizer_no_bos();
    test_tokenizer_bos_id_zero();
    test_detokenize();
    test_cli_help();
    test_cli_missing_model();
    test_cli_full();
    test_cli_device_cuda_errors();
    test_cli_chat_errors();

    printf("\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    return failures;
}
