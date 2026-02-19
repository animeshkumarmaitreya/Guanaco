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

/* ---- Tokenizer tests ---- */

static void test_tokenizer_create(void) {
    Tokenizer* tok = tokenizer_create("dummy_path");
    if (tok) {
        PASS("tokenizer_create");
        tokenizer_destroy(tok);
    } else {
        FAIL("tokenizer_create", "returned NULL");
    }
}

static void test_tokenizer_basic(void) {
    Tokenizer* tok = tokenizer_create("dummy");
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
    Tokenizer* tok = tokenizer_create("dummy");
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

static void test_detokenize(void) {
    Tokenizer* tok = tokenizer_create("dummy");
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

/* ---- Main ---- */

int main(void) {
    printf("=== Tokenizer & Sampler Test Suite ===\n");

    test_greedy_basic();
    test_greedy_first_max();
    test_greedy_negative();
    test_greedy_single();
    test_temperature_zero();
    test_temperature_deterministic();
    test_tokenizer_create();
    test_tokenizer_basic();
    test_tokenizer_empty();
    test_detokenize();
    test_cli_help();
    test_cli_missing_model();
    test_cli_full();

    printf("\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    return failures;
}
