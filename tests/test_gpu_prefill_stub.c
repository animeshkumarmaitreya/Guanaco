#include "gpu_prefill.h"

#include <stdio.h>

#define PASS(name) printf("  PASS: %s\n", name)
#define FAIL(name, msg) do { printf("  FAIL: %s — %s\n", name, msg); failures++; } while(0)

static int failures = 0;

static void test_gpu_prefill_unavailable(void) {
    int tokens[2] = {1, 2};

    int rc = gpu_prefill(NULL, NULL, NULL, NULL, tokens, 2, 0);
    if (rc != 0) PASS("gpu_prefill_unavailable");
    else FAIL("gpu_prefill_unavailable", "expected non-zero until implemented");
}

int main(void) {
    printf("=== GPU Prefill Test Suite (stub) ===\n");
    test_gpu_prefill_unavailable();

    printf("\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    return failures;
}
