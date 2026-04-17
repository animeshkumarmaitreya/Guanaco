#include "chat.h"

#include <stdio.h>

#define PASS(name) printf("  PASS: %s\n", name)
#define FAIL(name, msg) do { printf("  FAIL: %s — %s\n", name, msg); failures++; } while(0)

static int failures = 0;

static void test_chat_unimplemented(void) {
    BackendConfig cfg = {0};
    cfg.kind = BACKEND_CPU;
    cfg.threads = 1;
    cfg.device_id = 0;

    int rc = chat_repl(&cfg, "model.gguf", 8, 0.0f, 0, 1.0f);
    if (rc != 0) PASS("chat_unimplemented");
    else FAIL("chat_unimplemented", "expected non-zero until implemented");
}

int main(void) {
    printf("=== Chat Test Suite (stub) ===\n");
    test_chat_unimplemented();

    printf("\n");
    if (failures == 0) {
        printf("All tests PASSED ✓\n");
    } else {
        printf("%d test(s) FAILED ✗\n", failures);
    }
    return failures;
}
