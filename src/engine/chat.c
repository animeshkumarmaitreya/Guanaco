#include "chat.h"

#include <stdio.h>

int chat_repl(const BackendConfig* backend_cfg,
              const char* model_path,
              int max_tokens,
              float temperature,
              int top_k,
              float top_p) {
    (void)backend_cfg;
    (void)model_path;
    (void)max_tokens;
    (void)temperature;
    (void)top_k;
    (void)top_p;

    fprintf(stderr, "Error: --chat is not implemented yet\n");
    return -1;
}
