#ifndef LLMRT_CHAT_H
#define LLMRT_CHAT_H

#include "backend.h"

/*============================================================================
 * Chat REPL (MUST) — scaffolding
 *============================================================================*/

/* Returns 0 on clean exit; non-zero on error. */
int chat_repl(const BackendConfig* backend_cfg,
              const char* model_path,
              int max_tokens,
              float temperature,
              int top_k,
              float top_p);

#endif /* LLMRT_CHAT_H */
