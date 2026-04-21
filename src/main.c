/*============================================================================
 * main.c — Entry point
 *
 * Parses CLI args and calls generate() or chat_repl().
 *============================================================================*/

#include "engine.h"
#include "tokenizer.h"
#include "backend.h"
#include "chat.h"
#include <stdio.h>
#include <stdlib.h>

static const char* device_kind_str(DeviceKind k) {
    switch (k) {
        case DEVICE_CPU:  return "cpu";
        case DEVICE_CUDA: return "cuda";
        case DEVICE_AUTO:
        default:          return "auto";
    }
}

int main(int argc, char** argv) {
    CLIArgs args;
    int result = cli_parse(argc, argv, &args);

    if (result == 1) return 0;   /* --help printed */
    if (result < 0)  return 1;   /* error */

    BackendConfig backend_cfg = {0};
    backend_cfg.threads = args.threads;
    backend_cfg.n_gpu_layers = args.n_gpu_layers;
    backend_cfg.device_id = 0;

    /* Backend selection is centralized in backend_create(); DEVICE_AUTO maps to BACKEND_AUTO. */
    if (args.device == DEVICE_CUDA) {
        backend_cfg.kind = BACKEND_CUDA;
    } else if (args.device == DEVICE_CPU) {
        backend_cfg.kind = BACKEND_CPU;
    } else {
        backend_cfg.kind = BACKEND_AUTO;
    }

    /* ---- Route to chat mode or generate mode ---- */
    if (args.chat) {
        /* Chat REPL: interactive multi-turn conversation */
        return chat_repl(&backend_cfg, args.model_path,
                         args.max_tokens, args.temperature,
                         args.top_k, args.top_p, args.ctx_len);
    } else {
        /* Single-turn generation */
        printf("LLM Inference Runtime\n");
        printf("  Model:       %s\n", args.model_path);
        printf("  Prompt:      \"%s\"\n", args.prompt);
        printf("  Max tokens:  %d\n", args.max_tokens);
        printf("  Temperature: %.2f\n", args.temperature);
        printf("  Top-k:       %d\n", args.top_k);
        printf("  Top-p:       %.2f\n", args.top_p);
        printf("  Ctx limit:   %d\n", args.ctx_len);
        printf("  Device:      %s\n", device_kind_str(args.device));
        printf("  Threads:     %d\n", args.threads);
        printf("---\n");

        generate(args.model_path, args.prompt, args.max_tokens,
                 args.temperature, args.top_k, args.top_p, args.ctx_len,
                 &backend_cfg);

        return 0;
    }
}
