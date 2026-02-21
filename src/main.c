/*============================================================================
 * main.c — Entry point
 *
 * Parses CLI args and calls generate().
 *============================================================================*/

#include "engine.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    CLIArgs args;
    int result = cli_parse(argc, argv, &args);

    if (result == 1) return 0;   /* --help printed */
    if (result < 0)  return 1;   /* error */

    printf("LLM Inference Runtime\n");
    printf("  Model:       %s\n", args.model_path);
    printf("  Prompt:      \"%s\"\n", args.prompt);
    printf("  Max tokens:  %d\n", args.max_tokens);
    printf("  Temperature: %.2f\n", args.temperature);
    printf("  Top-k:       %d\n", args.top_k);
    printf("  Top-p:       %.2f\n", args.top_p);
    printf("---\n");

    generate(args.model_path, args.prompt, args.max_tokens,
             args.temperature, args.top_k, args.top_p);

    return 0;
}
