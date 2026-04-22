#!/bin/bash

# Guanaco Run Script
# This script simplifies running the Guanaco LLM Inference Runtime with various options.

set -e

# Configuration
BINARY="build/llmrt"

# ANSI Color Codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo -e "${YELLOW}Binary $BINARY not found. Building first...${NC}"
    if [ -f "./build.sh" ]; then
        ./build.sh
    else
        make
    fi
fi

# Help message
function show_help {
    echo -e "${YELLOW}>>> INSTRUCTIONS:${NC}"
    echo -e "${YELLOW}Run this script without arguments to start the Interactive Wizard.${NC}"
    echo -e "${YELLOW}Alternatively, pass specific flags (e.g., --model, --chat) for direct execution.${NC}"
    echo ""
    echo -e "${BLUE}Script Options:${NC}"
    echo "  --model <path>       Path to GGUF model file (required)"
    echo "  --prompt <text>      Prompt for inference"
    echo "  --chat               Enter interactive chat mode"
    echo "  --threads <n>        Number of threads to use"
    echo "  --device <kind>      Backend device: auto, cpu, or cuda"
    echo "  --n-gpu-layers <n>   Number of GPU layers to offload"
    echo "  --ctx <n>            Context buffer size limit"
    echo "  --max-tokens <n>     Maximum tokens to generate"
    echo "  --temp <val>         Sampling temperature"
    echo "  --top-k <val>        Top-K sampling value"
    echo "  --top-p <val>        Top-P sampling value"
    echo "  --help               Show this help message"
    echo ""
    
    if [ -f "$BINARY" ]; then
        echo -e "${BLUE}Binary Usage Information ($BINARY):${NC}"
        $BINARY --help || true
    fi
    echo "-------------------------------------"
}

# Wizard Mode
function run_wizard {
    echo -e "${BLUE}=== Guanaco Interactive Wizard ===${NC}"
    echo "Press Enter to accept [defaults] or type your value."
    echo ""

    # 1. Model Path
    DEFAULT_MODEL=""
    if [ -d "/home/animesh/Downloads" ]; then
        DEFAULT_MODEL=$(ls -t /home/animesh/Downloads/*.gguf 2>/dev/null | head -n 1)
    fi
    
    read -p "Enter model path [${DEFAULT_MODEL:-None}]: " MODEL_INPUT
    MODEL_PATH="${MODEL_INPUT:-$DEFAULT_MODEL}"
    if [ -z "$MODEL_PATH" ]; then
        echo -e "${RED}Error: Model path is required.${NC}"
        exit 1
    fi
    ARGS="$ARGS --model $MODEL_PATH"

    # 2. Mode
    read -p "Run in Chat mode? (y/n) [n]: " CHAT_INPUT
    if [[ "$CHAT_INPUT" =~ ^[Yy]$ ]]; then
        CHAT_MODE=1
        ARGS="$ARGS --chat"
    else
        read -p "Enter prompt [Hello]: " PROMPT_INPUT
        ARGS="$ARGS --prompt \"${PROMPT_INPUT:-Hello}\""
    fi

    # 3. Hardware Scaling
    read -p "Number of threads [8]: " THREADS_INPUT
    ARGS="$ARGS --threads ${THREADS_INPUT:-8}"

    read -p "Number of GPU layers to offload [0]: " GPU_LAYERS_INPUT
    ARGS="$ARGS --n-gpu-layers ${GPU_LAYERS_INPUT:-0}"

    read -p "Device (auto/cpu/cuda) [auto]: " DEVICE_INPUT
    ARGS="$ARGS --device ${DEVICE_INPUT:-auto}"

    # 4. Context & Generation
    read -p "Context buffer size limit [8192]: " CTX_INPUT
    ARGS="$ARGS --ctx ${CTX_INPUT:-8192}"

    read -p "Max tokens to generate [128]: " MAX_TOK_INPUT
    ARGS="$ARGS --max-tokens ${MAX_TOK_INPUT:-128}"

    # 5. Advanced (Sampling)
    read -p "Show advanced sampling parameters? (y/n) [n]: " ADV_INPUT
    if [[ "$ADV_INPUT" =~ ^[Yy]$ ]]; then
        read -p "Temperature [0.70]: " TEMP_INPUT
        ARGS="$ARGS --temperature ${TEMP_INPUT:-0.70}"
        read -p "Top-K [40]: " TOPK_INPUT
        ARGS="$ARGS --top-k ${TOPK_INPUT:-40}"
        read -p "Top-P [0.90]: " TOPP_INPUT
        ARGS="$ARGS --top-p ${TOPP_INPUT:-0.90}"
    fi
    echo "-------------------------------------"
}

# Parse arguments
ARGS=""
MODEL_PATH=""
CHAT_MODE=0

if [ "$#" -eq 0 ]; then
    run_wizard
else
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --model) MODEL_PATH="$2"; ARGS="$ARGS --model $2"; shift ;;
            --prompt) ARGS="$ARGS --prompt \"$2\""; shift ;;
            --chat) CHAT_MODE=1; ARGS="$ARGS --chat" ;;
            --threads) ARGS="$ARGS --threads $2"; shift ;;
            --n-gpu-layers) ARGS="$ARGS --n-gpu-layers $2"; shift ;;
            --device) ARGS="$ARGS --device $2"; shift ;;
            --ctx) ARGS="$ARGS --ctx $2"; shift ;;
            --max-tokens) ARGS="$ARGS --max-tokens $2"; shift ;;
            --temp) ARGS="$ARGS --temperature $2"; shift ;;
            --top-k) ARGS="$ARGS --top-k $2"; shift ;;
            --top-p) ARGS="$ARGS --top-p $2"; shift ;;
            --help) show_help; exit 0 ;;
            *) echo "Unknown parameter passed: $1"; exit 1 ;;
        esac
        shift
    done
fi

# Check for model path if not in help mode
if [ -z "$MODEL_PATH" ] && [ "$CHAT_MODE" -eq 0 ]; then
    # Try to find a model in a common location if not provided
    # For user convenience during assignment demonstration
    if [ -d "/home/animesh/Downloads" ]; then
        LATEST_MODEL=$(ls -t /home/animesh/Downloads/*.gguf 2>/dev/null | head -n 1)
        if [ ! -z "$LATEST_MODEL" ]; then
            echo -e "${BLUE}Auto-detected model: $LATEST_MODEL${NC}"
            MODEL_PATH="$LATEST_MODEL"
            ARGS="$ARGS --model $LATEST_MODEL"
        fi
    fi
    
    if [ -z "$MODEL_PATH" ]; then
        echo -e "${RED}Error: --model <path> is required.${NC}"
        show_help
        exit 1
    fi
fi

echo -e "${BLUE}=== Guanaco Execution Environment ===${NC}"
echo -e "Executing: $BINARY $ARGS"
echo -e "${NC}-------------------------------------"

# Run the binary
eval $BINARY $ARGS
