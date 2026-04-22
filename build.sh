#!/bin/bash

# Guanaco Build Script
# This script automates the build process for the Guanaco LLM Inference Runtime.

set -e

# Configuration
BUILD_DIR="build"
MAKE_FLAGS="-j$(nproc)"

# ANSI Color Codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== Guanaco Build System ===${NC}"

# Show Usage and Instructions First
function show_usage {
    echo -e "${YELLOW}>>> INSTRUCTIONS:${NC}"
    echo -e "${YELLOW}Use this script to compile the Guanaco LLM Runtime.${NC}"
    echo -e "${YELLOW}Pass --cuda to enable GPU acceleration, or --clean to rebuild from scratch.${NC}"
    echo ""
    echo -e "${BLUE}Script Options:${NC}"
    echo "  --cuda        Enable CUDA support"
    echo "  --no-pthread  Disable multithreading support"
    echo "  --clean       Clean build artifacts before building"
    echo "  --help        Show this help message"
    echo ""

    if [ -f "build/llmrt" ]; then
        echo -e "${BLUE}Binary Usage Information (build/llmrt):${NC}"
        build/llmrt --help || true
    else
        echo -e "${BLUE}Note: Compiled binary usage will be available after build.${NC}"
    fi
    echo "-------------------------------------"
}
show_usage

# Check for dependencies
if ! command -v gcc &> /dev/null; then
    echo -e "${RED}Error: gcc is not installed.${NC}"
    exit 1
fi

if ! command -v make &> /dev/null; then
    echo -e "${RED}Error: make is not installed.${NC}"
    exit 1
fi

# Parse arguments
USE_CUDA=0
USE_PTHREAD=1 # Enabled by default for performance
CLEAN=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda) USE_CUDA=1 ;;
        --no-pthread) USE_PTHREAD=0 ;;
        --clean) CLEAN=1 ;;
        --help) 
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Clean if requested
if [ "$CLEAN" -eq 1 ]; then
    echo -e "${YELLOW}Cleaning build artifacts...${NC}"
    make clean
fi

# Build command
echo -e "${BLUE}Building Guanaco...${NC}"
echo -e "Flags: USE_CUDA=$USE_CUDA USE_PTHREAD=$USE_PTHREAD"

if make USE_CUDA=$USE_CUDA USE_PTHREAD=$USE_PTHREAD $MAKE_FLAGS; then
    echo -e "${GREEN}Build successful! Binary located at $BUILD_DIR/llmrt${NC}"
else
    echo -e "${RED}Build failed.${NC}"
    exit 1
fi
