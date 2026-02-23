#!/usr/bin/env python3
"""
Generate Golden Data for LLM Runtime Validation
Uses HuggingFace PyTorch to run TinyLlama-1.1B through a fixed prompt
and dumps the intermediate tensors (attention, MLP, norms, etc.) per layer.
These are saved as raw binary .bin files for the C engine to load and diff.
"""

import os
import sys
import struct
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def save_tensor(tensor: torch.Tensor, name: str, out_dir: str):
    """Save a PyTorch tensor as a flat array of FP32 floats."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.bin")
    
    # Ensure CPU, FP32, contiguous memory
    t_data = tensor.detach().cpu().float().contiguous().numpy()
    
    # Write raw binary
    with open(path, 'wb') as f:
        f.write(t_data.tobytes())
    
    # Print shape for reference
    print(f"Saved {name:30} \t Shape: {list(t_data.shape)}")

def main():
    model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    out_dir = "golden_data"
    
    print(f"Loading {model_id} from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model = model.to("cpu")
    model.eval()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Input IDs: {input_ids[0].tolist()}\n")
    
    save_tensor(input_ids, "input_ids", out_dir)

    # We use PyTorch forward hooks to capture intermediate layer outputs
    # Let's target layer 0 specifically to validate a single transformer_layer() call.
    layer_idx = 0
    layer = model.model.layers[layer_idx]
    
    captured = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            # output is sometimes a tuple (e.g. attention outputs)
            val = output[0] if isinstance(output, tuple) else output
            captured[name] = val
        return hook

    # Register hooks on Layer 0 components
    hooks = [
        layer.input_layernorm.register_forward_hook(hook_fn("layer0_norm1")),
        layer.self_attn.register_forward_hook(hook_fn("layer0_attn_out")),
        layer.post_attention_layernorm.register_forward_hook(hook_fn("layer0_norm2")),
        layer.mlp.register_forward_hook(hook_fn("layer0_mlp_out")),
        layer.register_forward_hook(hook_fn("layer0_output"))
    ]

    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save outputs
    print("\n--- Saving Golden Data ---")
    save_tensor(logits[0, -1, :], "logits_final_token", out_dir)
    
    for name, tensor in captured.items():
        save_tensor(tensor[0], name, out_dir)  # [0] strips batch dimension
        
    print("\nGolden data generation complete. You can now diff C arrays against these .bin files.")

if __name__ == "__main__":
    main()
