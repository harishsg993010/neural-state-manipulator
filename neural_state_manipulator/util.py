"""
Utility functions for the Neural State Manipulator package.
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Any
from contextlib import contextmanager

@contextmanager
def hook_manager(hook_list: List[Any]):
    """
    Context manager to automatically remove hooks when exiting the context.
    
    Args:
        hook_list: List of hooks to manage.
    """
    try:
        yield hook_list
    finally:
        for hook in hook_list:
            hook.remove()

def list_manipulable_layers(model_name: str = "meta-llama/Llama-3.2-1B-Instruct") -> Dict[str, List[str]]:
    """
    Helper function to list all potentially manipulable layers in a model.
    
    Args:
        model_name: Hugging Face model identifier.
        
    Returns:
        Dictionary containing lists of attention layers, MLP layers, and other layers.
    """
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quant_config
    )

    attention_layers = []
    mlp_layers = []
    other_layers = []

    # Inspect model structure
    for name, module in model.named_modules():
        if len(list(module.parameters())) > 0:
            if 'attention' in name.lower():
                attention_layers.append(name)
            elif 'mlp' in name.lower():
                mlp_layers.append(name)
            else:
                other_layers.append(name)

    print(f"Model: {model_name}")
    print(f"Total attention layers: {len(attention_layers)}")
    print(f"Total MLP layers: {len(mlp_layers)}")
    print(f"Total other parameter layers: {len(other_layers)}")

    print("\nSample attention layers:")
    for layer in attention_layers[:5]:
        print(f"  - {layer}")

    print("\nSample MLP layers:")
    for layer in mlp_layers[:5]:
        print(f"  - {layer}")

    return {
        'attention': attention_layers,
        'mlp': mlp_layers,
        'other': other_layers
    }