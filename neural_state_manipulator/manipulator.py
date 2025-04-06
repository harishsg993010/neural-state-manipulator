"""
Core implementation of the Neural State Manipulator class.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from typing import List, Dict, Any, Optional
from .utils import hook_manager

class NeuralStateManipulator:
    """
    A class for recording and manipulating the internal neural activations of LLMs.
    
    This class provides methods to record activation patterns from specific
    text inputs and then use those patterns to influence generation behavior.
    """
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", 
                 load_in_4bit: bool = True, 
                 load_in_8bit: bool = False):
        """
        Initialize the NeuralStateManipulator with a specified model.
        
        Args:
            model_name: Hugging Face model identifier.
            load_in_4bit: Whether to load the model in 4-bit precision.
            load_in_8bit: Whether to load the model in 8-bit precision.
        """
        print(f"Loading {model_name}...")
        
        # Set up quantization config
        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif load_in_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quant_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hooks = []
        self.layer_outputs = {}
        self.behavior_patterns = {}
        self.model_layers = self._identify_manipulable_layers()

    def _identify_manipulable_layers(self) -> Dict[str, List[str]]:
        """
        Identify layers in the model that can be manipulated.
        
        Returns:
            Dictionary of layer names categorized by type.
        """
        layers = {'attention': [], 'mlp': []}
        for name, module in self.model.named_modules():
            if len(list(module.parameters())) > 0:
                if 'attention' in name.lower():
                    layers['attention'].append(name)
                elif 'mlp' in name.lower():
                    layers['mlp'].append(name)
        return layers

    def register_activation_hook(self, layer_name: str):
        """
        Register a forward hook to capture activations from a specific layer.
        
        Args:
            layer_name: Name of the layer to hook.
            
        Returns:
            The hook handle.
        """
        def get_activation(name):
            def hook(module, inputs, output):
                if isinstance(output, tuple):
                    self.layer_outputs[name] = output[0].clone().detach()
                else:
                    self.layer_outputs[name] = output.clone().detach()
            return hook

        target_layer = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_layer = module
                break
        if target_layer is None:
            raise ValueError(f"Layer {layer_name} not found in model.")
        hook = target_layer.register_forward_hook(get_activation(layer_name))
        self.hooks.append(hook)
        return hook

    def record_behavior_pattern(self, text: str, behavior_name: str,
                                layers_to_monitor: List[str]):
        """
        Process text to record the activations for selected layers.
        
        Args:
            text: Input text to process.
            behavior_name: Name to assign to the recorded pattern.
            layers_to_monitor: List of layer names to record activations from.
        """
        self.layer_outputs = {}
        for layer in layers_to_monitor:
            self.register_activation_hook(layer)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            _ = self.model(**inputs)
        behavior_pattern = {}
        for layer_name, activation in self.layer_outputs.items():
            # Average over sequence dimension for a smoother pattern
            behavior_pattern[layer_name] = activation.mean(dim=1, keepdim=True).clone().detach()
        self.behavior_patterns[behavior_name] = behavior_pattern
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print(f"Recorded activation pattern for behavior: {behavior_name}")

    def generate_plain(self, prompt: str, max_new_tokens: int = 300, 
                      temperature: float = 0.7) -> str:
        """
        Generate text without any manipulation.
        
        Args:
            prompt: Input text to start generation.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated text.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_with_manipulation(self, prompt: str,
                                  target_behavior: str,
                                  influence_strength: float = 0.3,
                                  target_layers: Optional[List[str]] = None,
                                  max_new_tokens: int = 300,
                                  temperature: float = 0.7,
                                  min_new_tokens: int = 300,
                                  ramp_start: int = 10,
                                  ramp_length: int = 20) -> str:
        """
        Generate text while adding a recorded behavior activation pattern.
        A ramp function gradually increases the influence over generation steps.
        
        Args:
            prompt: Input text to start generation.
            target_behavior: Name of the behavior pattern to apply.
            influence_strength: Strength of the manipulation.
            target_layers: List of layers to manipulate (defaults to all recorded layers).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            min_new_tokens: Minimum number of tokens to generate.
            ramp_start: Token position to start ramping up influence.
            ramp_length: Number of tokens over which to ramp up influence.
            
        Returns:
            Generated text with manipulation applied.
        """
        if target_behavior not in self.behavior_patterns:
            raise ValueError(f"Behavior pattern '{target_behavior}' not found. Please record it first.")
        if target_layers is None:
            target_layers = list(self.behavior_patterns[target_behavior].keys())
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        manipulation_hooks = []
        for layer_name in target_layers:
            if layer_name not in self.behavior_patterns[target_behavior]:
                print(f"Warning: No recorded pattern for layer {layer_name}. Skipping.")
                continue
            target_layer = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    target_layer = module
                    break
            if target_layer is None:
                print(f"Warning: Layer {layer_name} not found. Skipping.")
                continue
            pattern = self.behavior_patterns[target_behavior][layer_name]
            token_pos = 0
            def ramp(token_idx):
                if token_idx < ramp_start:
                    return 0.0
                elif token_idx >= ramp_start + ramp_length:
                    return 1.0
                else:
                    return (token_idx - ramp_start) / ramp_length
            def make_manipulation_hook(layer_name, pattern):
                nonlocal token_pos
                def hook(module, input_val, output):
                    nonlocal token_pos
                    if isinstance(output, tuple):
                        current_output = output[0]
                        is_tuple = True
                    else:
                        current_output = output
                        is_tuple = False
                    if len(current_output.shape) == 3:
                        if current_output.shape[2] != pattern.shape[2]:
                            adapted_pattern = pattern.mean(dim=1, keepdim=True).expand(-1, current_output.shape[1], -1)
                        else:
                            adapted_pattern = pattern
                        factor = ramp(token_pos)
                        manipulated = current_output + influence_strength * factor * adapted_pattern.to(current_output.device)
                    elif len(current_output.shape) == 4:
                        if current_output.shape[3] != pattern.shape[3]:
                            adapted_pattern = pattern.mean(dim=2, keepdim=True).expand(-1, -1, current_output.shape[2], -1)
                        else:
                            adapted_pattern = pattern
                        factor = ramp(token_pos)
                        manipulated = current_output + influence_strength * factor * adapted_pattern.to(current_output.device)
                    else:
                        manipulated = current_output
                    token_pos += 1
                    if is_tuple:
                        return (manipulated,) + output[1:]
                    return manipulated
                return hook
            hook = target_layer.register_forward_hook(make_manipulation_hook(layer_name, pattern))
            manipulation_hooks.append(hook)

        with hook_manager(manipulation_hooks):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    min_length=input_ids.shape[1] + min_new_tokens,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    do_sample=True,
                    temperature=temperature
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def generate_with_erasure(self, prompt: str,
                              target_behavior: str,
                              erasure_strength: float = 0.3,
                              target_layers: Optional[List[str]] = None,
                              max_new_tokens: int = 300,
                              temperature: float = 0.7) -> str:
        """
        Generate text while subtracting the recorded behavior pattern from the activations.
        This serves to "erase" the behavior from the generation.
        
        Args:
            prompt: Input text to start generation.
            target_behavior: Name of the behavior pattern to erase.
            erasure_strength: Strength of the erasure.
            target_layers: List of layers to manipulate.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated text with erasure applied.
        """
        if target_behavior not in self.behavior_patterns:
            raise ValueError(f"Behavior pattern '{target_behavior}' not found. Please record it first.")
        if target_layers is None:
            target_layers = list(self.behavior_patterns[target_behavior].keys())
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        erasure_hooks = []
        for layer_name in target_layers:
            if layer_name not in self.behavior_patterns[target_behavior]:
                print(f"Warning: No recorded pattern for layer {layer_name}. Skipping.")
                continue
            target_layer = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    target_layer = module
                    break
            if target_layer is None:
                print(f"Warning: Layer {layer_name} not found. Skipping.")
                continue
            pattern = self.behavior_patterns[target_behavior][layer_name]
            def make_erasure_hook(pattern):
                def hook(module, input_val, output):
                    if isinstance(output, tuple):
                        current_output = output[0]
                        is_tuple = True
                    else:
                        current_output = output
                        is_tuple = False
                    # Subtract the pattern (erasure)
                    manipulated = current_output - erasure_strength * pattern.to(current_output.device)
                    if is_tuple:
                        return (manipulated,) + output[1:]
                    return manipulated
                return hook
            hook = target_layer.register_forward_hook(make_erasure_hook(pattern))
            erasure_hooks.append(hook)

        with hook_manager(erasure_hooks):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def generate_with_amplification(self, prompt: str,
                                    target_behavior: str,
                                    amplification_factor: float = 1.5,
                                    target_layers: Optional[List[str]] = None,
                                    max_new_tokens: int = 300,
                                    temperature: float = 0.7) -> str:
        """
        Generate text while amplifying (adding a scaled version of) the recorded behavior pattern.
        This is akin to "enhancing" the behavior in generation.
        
        Args:
            prompt: Input text to start generation.
            target_behavior: Name of the behavior pattern to amplify.
            amplification_factor: Factor by which to amplify the pattern.
            target_layers: List of layers to manipulate.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated text with amplification applied.
        """
        if target_behavior not in self.behavior_patterns:
            raise ValueError(f"Behavior pattern '{target_behavior}' not found. Please record it first.")
        if target_layers is None:
            target_layers = list(self.behavior_patterns[target_behavior].keys())
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        amplification_hooks = []
        for layer_name in target_layers:
            if layer_name not in self.behavior_patterns[target_behavior]:
                print(f"Warning: No recorded pattern for layer {layer_name}. Skipping.")
                continue
            target_layer = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    target_layer = module
                    break
            if target_layer is None:
                print(f"Warning: Layer {layer_name} not found. Skipping.")
                continue
            pattern = self.behavior_patterns[target_behavior][layer_name]
            def make_amplification_hook(pattern):
                def hook(module, input_val, output):
                    if isinstance(output, tuple):
                        current_output = output[0]
                        is_tuple = True
                    else:
                        current_output = output
                        is_tuple = False
                    # Add amplified pattern to the activations
                    manipulated = current_output + amplification_factor * pattern.to(current_output.device)
                    if is_tuple:
                        return (manipulated,) + output[1:]
                    return manipulated
                return hook
            hook = target_layer.register_forward_hook(make_amplification_hook(pattern))
            amplification_hooks.append(hook)

        with hook_manager(amplification_hooks):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def generate_with_interpolation(self, prompt: str,
                                    target_behavior: str,
                                    interpolation_factor: float = 0.5,
                                    target_layers: Optional[List[str]] = None,
                                    max_new_tokens: int = 300,
                                    temperature: float = 0.7) -> str:
        """
        Generate text while interpolating between the original activations and the behavior pattern.
        
        Args:
            prompt: Input text to start generation.
            target_behavior: Name of the behavior pattern to interpolate with.
            interpolation_factor: Interpolation factor (0=baseline, 1=full behavior).
            target_layers: List of layers to manipulate.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Generated text with interpolation applied.
        """
        if target_behavior not in self.behavior_patterns:
            raise ValueError(f"Behavior pattern '{target_behavior}' not found. Please record it first.")
        if target_layers is None:
            target_layers = list(self.behavior_patterns[target_behavior].keys())
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        interp_hooks = []
        for layer_name in target_layers:
            if layer_name not in self.behavior_patterns[target_behavior]:
                print(f"Warning: No recorded pattern for layer {layer_name}. Skipping.")
                continue
            target_layer = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    target_layer = module
                    break
            if target_layer is None:
                print(f"Warning: Layer {layer_name} not found. Skipping.")
                continue
            pattern = self.behavior_patterns[target_behavior][layer_name]
            def make_interpolation_hook(pattern):
                def hook(module, input_val, output):
                    if isinstance(output, tuple):
                        current_output = output[0]
                        is_tuple = True
                    else:
                        current_output = output
                        is_tuple = False
                    # Interpolate between original activation and behavior pattern
                    manipulated = (1 - interpolation_factor) * current_output + interpolation_factor * pattern.to(current_output.device)
                    if is_tuple:
                        return (manipulated,) + output[1:]
                    return manipulated
                return hook
            hook = target_layer.register_forward_hook(make_interpolation_hook(pattern))
            interp_hooks.append(hook)

        with hook_manager(interp_hooks):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def capture_concept_neurons(self, positive_texts: List[str],
                               negative_texts: List[str],
                               concept_name: str,
                               layers_to_analyze: Optional[List[str]] = None):
        """
        Identify neurons associated with a specific concept by comparing
        activations on positive and negative examples.
        
        Args:
            positive_texts: List of texts that exemplify the concept.
            negative_texts: List of texts that do not contain the concept.
            concept_name: Name to assign to the identified concept neurons.
            layers_to_analyze: List of layers to analyze for concept neurons.
        """
        if layers_to_analyze is None:
            attn_layers = self.model_layers['attention'][:3]
            mlp_layers = self.model_layers['mlp'][:3]
            layers_to_analyze = attn_layers + mlp_layers
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_outputs = {}
        for layer in layers_to_analyze:
            self.register_activation_hook(layer)
        positive_activations = {layer: [] for layer in layers_to_analyze}
        for text in positive_texts:
            self.layer_outputs = {}
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                _ = self.model(**inputs)
            for layer, activation in self.layer_outputs.items():
                if layer in layers_to_analyze:
                    mean_activation = activation.mean(dim=0).to(torch.float32).cpu().numpy()
                    positive_activations[layer].append(mean_activation)
        negative_activations = {layer: [] for layer in layers_to_analyze}
        for text in negative_texts:
            self.layer_outputs = {}
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                _ = self.model(**inputs)
            for layer, activation in self.layer_outputs.items():
                if layer in layers_to_analyze:
                    mean_activation = activation.mean(dim=0).to(torch.float32).cpu().numpy()
                    negative_activations[layer].append(mean_activation)
        concept_neurons = {}
        for layer in layers_to_analyze:
            if not positive_activations[layer] or not negative_activations[layer]:
                print(f"No activations captured for layer {layer}, skipping")
                continue
            pos_shapes = [arr.shape for arr in positive_activations[layer]]
            neg_shapes = [arr.shape for arr in negative_activations[layer]]
            if len(set(pos_shapes)) > 1 or len(set(neg_shapes)) > 1:
                print(f"Warning: Inconsistent shapes in layer {layer}, skipping")
                print(f"  Positive shapes: {pos_shapes}")
                print(f"  Negative shapes: {neg_shapes}")
                continue
            try:
                pos_mean = np.mean(np.stack(positive_activations[layer]), axis=0)
                neg_mean = np.mean(np.stack(negative_activations[layer]), axis=0)
                diff = pos_mean - neg_mean
                neuron_indices = np.argsort(np.abs(diff))[-50:]
                concept_neurons[layer] = {
                    'indices': neuron_indices,
                    'sensitivity': diff[neuron_indices]
                }
            except Exception as e:
                print(f"Error processing layer {layer}: {e}")
                continue
        self.behavior_patterns[concept_name] = {
            'type': 'concept_neurons',
            'neurons': concept_neurons
        }
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print(f"Identified concept neurons for: {concept_name}")

    def generate_with_concept_neurons(self, prompt: str,
                                     concept_name: str,
                                     influence_strength: float = 2.0,
                                     max_new_tokens: int = 300,
                                     temperature: float = 0.7,
                                     min_new_tokens: int = 300) -> str:
        """
        Generate text while stimulating specific concept neurons.
        
        Args:
            prompt: Input text to start generation.
            concept_name: Name of the concept neurons to stimulate.
            influence_strength: Strength of the neuron stimulation.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            min_new_tokens: Minimum number of tokens to generate.
            
        Returns:
            Generated text with concept neuron stimulation applied.
        """
        if concept_name not in self.behavior_patterns:
            raise ValueError(f"Concept '{concept_name}' not found. Please capture it first.")
        if self.behavior_patterns[concept_name]['type'] != 'concept_neurons':
            raise ValueError(f"'{concept_name}' is not a concept neuron pattern.")
        concept_info = self.behavior_patterns[concept_name]['neurons']
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        manipulation_hooks = []
        for layer_name, neuron_data in concept_info.items():
            target_layer = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    target_layer = module
                    break
            if target_layer is None:
                print(f"Warning: Layer {layer_name} not found. Skipping.")
                continue
            def make_stimulation_hook(layer_name, neuron_indices, sensitivities):
                def hook(module, input_val, output):
                    if isinstance(output, tuple):
                        orig_output = output[0]
                        is_tuple = True
                    else:
                        orig_output = output
                        is_tuple = False
                    modifier = torch.zeros_like(orig_output)
                    if len(orig_output.shape) == 3:
                        for idx, sensitivity in zip(neuron_indices, sensitivities):
                            if idx < orig_output.shape[2]:
                                modifier[:, :, idx] = influence_strength * sensitivity
                    elif len(orig_output.shape) == 4:
                        for idx, sensitivity in zip(neuron_indices, sensitivities):
                            for head in range(orig_output.shape[1]):
                                modifier[:, head, :, idx % orig_output.shape[3]] = influence_strength * sensitivity
                    manipulated = orig_output + modifier
                    if is_tuple:
                        return (manipulated,) + output[1:]
                    return manipulated
                return hook
            indices = neuron_data['indices']
            sensitivities = neuron_data['sensitivity']
            hook = target_layer.register_forward_hook(
                make_stimulation_hook(layer_name, indices, sensitivities)
            )
            manipulation_hooks.append(hook)
        with hook_manager(manipulation_hooks):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    min_length=input_ids.shape[1] + min_new_tokens,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                    do_sample=True,
                    temperature=temperature
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text