# Neural State Manipulator

A Python library for recording and manipulating the internal neural activations of language models to control their generation behavior.

This is an alternative for representation engineering and folks no longer need to fine-tune or prompt a model to explicitly control a Large Language Model's generation behavior.

## Description

Neural State Manipulator allows researchers and developers to:

1. Identify and access internal layers in transformer-based language models
2. Record activation patterns associated with specific behaviors or styles
3. Apply these patterns during generation to influence model outputs
4. Identify and manipulate concept-specific neurons

This package is useful for research in interpretability, controlled text generation, and model behavior steering.

## Use Cases

### Behavioral Steering
* **Trait Modulation**: Adjust abstract traits like honesty, refusal tendencies, or creativity by applying activation patterns derived from contrasting examples.
* **Role-Playing**: Induce specific personas (e.g., a helpful assistant vs. a sarcastic character) by steering hidden states toward predefined behavioral patterns.

### Factual and Contextual Adjustment
* **Controlled Fact Editing**: Alter memorized facts by manipulating internal neural activations. This modifies the model's internal representations, affecting downstream responses.
* **Contextual Consistency**: Ensure responses align with specific themes (e.g., happiness or urgency) by injecting activation patterns at inference time.

### Safety and Alignment
* **Refusal Behavior Control**: Increase or decrease the likelihood of a model refusing requests, useful for content moderation or ethical guardrails.
* **Trait Validation**: Test whether a model distinguishes between genuine and feigned traits (e.g., "honesty" vs. "feigned honesty") by comparing activation patterns, enhancing transparency.

### Efficiency and Flexibility
* **Low-Cost Adaptation**: Avoid resource-intensive fine-tuning by applying lightweight neural state manipulation during inference. This is particularly useful for edge deployments or scenarios requiring rapid iteration.
* **Coefficient Tuning**: Adjust the strength of a trait (e.g., dialing humor from subtle to exaggerated) via scalar coefficients attached to activation patterns.

## Installation

```bash
pip install neural-state-manipulator
```

## Requirements

- Python 3.8+
- PyTorch 2.0.0+
- Transformers 4.30.0+
- Numpy 1.20.0+
- BitsAndBytes 0.40.0+

## Quick Start

Here's a simple example to get you started:

```python
from neural_state_manipulator import NeuralStateManipulator, list_manipulable_layers

# Optional: List manipulable layers in a model
layers = list_manipulable_layers("meta-llama/Llama-3.2-1B-Instruct")

# Initialize the manipulator with your model
manipulator = NeuralStateManipulator("meta-llama/Llama-3.2-1B-Instruct")

# Select specific layers for manipulation
# prefer final layers for example if a model has 30 layers, use layers 28, 29, 30 and same for any layer and here I am using the 15th layer since the model has only 16 layers
target_layers = ['model.layers.15.post_attention_layernorm', 'model.layers.15.mlp.gate_proj']

# Record behavior patterns
formal_text = """
The implementation of artificial intelligence in industrial settings has been
demonstrated to enhance operational efficiency by an average of 27%, according to
recent comprehensive analyses. The methodology employed in these studies involved
rigorous examination of productivity metrics across diverse manufacturing environments.
"""
manipulator.record_behavior_pattern(formal_text, "formal_writing", target_layers)

# Generate text with the formal pattern applied
prompt = "Write about climate change:"
output = manipulator.generate_with_manipulation(
    prompt, "formal_writing", influence_strength=0.5, 
    max_new_tokens=300, temperature=0.7
)
print(output)
```

## Advanced Usage

### Recording Different Behavior Patterns

```python
# Record a creative writing style
creative_text = """
Sunlight danced through the leaves, painting the forest floor with shimmering gold.
The ancient trees whispered secrets to one another, their roots intertwined beneath
the soil like old friends holding hands. A fox, clever and quick, darted between shadows.
"""
manipulator.record_behavior_pattern(creative_text, "creative_writing", target_layers)

# Generate with the creative pattern
creative_output = manipulator.generate_with_manipulation(
    prompt, "creative_writing", influence_strength=0.5,
    max_new_tokens=300, temperature=0.7
)
```

### Identifying and Using Concept Neurons

```python
# Define examples of technical and non-technical text
technical_texts = [
    "The algorithm complexity is O(n log n) in the average case.",
    "Quantum computing utilizes qubits rather than classical bits.",
    "The framework implements dependency injection via constructor parameters."
]
non_technical_texts = [
    "The sunset painted the sky with brilliant colors.",
    "She enjoyed her coffee while reading the morning newspaper.",
    "The dog chased the ball across the grassy field."
]

# Identify neurons associated with technical language
manipulator.capture_concept_neurons(
    technical_texts, non_technical_texts, "technical_concept"
)

# Generate text with technical concept neurons stimulated
technical_output = manipulator.generate_with_concept_neurons(
    "Explain how solar panels work:", "technical_concept", 
    influence_strength=0.5, max_new_tokens=300
)
```

### Other Manipulation Methods

```python
# Generate without manipulation (baseline)
plain_output = manipulator.generate_plain(prompt, max_new_tokens=300)

# Generate with erasure (subtracting a behavior pattern)
erasure_output = manipulator.generate_with_erasure(
    prompt, "formal_writing", erasure_strength=0.5, 
    target_layers=target_layers, max_new_tokens=300
)

# Generate with amplification (enhancing a behavior pattern)
amplified_output = manipulator.generate_with_amplification(
    prompt, "creative_writing", amplification_factor=0.5,
    target_layers=target_layers, max_new_tokens=300
)

# Generate with interpolation (blending baseline and behavior)
interpolated_output = manipulator.generate_with_interpolation(
    prompt, "formal_writing", interpolation_factor=0.5,
    target_layers=target_layers, max_new_tokens=300
)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```
@software{neural_state_manipulator,
  author = {Your Name},
  title = {Neural State Manipulator: A Tool for Manipulating Internal Neural Activations of Language Models},
  year = {2025},
  url = {https://github.com/yourusername/neural-state-manipulator}
}
```