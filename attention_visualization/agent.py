import time

import torch
from athena_core import PostInitMeta, setup_logger
from data_models import AttentionTracker, GenerationResult
from trajectory_storage import TrajectoryStore
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList

logger = setup_logger(__name__, "INFO")


class AttentionVisualizationAgent(metaclass=PostInitMeta):
    """
    Agent that generates text using Qwen3 0.6B while tracking attention weights
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", attention_layer_index: int = -1, verbose: bool = True):
        """
        Initialize the agent with Qwen3 model

        Args:
            model_name: Hugging Face model name
            attention_layer_index: Which layer's attention to track (-1 for last)
            verbose: Whether to print debug info
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.attention_layer_index = attention_layer_index
        self.verbose = verbose
        logger.info("Initializing %s on %s", self.model_name, self.device)

    def __post_init__(self):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32 if self.device == "cpu" else torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",  # Enable attention output
        ).to(self.device)
        # Determine number of layers
        num_layers = None
        if hasattr(self.model, "config"):
            for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
                if hasattr(self.model.config, attr):
                    num_layers = getattr(self.model.config, attr)
                    logger.info("Model has %d layers", num_layers)
                    break
        self.num_layers = num_layers

        # Initialize attention tracker
        self.tracker = None
        self.conversation_history = []
        # Initialize tracjectory storage
        self.trajectory_store = TrajectoryStore(self.model_name, self.device)

    def _capture_attention_hook(self, module, input, output):
        """Hook to capture attention weights from model layers"""
        if self.tracker is None:
            return

        try:
            attention_weights = None

            # Try different ways to extract attention
            if hasattr(output, "attentions") and output.attentions is not None:
                attention_weights = output.attentions
            elif isinstance(output, tuple) and len(output) > 1:
                for item in output:
                    if isinstance(item, torch.Tensor) and len(item.shape) == 4:
                        attention_weights = item
                        break

            if attention_weights is not None:
                # Handle multiple layers
                if isinstance(attention_weights, (list, tuple)):
                    layer_idx = self.attention_layer_index
                    if layer_idx >= 0 and layer_idx < len(attention_weights):
                        attention_weights = attention_weights[layer_idx]
                    else:
                        attention_weights = attention_weights[-1]  # Default to last

                # Extract attention for last token
                if isinstance(attention_weights, torch.Tensor) and attention_weights.dim() >= 3:
                    if attention_weights.dim() == 4:
                        # Average across heads: [batch, heads, seq, seq] -> [seq]
                        avg_attention = attention_weights[0, :, -1, :].mean(dim=0)
                    else:
                        avg_attention = attention_weights[0, -1, :]

                    current_pos = avg_attention.shape[0] - 1

                    # Only track attention for output tokens
                    if current_pos >= self.tracker.context_length:
                        self.tracker.update_attention(current_pos, avg_attention)

        except Exception as e:
            if self.verbose:
                logger.warning(f"Error in attention hook: {e}")

    def _process_generation_attentions(self, attentions, context_length):
        """Process attention weights from generation output"""
        if not attentions or not self.tracker:
            return

        try:
            for step_idx, step_attentions in enumerate(attentions):
                if step_attentions is None or len(step_attentions) == 0:
                    continue

                # Select layer
                layer_index = self.attention_layer_index
                if layer_index >= 0 and layer_index < len(step_attentions):
                    selected_attention = step_attentions[layer_index]
                elif layer_index < 0 and abs(layer_index) <= len(step_attentions):
                    selected_attention = step_attentions[layer_index]
                else:
                    selected_attention = step_attentions[-1]

                if isinstance(selected_attention, torch.Tensor):
                    # Get attention for last position
                    current_seq_len = selected_attention.shape[2]
                    last_pos = current_seq_len - 1

                    # Average across heads
                    avg_attention = selected_attention[0, :, last_pos, :].mean(dim=0)

                    # Store in tracker
                    seq_pos = context_length + step_idx
                    self.tracker.update_attention(seq_pos, avg_attention)

        except Exception as e:
            if self.verbose:
                logger.warning(f"Error processing generation attentions: {e}")

    def generate_with_attention(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        save_trajectory: bool = True,
        category: str = "General",
        store_full_tokens: bool = True,
    ) -> GenerationResult:
        """
        Generate text while tracking attention weights

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            store_full_tokens: Whether to store all input tokens (not truncated)

        Returns:
            GenerationResult with tokens and attention information
        """
        # Tokenize input without truncation to preserve all tokens
        inputs = self.tokenizer([prompt], return_tensors="pt", truncation=False).to(self.device)
        input_token_ids = inputs.input_ids[0].tolist()
        input_tokens = [self.tokenizer.decode(token_id, skip_special_tokens=False) for token_id in input_token_ids]
        context_length = len(input_token_ids)
        logger.info("Input %d tokens: %s", context_length, input_tokens)

        # Initialize tracker
        self.tracker = AttentionTracker(self.tokenizer, context_length, self.verbose)

        # Set up generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            repetition_penalty=1.1,
        )

        # Register attention hooks
        hooks = []
        hook_modules = []

        # Find attention modules
        for name, module in self.model.named_modules():
            if any(pattern in name.lower() for pattern in ["attn", "attention", "self_attn"]):
                if hasattr(module, "forward"):
                    hook = module.register_forward_hook(self._capture_attention_hook)
                    hooks.append(hook)
                    hook_modules.append(name)
        if self.verbose:
            logger.info("Registered %d attention hooks", len(hooks))

        try:
            # Generate with attention tracking
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    logits_processor=LogitsProcessorList([self.tracker]),
                    output_attentions=True,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Process attention from generate output if available
            if hasattr(outputs, "attentions") and outputs.attentions is not None:
                self._process_generation_attentions(outputs.attentions, len(input_tokens))

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        # Decode output
        generated_ids = outputs.sequences[0][context_length:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # Keep special tokens in token list for accurate representation
        output_tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in generated_ids.tolist()]

        # Get attention steps
        attention_steps = self.tracker.get_attention_steps()

        logger.info(f"Generated {len(output_tokens)} tokens with {len(attention_steps)} attention steps")

        # Store all tokens (input + output) for complete sequence
        all_token_ids = outputs.sequences[0].tolist()
        all_tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in all_token_ids]

        result = GenerationResult(
            input_text=prompt,
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens=all_tokens,  # Complete token sequence
            attention_steps=attention_steps,
            context_length=context_length,
        )

        # Save trajectory if requested
        if save_trajectory:
            self.trajectory_store.save(generation_config, result, query=prompt, category=category)

        return result

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []


def demonstrate_attention_tracking():
    """Demonstrate the attention tracking functionality"""
    print("=" * 80)
    print("Attention Visualization Demo")
    print("=" * 80)

    # Initialize agent
    agent = AttentionVisualizationAgent()

    test_prompts = [
        ("What is the capital of France?", "Knowledge"),
        ("Calculate 25 * 4 + 10", "Math"),
        ("Write a haiku about spring", "Creative"),
        ("If all cats are animals, and some animals are pets, can we conclude that all cats are pets?", "Reasoning"),
        ("Write a Python function to calculate factorial", "Code"),
    ]

    results = []
    for i, (prompt, category) in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}: {category} ---")
        print(f"Prompt: {prompt}")
        result = agent.generate_with_attention(
            prompt,
            max_new_tokens=100,
            temperature=0.7,
            save_trajectory=True,
            category=category,
        )
        print(f"Response: {result.output_text}")
        print(f"Input tokens: {len(result.input_tokens)}")
        print(f"Output tokens: {len(result.output_tokens)}")
        print(f"Attention steps tracked: {len(result.attention_steps)}")

        results.append(result)
        time.sleep(1)  # Ensure unique timestamps
    return results


if __name__ == "__main__":
    results = demonstrate_attention_tracking()

    print("\n" + "*" * 80)
    print("✨ Demo Complete!")
    print("\n🌐 To view the visualizations:")
    print("   1. cd frontend")
    print("   2. npm install (if not already done)")
    print("   3. npm run dev")
    print("   4. Open http://localhost:3000")
    print("\n💾 Trajectories saved to frontend/public/trajectories/")
    print("*" * 80)
