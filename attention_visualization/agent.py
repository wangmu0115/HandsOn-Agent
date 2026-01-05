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
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False).to(self.device)
        input_tokens = inputs.tokens()
        # Initialize tracker
        self.tracker = AttentionTracker(self.tokenizer, len(input_tokens), self.verbose)

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
            logger.info("Registered %d attention hooks", len(hook))

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
