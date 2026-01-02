import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AttentionStep:
    """Records attention information for a single generation step"""

    step: int
    token_id: str
    token: str
    position: int
    attention_weights: list[list[float]]

    def asdict(self):
        return dataclasses.asdict(self)


@dataclass
class GenerationResult:
    """Complete result from a generation with attention tracking"""

    input_text: str
    output_text: str
    input_tokens: list[str]
    output_tokens: list[str]
    attention_steps: list[AttentionStep]
    context_length: int
    response: str = ""
    tokens: list[str] = field(default_factory=list)
    attention_weights: dict = field(default_factory=dict)


class AttentionVisualizationAgent:
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
        self.attention_layer_index = attention_layer_index
        self.verbose = verbose
        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("Initializing %s on %s.", self.model_name, self.device)
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32 if self.device == "cpu" else torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",  # Enable attention output
        ).to(self.device)

        # Determine number of layers
        self.num_layers = self._get_num_layers()
        if self.num_layers:
            logger.info("Model has %d layers", self.num_layers)

        # Initialize attention tracker
        self.tracker = None
        self.conversation_history = []

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
        input_tokens: BatchEncoding = self.tokenizer(prompt, return_tensors="pt", truncation=False).to(self.device)
        context_length = input_tokens["input_ids"].shape[1]
        print()

        pass

    def _get_num_layers(self) -> Optional[int]:
        """Get the number of transformer layers in the model"""
        if hasattr(self.model, "config"):
            for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
                if hasattr(self.model.config, attr):
                    return getattr(self.model.config, attr)
        return None


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
    saved_files = []
    for i, (prompt, category) in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}: {category} ---")
        print(f"Prompt: {prompt}")
        result = agent.generate_with_attention(
            prompt, max_new_tokens=100, temperature=0.7, save_trajectory=True, category=category
        )


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
