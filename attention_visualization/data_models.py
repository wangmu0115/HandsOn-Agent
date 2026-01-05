import dataclasses
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import LogitsProcessor


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

    def __post_init__(self):
        if not self.tokens:
            self.tokens = self.input_tokens + self.output_tokens
        if not self.response:
            self.response = self.output_text

    def asdict(self):
        return dataclasses.asdict(self)


class AttentionTracker(LogitsProcessor):
    def __init__(self, tokenizer, context_length: int, verbose: bool = False):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.verbose = verbose
        self.attention_cache = {}
        self.generation_step = 0
        self.generated_tokens = []
        self.output_only = True  # Only track attention from output tokens

    def reset(self):
        """Reset tracker for new generation"""
        self.attention_cache = {}
        self.generation_step = 0
        self.generated_tokens = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Called during generation to track tokens"""
        self.generation_step += 1

        # Track generated token
        if input_ids.shape[1] > self.context_length:
            last_token_id = input_ids[0, -1].item()
            last_token = self.tokenizer.decode([last_token_id])
            current_position = input_ids.shape[1] - 1

            self.generated_tokens.append({"step": self.generation_step, "token_id": last_token_id, "token": last_token, "position": current_position})

            if self.verbose:
                print(f"  Step {self.generation_step}: Generated '{last_token}' at position {current_position}")

        return scores

    def update_attention(self, position: int, attention_weights):
        """Store attention weights for a position (only for output tokens)"""
        # Only store attention for output tokens (positions >= context_length)
        if self.output_only and position < self.context_length:
            return  # Skip input token attention
        self.attention_cache[position] = attention_weights

    def get_attention_steps(self) -> list[AttentionStep]:
        """Convert cached data into AttentionStep objects"""
        steps = []
        for token_info in self.generated_tokens:
            position = token_info["position"]
            if position in self.attention_cache:
                attention = self.attention_cache[position]
                if isinstance(attention, torch.Tensor):
                    attention = attention.cpu().numpy().tolist()
                elif isinstance(attention, np.ndarray):
                    attention = attention.tolist()

                steps.append(
                    AttentionStep(
                        step=token_info["step"], token_id=token_info["token_id"], token=token_info["token"], position=position, attention_weights=attention
                    )
                )
        return steps
