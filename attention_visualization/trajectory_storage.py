import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from athena_core import PostInitMeta
from data_models import GenerationResult
from transformers import GenerationConfig


class TrajectoryStore(metaclass=PostInitMeta):
    def __init__(self, model_name: str, device: str, output_dir: str = "frontend/public/trajectories"):
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, config: GenerationConfig, result: GenerationResult, query: Optional[str] = None, category: str = "General") -> str:
        """Save a trajectory to frontend/public/ with unique filename"""
        now_dt = datetime.now()
        unique_id = now_dt.strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"trajectory_{unique_id}.json"

        # Extract attention data for visualization (output tokens only)
        attention_matrix = []
        if result.attention_steps:
            for step in result.attention_steps:
                if step.attention_weights:
                    attention_matrix.append(step.attention_weights)

        # Build trajectory data for frontend
        trajectory_data = {
            "id": unique_id,
            "timestamp": now_dt.strftime("%Y:%m:%d %H:%M:%S"),
            "test_case": {
                "category": category,
                "query": query or result.input_text,
                "description": f"Agent trajectory from {now_dt.strftime('%Y-%m-%d %H:%M:%S')}",
            },
            "response": result.output_text,
            "tokens": result.tokens,
            "attention_data": {
                "tokens": result.tokens,
                "attention_matrix": attention_matrix,
                "num_layers": 1,  # Simplified for now
                "num_heads": len(attention_matrix[0]) if attention_matrix and attention_matrix[0] else 0,
                "output_only": True,  # Flag to indicate output-only attention
                "context_length": result.context_length,  # Where output tokens start
            },
            "metadata": {
                "model": self.model_name,
                "temperature": config.temperature,
                "max_tokens": config.max_new_tokens,
                "device": self.device,
                "attention_type": "output_only",  # Clarify attention type
            },
        }

        # Save the trajectory data to file
        with open(filename, "w") as f:
            json.dump(trajectory_data, f, indent=2, default=str)

        manifest_file = self.output_dir / "manifest.json"
        all_manifests = []
        if manifest_file.exists():
            try:
                with open(manifest_file, "r") as f:
                    all_manifests = json.load(f)
            except:  # noqa: E722
                ...
        all_manifests.append(
            {
                "filename": f"trajectory_{unique_id}.json",
                "id": unique_id,
                "timestamp": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "category": category,
                "query": query or result.input_text,
            }
        )
        # Keep only last 50 trajectories in manifest
        all_manifests = all_manifests[-50:]
        with open(manifest_file, "w") as f:
            json.dump(all_manifests, f, indent=2, default=str)
        return str(filename)
