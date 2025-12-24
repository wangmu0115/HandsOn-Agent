import functools
import itertools
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Optional

from ollama_native import OllamaNativeAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ToolCallingAgent:
    """
    Universal tool calling agent that works on all platforms
    Automatically selects vLLM (if GPU available) or Ollama
    """

    def __init__(self, backend: Optional[str] = None, model: str = "qwen3:0.6b"):
        """
        Initialize with automatic backend detection

        Args:
            backend: Force a specific backend ('vllm', 'ollama', or None for auto)
        """
        self.model = model
        self.agent = None
        self.backend_type = backend or self._detect_best_backend()
        self._initialize_backend()

    def _detect_best_backend(self) -> str:
        system = platform.system()
        if system in ["Linux", "Windows"]:
            try:
                import torch

                if torch.cuda.is_available():
                    logger.info(f"CUDA detected on {system} - will use vLLM")
                    return "vllm"

            except ImportError:
                pass
        logger.info(f"Using Ollama on {system}")
        return "ollama"

    def _initialize_backend(self):
        if self.backend_type == "vllm":
            self._init_vllm()
        else:
            self._init_ollama()

    def _init_vllm(self):
        pass

    def _init_ollama(self):
        try:
            import ollama

            client = ollama.Client()  # Check if Ollama is running
            try:
                models_response = client.list()
                available_models = []
                if hasattr(models_response, "models"):
                    available_models = [m.model for m in models_response.models]
                if not available_models:
                    logger.error("No Ollama models installed")
                    sys.exit(1)
                if self.model not in available_models:
                    logger.warning(f"Recommended model {self.model} not found in available models")
                    sys.exit(1)
                self.agent = OllamaNativeAgent(self.model)
            except Exception as e:
                logger.error(f"Ollama is not running: {e}")
                sys.exit(1)
        except ImportError:
            logger.error("Ollama not installed, install with: `pip install ollama`")
            sys.exit(1)

    def chat(self, message: str, use_tools: bool = True, stream: bool = False, **kwargs) -> str:
        """
        Send a message to the agent

        Args:
            message: User message
            use_tools: Whether to enable tool calling
            stream: Whether to stream the response
            **kwargs: Additional backend-specific parameters

        Returns:
            Agent response (or generator if streaming)
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized.")
        return self.agent.chat(message, use_tools=use_tools, stream=stream, **kwargs)


def get_sample_tasks() -> list[dict[str, str]]:
    """Get sample tasks for demon."""
    script_path = Path(__file__).resolve()
    tasks_file_path = script_path.parent / "sample_tasks.json"
    with open(tasks_file_path) as f:
        content = f.read()
        return json.loads(content)


def run_single_task(agent: ToolCallingAgent, task: str, stream: bool = True):
    """Run a single task with optional streaming."""
    print("\n" + "=" * 80)
    print("TASK EXECUTION")
    print("=" * 80)
    print(f"\n📋 Task: {task}")
    print("-" * 80)
    if stream:
        pass
    else:
        print("\n⏳ Processing...")
        response = agent.chat(task, use_tools=True, stream=False)
        print("\n✅ Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)


def interactive_mode(agent: ToolCallingAgent, stream: bool = True):
    """Run interactive chat mode with optional streaming."""
    pass


def main():
    args = _add_args().parse_args()  # Get arguments

    # Header
    print("=" * 80)
    print("🚀 Universal Tool Calling Agent")
    print("=" * 80)
    if args.info:  # Show system info and exist
        _show_system_info()
        return 0
    # Initialize agent
    print("\n⚙️  Initializing agent...")
    backend = None if args.backend == "auto" else args.backend
    try:
        agent = ToolCallingAgent(backend=backend)
    except SystemExit:
        return 1
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return 1
    print(f"✅ Agent ready! Using {agent.backend_type} backend")

    # All sample tasks
    sample_tasks = get_sample_tasks()

    # Execute based on mode
    if args.mode == "single":
        if not args.task:
            print("\n" + "=" * 80)
            print("SINGLE TASK MODE - No task provided")
            print("=" * 80)
            print("\n📋 Available sample tasks:")
            for i, sample in enumerate(sample_tasks, 1):
                print(f"\n{i}. {sample['name']}")
                print(f"   {sample['description']}")
            print("\n" + "=" * 80)
            try:
                choice = input(f"\nSelect a task number (1-{len(sample_tasks)}) or 'q' to quit: ").strip()
                if choice == "q":
                    return 0
                task_num = int(choice)
                if 1 <= task_num <= len(sample_tasks):
                    selected_task = sample_tasks[task_num - 1]
                    print(f"\n✅ Selected: {selected_task['name']}")
                    print("\nTask details:")
                    print("-" * 60)
                    print(selected_task["task"])
                    print("-" * 60)
                    confirm = input("\nRun this task? (y/n): ").strip().lower()
                    if confirm == "y":
                        stream_enabled = not hasattr(args, "no-stream")
                        run_single_task(agent, selected_task["task"], stream=stream_enabled)
                    else:
                        print("Task cancelled.")
                        return 0
                else:
                    print(f"Invalid selection. Please choose 1-{len(sample_tasks)}")
                    return 1
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                return 0
        else:
            mapped_task = list(filter(lambda t: t["name"] == args.task, sample_tasks))
            if len(mapped_task) > 0:
                stream_enabled = not hasattr(args, "no-stream")
                run_single_task(agent, mapped_task["task"], stream=stream_enabled)
                return 0
            else:
                print(f"Unknown task name: {args.task}")
                return 0
    else:  # interactive mode
        stream_enabled = not hasattr(args, "no-stream")
        interactive_mode(agent, stream_enabled)
        return 0


def _add_args():
    import argparse

    parser = argparse.ArgumentParser(description="Universal Tool Calling Agent - Works on all platforms")
    parser.add_argument("--mode", choices=["single", "interactive"], default="interactive", help="Execution mode (default: interactive)")
    parser.add_argument("--task", type=str, help="Task to execute (for single mode)")
    parser.add_argument("--backend", choices=["vllm", "ollama", "auto"], default="auto", help="Backend to use (default: auto-detect)")
    parser.add_argument("--info", action="store_true", help="Show system information and exit")
    parser.add_argument("--stream", action="store_true", default=True, help="Enable streaming mode (default: True)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming mode")
    return parser


def _show_system_info():
    print("\nSystem Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python: {sys.version.split()[0]}")
    try:  # Check CUDA
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"  CUDA: ✅ Available (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("  CUDA: ❌ Not available")
    except ImportError:
        print("  CUDA: ❌ PyTorch not installed")
    try:
        import ollama  # noqa: F401

        print("  Ollama: ✅ Package installed")
    except ImportError:
        print("  Ollama: ❌ Package not installed")


if __name__ == "__main__":
    sys.exit(main())
