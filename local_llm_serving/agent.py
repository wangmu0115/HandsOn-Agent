import logging
import platform
import sys
from typing import Iterator, Literal, Union, overload

import ollama
import torch
from ollama_native import ChatResponseChunk, OllamaNativeAgent

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ToolCallingAgent:
    """
    Universal tool calling agent that works on all platforms
    Automatically selects vLLM (if GPU available) or Ollama
    """

    def __init__(self, backend: Union[Literal["vllm", "ollama"] | None] = None, model: str = "qwen3:0.6b"):
        self.model = model
        self.agent = None
        self.backend_type = backend or self._choose_best_backend()
        self._initialize_backend()

    def reset_conversation(self):
        """Reset conversation history"""
        if hasattr(self.agent, "reset_conversation"):
            self.agent.reset_conversation()

    @overload
    def chat(
        self,
        message: str,
        *,
        use_tools: bool = True,
        stream: Literal[True] = True,
        **kwargs,
    ) -> Iterator[ChatResponseChunk]: ...

    @overload
    def chat(
        self,
        message: str,
        *,
        use_tools: bool = True,
        stream: Literal[False] = False,
        **kwargs,
    ) -> str: ...

    def chat(
        self,
        message: str,
        *,
        use_tools: bool = True,
        stream: bool = False,
        **kwargs,
    ) -> Union[str | Iterator[ChatResponseChunk]]:
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

    def _choose_best_backend(self) -> str:
        system = platform.system()
        if system in ["Linux", "Windows"]:
            try:
                if torch.cuda.is_available():
                    logger.info(f"CUDA detected on {system} - will use vLLM")
                    return "vllm"
            except Exception:
                pass
        logger.info(f"Using Ollama on {system}")
        return "ollama"

    def _initialize_backend(self):
        if self.backend_type == "vllm":
            self._init_vllm()
        else:
            self._init_ollama()

    def _init_vllm(self):  # TODO
        raise NotImplementedError("vLLM not supported")

    def _init_ollama(self):
        try:
            client = ollama.Client()
            models_data = client.list()
            available_models = [m.model for m in models_data.models] if hasattr(models_data, "models") else None
            if not available_models:  # None or empty
                logger.warning("No Ollama models installed.")
                sys.exit(1)
            if self.model not in available_models:
                logger.warning(f"Recommended model {self.model} not found in available models.")
                sys.exit(1)

            self.agent = OllamaNativeAgent(self.model)
        except Exception as e:
            logger.exception("Ollama is not running: %s", e)
            sys.exit(1)
