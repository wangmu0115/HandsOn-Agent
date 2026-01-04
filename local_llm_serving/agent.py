import sys
from typing import Iterator, Literal, Union, overload

import ollama
from loggers import setup_logger
from ollama_native import OllamaNativeAgent
from outputs import ChatGenerationChunk

logger = setup_logger(__name__, "WARNING")


class ToolCallingAgent:
    """
    Universal tool calling agent that works on all platforms
    """

    def __init__(self, backend: Literal["vllm", "ollama"] = "ollama", model: str = "qwen3:0.6b"):
        self.model = model
        self.backend_type = backend
        self.agent = None
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
    ) -> Iterator[ChatGenerationChunk]: ...

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
    ) -> Union[str | Iterator[ChatGenerationChunk]]:
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

    def _initialize_backend(self):
        if self.backend_type == "vllm":
            self._init_vllm()
        else:
            self._init_ollama()

    def _init_vllm(self):  # TODO
        raise NotImplementedError("vLLM not supported")

    def _init_ollama(self):
        success_inited = True
        try:
            client = ollama.Client()
            models_data = client.list()
            available_models = [m.model for m in models_data.models] if hasattr(models_data, "models") else None
            if not available_models or self.model not in available_models:
                success_inited = False
                logger.warning("Available models: %s, recommended model: %s", available_models, self.model)
            else:
                self.agent = OllamaNativeAgent(self.model)
        except Exception as e:
            success_inited = False
            logger.exception("Ollama is not running: ", e)

        if not success_inited:
            sys.exit(1)
