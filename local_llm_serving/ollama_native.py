import logging

import ollama
from tools import ToolRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OllamaNativeAgent:
    """"""

    def __init__(self, model: str):
        self.model = model
        self.client = ollama.Client()
        self.tool_register = ToolRegistry()
        self.conversation_history = []
        self._check_ollama_serve()  # Check if Ollama is running

    def chat(self, message: str, use_tools: bool = True, temperature: float = 0.7, stream: bool = True) -> str:
        """Send a message using Ollama's native tool calling.
        Args:
            message: User message
            use_tools: Whether to enable tool calling
            temperature: Sampling temperature
            stream: Whether to stream the response

        Returns:
            Final response from the model (or generator if streaming)
        """
        if stream:
            return self.chat_stream(message, use_tools, temperature)

        # Add use message to history
        self.conversation_history.append({"role": "user", "content": message})

        tools = self.tool_register.get_tool_schemas() if use_tools else None  # Prepare tools
        resp = self.client.chat(
            self.model,
            self.conversation_history,
            tools=tools,
            options={"temperature": temperature},
        )
        print(resp)

    def chat_stream(self, message: str, use_tools: bool = True, temperature: float = 0.7):
        """Stream a message to the model and handle tool calls in a ReAct loop.

        Yields chunks that include:
        - type: 'thinking', 'tool_call', 'tool_result', 'content'
        - content: The actual content
        """

    def _check_ollama_serve(self):
        try:
            self.client.list()
            logger.info(f"Connected to Ollama with model: {self.model}.")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
