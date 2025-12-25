import json
import logging

import ollama
from tools import ToolRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OllamaNativeAgent:
    """Agent using Ollama's native tool calling support"""

    def __init__(self, model: str):
        self.model = model
        self.client = ollama.Client()
        self.tool_register = ToolRegistry()
        self.conversation_history = []

    def chat(self, message: str, *, use_tools: bool = True, stream: bool = True, temperature: float = 0.7) -> str:
        """Send a message using Ollama's native tool calling.

        Args:
            message: User message
            use_tools: Whether to enable tool calling
            stream: Whether to stream the response
            temperature: Sampling temperature

        Returns:
            Final response from the model (or generator if streaming)
        """
        if stream:
            return self.chat_stream(message, use_tools, temperature)

        # Add use message to history
        self.conversation_history.append({"role": "user", "content": message})
        # Tools
        tools = self.tool_register.get_tool_schemas() if use_tools else None
        try:
            # Call Ollama with tools
            resp = self.client.chat(self.model, self.conversation_history, tools=tools, options={"temperature": temperature})
            message_content = resp.message
            tool_calls = message_content.tool_calls
            if tool_calls is not None and len(tool_calls) > 0:
                logger.info(f"Model requested {len(tool_calls)} tool call(s).")
                # Add assistant's message with tool calls to history
                self.conversation_history.append(
                    {
                        "role": "assistant",
                        "content": message_content.content or "",
                        "tool_calls": tool_calls,
                    }
                )
                # Execute each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    result = self.tool_register.execute_tool(tool_name, tool_args)
                    # Add tool result to conversation
                    self.conversation_history.append(
                        {
                            "role": "tool",
                            "content": result,
                        }
                    )

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error: {e}"

    def chat_stream(self, message: str, use_tools: bool = True, temperature: float = 0.7):
        """Stream a message to the model and handle tool calls in a ReAct loop.

        Yields chunks that include:
        - type: 'thinking', 'tool_call', 'tool_result', 'content'
        - content: The actual content
        """

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
