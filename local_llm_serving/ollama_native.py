from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Literal, Optional, Sequence, Union

import ollama
from ollama import Message
from tools import ToolRegistry

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Chat:
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: Sequence[Message.ToolCall] = field(default_factory=list)

    def asdict(self):
        d = {"role": self.role, "content": self.content}
        if self.tool_calls is not None and len(self.tool_calls) > 0:
            d["tool_calls"] = self.tool_calls
        return d

    @classmethod
    def user(cls, content: str) -> Chat:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str, tool_calls: Optional[Sequence[Message.ToolCall]] = None) -> Chat:
        chat = cls("assistant", content)
        if tool_calls:
            chat.tool_calls = tool_calls
        return chat

    @classmethod
    def tool(cls, content: str) -> Chat:
        return cls("tool", content)


@dataclass
class ChatResponseChunk:
    type: Literal["answer", "thinking", "tool_call", "tool_result", "error"]
    content: str | dict

    @classmethod
    def answer(cls, content: str):
        return cls("answer", content)

    @classmethod
    def thinking(cls, content: str):
        return cls("thinking", content)

    @classmethod
    def tool_call(cls, tool_name: str, tool_args: dict[str, any]):
        return cls("tool_call", {"name": tool_name, "args": tool_args})

    @classmethod
    def tool_result(cls, tool_result: str):
        return cls("tool_result", tool_result)

    @classmethod
    def error(cls, content: str | Exception):
        return cls("error", str(content))


class OllamaNativeAgent:
    """Agent using Ollama's native tool calling support"""

    def __init__(self, model: str):
        self.model = model
        self.client = ollama.Client()
        self.tool_register = ToolRegistry()
        self.conversation_history = []

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []

    def chat(
        self,
        message: str,
        *,
        use_tools: bool = True,
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Union[str | Iterator[ChatResponseChunk]]:
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
            return self.chat_stream(message, use_tools=use_tools, temperature=temperature)

        # Add use message to history
        self.conversation_history.append(Chat.user(message).asdict())
        tools = self.tool_register.get_tool_schemas() if use_tools else None

        try:
            # Call Ollama with tools
            resp = self.client.chat(
                self.model,
                self.conversation_history,
                tools=tools,
                options={"temperature": temperature},
            )

            message_content = resp.message
            tool_calls = message_content.tool_calls
            if tool_calls is not None and len(tool_calls) > 0:  # Handle tool calls
                logger.info(f"Model requested {len(tool_calls)} tool call(s).")
                # Add assistant's message with tool calls to history
                self.conversation_history.append(Chat.assistant(message_content.thinking or "", tool_calls).asdict())
                for tool_call in tool_calls:
                    name = tool_call.function.name
                    args = tool_call.function.arguments
                    logger.info(f"Executing tool: {name} with args: {args}")
                    result = self.tool_register.execute_tool(name, args)
                    # Add tool result to history
                    self.conversation_history.append(Chat.tool(result).asdict())

                final_resp = self.client.chat(
                    self.model,
                    self.conversation_history,
                    tools=tools,
                    options={"temperature": temperature},
                )
                final_message = final_resp.message
                # Add final response to history
                self.conversation_history.append(Chat.assistant(final_message.content).asdict())
                return final_message.content
            else:  # No tool calls
                content = message_content.content
                self.conversation_history.append(Chat.assistant(content).asdict())
                return content
        except Exception as e:
            logger.exception("Error in chat: %s", e)
            return f"Error: {e}"

    def chat_stream(
        self,
        message: str,
        *,
        use_tools: bool = True,
        temperature: float = 0.7,
    ) -> Iterator[ChatResponseChunk]:
        """Stream a message to the model and handle tool calls in a ReAct loop.

        Yields chunks that include:
        - type: 'thinking', 'tool_call', 'tool_result', 'content'
        - content: The actual content
        """
        # Add use message to history
        self.conversation_history.append(Chat.user(message).asdict())
        tools = self.tool_register.get_tool_schemas() if use_tools else None

        # ReAct loop, and Prevent infinite loops
        max_iterations, iteration = 10, 0
        while iteration < max_iterations:
            iteration += 1
            try:
                stream_response = self.client.chat(
                    self.model,
                    self.conversation_history,
                    tools=tools,
                    options={"temperature": temperature},
                    stream=True,
                )

                collected_content = []  # Collect chunk content or thinking
                answered = False
                # Process the stream
                for chunk in stream_response:
                    chunk_message = chunk.message
                    content, thinking, tool_calls = chunk_message.content, chunk_message.thinking, chunk_message.tool_calls
                    if content is not None and content != "":  # Get answered, after chunks exit the ReAct loop
                        answered = True
                        collected_content.append(content)
                        yield ChatResponseChunk.answer(content)
                    elif thinking is not None and thinking != "":  # Thinking
                        collected_content.append(thinking)
                        yield ChatResponseChunk.thinking(thinking)
                    elif tool_calls is not None and len(tool_calls) > 0:  # ToolCalls
                        # Add assistant's message with tool calls to history
                        self.conversation_history.append(Chat.assistant("".join(collected_content), tool_calls).asdict())
                        for tool_call in tool_calls:
                            name = tool_call.function.name
                            args = tool_call.function.arguments
                            yield ChatResponseChunk.tool_call(name, args)
                            result = self.tool_register.execute_tool(name, args)
                            yield ChatResponseChunk.tool_result(result)
                            # Add tool result to history
                            self.conversation_history.append(Chat.tool(result).asdict())
                    else:  # Otherwise, just break
                        break

                if answered:  # Exit the ReAct loop
                    self.conversation_history.append(Chat.assistant("".join(collected_content)).asdict())
                    break
            except Exception as e:
                logger.exception("Error in chat stream: %s", e)
                yield ChatResponseChunk.error(e)
                break
        if iteration >= max_iterations:
            yield ChatResponseChunk.error("Maximum iterations reached in ReAct loop.")
