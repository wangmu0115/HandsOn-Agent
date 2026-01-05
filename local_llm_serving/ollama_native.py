"""
Ollama Native Tool Calling Implementation
Uses Ollama's standard tool calling API (requires compatible models)
"""

from typing import Iterator, Union

import ollama
from athena_core.loggers import setup_logger
from athena_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
    UserMessage,
    messages_to_dict,
    tool_call,
)
from outputs import ChatGenerationChunk
from tools import ToolRegistry

logger = setup_logger(__name__, "INFO")


class OllamaNativeAgent:
    """Agent using Ollama's native tool calling support"""

    def __init__(self, model: str):
        self.model = model
        self.client = ollama.Client()
        self.tool_register = ToolRegistry()
        self.conversation_history: list[BaseMessage] = []

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
    ) -> Union[str | Iterator[ChatGenerationChunk]]:
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
        self.conversation_history.append(UserMessage(message))
        tools = self.tool_register.get_tool_schemas() if use_tools else None

        try:
            # Call Ollama with tools
            response = self.client.chat(
                self.model,
                messages_to_dict(self.conversation_history),
                tools=tools,
                options={"temperature": temperature},
            )
            response_message = response.message
            if response_message.tool_calls:  # Handle tool calls
                logger.info("Model requested %d tool call(s).", len(response_message.tool_calls))
                # Add assistant's message with tool calls to history
                ai_message = AIMessage(response_message.thinking or "")
                ai_message.tool_calls = [tool_call(name=tc.function.name, arguments=tc.function.arguments) for tc in response_message.tool_calls]
                self.conversation_history.append(ai_message)

                for tc in response_message.tool_calls:
                    name = tc.function.name
                    args = tc.function.arguments
                    result = self.tool_register.execute_tool(name, args)
                    logger.info("Executing tool: %s with args: %s, result: %s", name, args, result)
                    # Add tool result to history
                    self.conversation_history.append(ToolMessage(result))

                final_response = self.client.chat(
                    self.model,
                    messages_to_dict(self.conversation_history),
                    tools=tools,
                    options={"temperature": temperature},
                )
                final_response_message = final_response.message
                # Add final response to history
                self.conversation_history.append(AIMessage(final_response_message.content))
                return final_response_message.content
            else:  # No tool calls
                self.conversation_history.append(AIMessage(response_message.content))
                return response_message.content
        except Exception as e:
            logger.exception("Error in chat:", e)
            return f"Error: {e}"

    def chat_stream(
        self,
        message: str,
        *,
        use_tools: bool = True,
        temperature: float = 0.7,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream a message to the model and handle tool calls in a ReAct loop.

        Yields chunks that include:
        - type: 'final', 'thinking', 'tool_call', 'tool_result', 'error'
        - content: The actual content
        """
        # Add use message to history
        self.conversation_history.append(UserMessage(message))
        tools = self.tool_register.get_tool_schemas() if use_tools else None

        # ReAct loop, and Prevent infinite loops
        max_iterations, iteration = 10, 0
        while iteration < max_iterations:
            iteration += 1
            try:
                stream_response = self.client.chat(
                    self.model,
                    messages_to_dict(self.conversation_history),
                    tools=tools,
                    options={"temperature": temperature},
                    stream=True,
                )

                collected_content = []  # Collect chunk content or thinking
                final_response = False
                # Process the stream
                for chunk in stream_response:
                    chunk_message = chunk.message
                    if chunk_message.content:  # Get final response
                        final_response = True
                        collected_content.append(chunk_message.content)
                        yield ChatGenerationChunk.final(chunk_message.content)
                    elif chunk_message.thinking:  # Thinking
                        collected_content.append(chunk_message.thinking)
                        yield ChatGenerationChunk.thinking(chunk_message.thinking)
                    elif chunk_message.tool_calls:  # ToolCall
                        # Add assistant's message with tool calls to history
                        ai_message = AIMessage("".join(collected_content))
                        ai_message.tool_calls = [tool_call(name=tc.function.name, arguments=tc.function.arguments) for tc in chunk_message.tool_calls]
                        self.conversation_history.append(ai_message)
                        for tc in chunk_message.tool_calls:
                            name = tc.function.name
                            args = tc.function.arguments
                            yield ChatGenerationChunk.tool_call(name, args)
                            result = self.tool_register.execute_tool(name, args)
                            yield ChatGenerationChunk.tool_result(result)
                            # Add tool result to history
                            self.conversation_history.append(ToolMessage(result))
                    else:
                        break
                if final_response:  # Exit the ReAct loop
                    self.conversation_history.append(AIMessage("".join(collected_content)))
                    break
            except Exception as e:
                logger.exception("Error in chat stream:", e)
                yield ChatGenerationChunk.error(e)
                break
        if iteration >= max_iterations:
            yield ChatGenerationChunk.error("Maximum iterations reached in ReAct loop.")
