"""AI message."""

from typing import Any, Literal

from messages.base import BaseMessage
from messages.tool import ToolCall


class AIMessage(BaseMessage):
    """
    Message from an AI.

    AIMessage is returned from a chat model as a response to a prompt.
    """

    type: Literal["assistant"] = "assistant"
    """The type of the message (used for serialization). Defaults to "assistant"."""

    tool_calls: list[ToolCall] = []
    """If provided, tool calls associated with the message."""

    def __init__(self, content, **kwargs: Any):
        super().__init__(content=content, **kwargs)
