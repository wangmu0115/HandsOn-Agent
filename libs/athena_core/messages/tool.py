from typing import Any, Literal, TypedDict

from athena_core.messages.base import BaseMessage


class ToolMessage(BaseMessage):
    type: Literal["tool"] = "tool"
    """The type of the message (used for serialization). Defaults to "tool"."""

    def __init__(self, content, **kwargs: Any):
        super().__init__(content=content, **kwargs)


class ToolCall(TypedDict):
    class Function(TypedDict):
        name: str
        """The name of the tool to be called."""
        arguments: dict[str, Any]
        """The arguments to the tool call."""

    function: Function
    """Function to be called."""


def tool_call(*, name: str, arguments: dict[str, Any]) -> ToolCall:
    """Create a tool call.

    Args:
        name: The name of the tool to be called.
        args: The arguments to the tool call.
    """
    return ToolCall(function=ToolCall.Function(name=name, arguments=arguments))
