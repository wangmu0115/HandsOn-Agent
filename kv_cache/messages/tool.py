from typing import Literal

from messages.base import BaseMessage


class ToolMessage(BaseMessage):
    type: Literal["tool"] = "tool"
    """The type of the message (used for serialization). Defaults to "tool"."""
