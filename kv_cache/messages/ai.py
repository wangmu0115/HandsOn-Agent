"""AI message."""

from typing import Literal

from messages.base import BaseMessage


class AIMessage(BaseMessage):
    """
    Message from an AI.

    AIMessage is returned from a chat model as a response to a prompt.
    """

    type: Literal["assistant"] = "assistant"
    """The type of the message (used for serialization). Defaults to "assistant"."""

    def __init__(self, content):
        super().__init__(content=content)
