"""System message."""

from typing import Any, Literal

from athena_core.messages.base import BaseMessage


class SystemMessage(BaseMessage):
    """
    Message for priming AI behavior.

    The system message is usually passed in as the first of a sequence
    of input messages.
    """

    type: Literal["system"] = "system"
    """The type of the message (used for serialization). Defaults to "system"."""

    def __init__(self, content, **kwargs: Any):
        super().__init__(content=content, **kwargs)
