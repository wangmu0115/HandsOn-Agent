"""User message."""

from typing import Literal

from messages.base import BaseMessage


class UserMessage(BaseMessage):
    """
    Message from a user.

    UserMessage are messages that are passed in from a user to the model.
    """

    type: Literal["user"] = "user"
    """The type of the message (used for serialization). Defaults to "user"."""

    def __init__(self, content):
        super().__init__(content=content)
