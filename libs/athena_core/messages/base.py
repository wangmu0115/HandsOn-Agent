"""Base message."""

from typing import Any, Sequence

from pydantic import BaseModel


class BaseMessage(BaseModel):
    content: str
    """The string contents of the message."""

    type: str
    """The type of the message. Must be a string that is unique to the message type.

    The purpose of this field is to allow for easy identification of the message type
    when deserializing messages.
    """

    model_config = {"extra": "allow"}

    def __init__(self, content: str, **kwargs: Any):
        super().__init__(content=content, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.model_dump().items())})"


def message_to_dict(message: BaseMessage) -> dict:
    """
    Convert a Message to a dictionary.

    Args:
        message: Message to convert.

    Returns:
        Message as a dict. The dict will have a "role" key with the message type
        and a "content" key with the message content as a dict.
    """
    d = {"role": message.type}
    d.update(message.model_dump(exclude=["type"]))
    return d


def messages_to_dict(messages: Sequence[BaseMessage]) -> list[dict]:
    """
    Convert a sequence of Messages to a list of dictionaries.

    Args:
        messages: Sequence of messages (as BaseMessages) to convert.

    Returns:
        List of messages as dicts.
    """
    return [message_to_dict(m) for m in messages]
