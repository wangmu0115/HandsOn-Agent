from typing import TYPE_CHECKING

from _import_utils import import_attr

if TYPE_CHECKING:
    from messages.ai import AIMessage
    from messages.base import BaseMessage, message_to_dict, messages_to_dict
    from messages.system import SystemMessage
    from messages.tool import ToolMessage
    from messages.user import UserMessage

__all__ = [
    "BaseMessage",
    "message_to_dict",
    "messages_to_dict",
    "SystemMessage",
    "UserMessage",
    "AIMessage",
    "ToolMessage",
]

_dynamic_imports = {
    "BaseMessage": "base",
    "message_to_dict": "base",
    "messages_to_dict": "base",
    "SystemMessage": "system",
    "UserMessage": "user",
    "AIMessage": "ai",
    "ToolMessage": "tool",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
