from typing import TYPE_CHECKING

from athena_core._import_utils import import_attr

if TYPE_CHECKING:
    from athena_core.messages.ai import AIMessage
    from athena_core.messages.base import BaseMessage, message_to_dict, messages_to_dict
    from athena_core.messages.system import SystemMessage
    from athena_core.messages.tool import ToolCall, ToolMessage, tool_call
    from athena_core.messages.user import UserMessage

__all__ = [
    "BaseMessage",
    "message_to_dict",
    "messages_to_dict",
    "SystemMessage",
    "UserMessage",
    "AIMessage",
    "ToolMessage",
    "ToolCall",
    "tool_call",
]

_dynamic_imports = {
    "BaseMessage": "base",
    "message_to_dict": "base",
    "messages_to_dict": "base",
    "SystemMessage": "system",
    "UserMessage": "user",
    "AIMessage": "ai",
    "ToolMessage": "tool",
    "ToolCall": "tool",
    "tool_call": "tool",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
