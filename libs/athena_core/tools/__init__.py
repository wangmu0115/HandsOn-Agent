from typing import TYPE_CHECKING

from athena_core._import_utils import import_attr

if TYPE_CHECKING:
    from athena_core.tools.tool_register import Tool, ToolRegister

__all__ = [
    "Tool",
    "ToolRegister",
]

_dynamic_imports = {
    "Tool": "tool_register",
    "ToolRegister": "tool_register",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
