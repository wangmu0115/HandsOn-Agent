from typing import TYPE_CHECKING

from athena_core._import_utils import import_attr

if TYPE_CHECKING:
    from athena_core.loggers import setup_logger
    from athena_core.metas import PostInitMeta

__all__ = [
    "setup_logger",
    "PostInitMeta",
]

_dynamic_imports = {
    "setup_logger": "loggers",
    "PostInitMeta": "metas",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
