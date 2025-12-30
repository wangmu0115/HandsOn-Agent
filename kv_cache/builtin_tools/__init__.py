from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_files import LocalFileTools

__all__ = [
    "LocalFileTools",
]

_dynamic_imports = {
    "LocalFileTools": "local_files",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    package = __spec__.parent
    if module_name == "__module__" or module_name is None:
        result = import_module(f".{attr_name}", package=package)
    else:
        module = import_module(f".{module_name}", package=package)
        result = getattr(module, attr_name)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return __all__
