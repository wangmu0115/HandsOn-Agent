from typing import TYPE_CHECKING

from _import_utils import import_attr

if TYPE_CHECKING:
    from currency import convert_currency
    from dt import get_current_time
    from pdf import parse_pdf
    from py_interpreter import code_interpreter
    from weather import get_current_temperature

__all__ = [
    "convert_currency",
    "parse_pdf",
    "get_current_time",
    "get_current_temperature",
    "code_interpreter",
]

_dynamic_imports = {
    "convert_currency": "currency",
    "parse_pdf": "pdf",
    "get_current_time": "dt",
    "get_current_temperature": "weather",
    "code_interpreter": "py_interpreter",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return __all__
