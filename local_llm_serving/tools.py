import json
from typing import Any, Callable, Optional

from builtin_tools import (
    code_interpreter,
    convert_currency,
    get_current_temperature,
    get_current_time,
)


class Tool:
    class Parameter:
        def __init__(
            self,
            name: str,
            type: str,
            description: str,
            *,
            enums: Optional[list] = None,
            required: bool = False,
            default: Any = None,
        ):
            self.name = name
            self.type = type
            self.description = description
            self.enums = enums
            self.required = required
            self.default = default

        def asdict(self, with_name: bool = False):
            d = dict(type=self.type, description=self.description)
            if not self.required:
                d["default"] = self.default
            if self.enums is not None:
                d["enum"] = self.enums
            if with_name:
                d["name"] = self.name
            return d

    def __init__(
        self,
        *,
        function: Callable,
        description: str,
        name: Optional[str] = None,
        parameters: list[Parameter] = None,
    ):
        self.name = name or function.__name__
        self.function = function
        self.description = description
        self.parameters = parameters or []

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {p.name: p.asdict() for p in self.parameters},
                    "required": [p.name for p in self.parameters if p.required],
                },
            },
        }


class ToolRegistry:
    """Registry for managing available tools"""

    def __init__(self, **kwargs):
        self.tools: dict[str, Tool] = dict()
        _register_default_tools(self)

    def register_tool(self, tool: Tool):
        """Register a new tool."""
        self.tools[tool.name] = tool

    def get_tool_schemas(self):
        """Get OpenAI-compatible tool schemas."""
        return [tool.get_schema() for tool in self.tools.values()]

    def execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name with given arguments"""
        tool = self.tools.get(name, None)
        if tool is None:
            return json.dumps({"error": f"Tool `{name}` not found."})
        try:
            result = tool.function(**args)
            return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            return json.dumps({"error": str(e)})


def _register_default_tools(tool_registry: ToolRegistry):
    """Register Tools"""
    tool_registry.register_tool(
        Tool(
            function=get_current_temperature,
            description="Get the current temperature for a specific location",
            name="get_current_temperature",
            parameters=[
                Tool.Parameter(
                    "location",
                    "string",
                    "The city and country, e.g., 'Paris, France'",
                    required=True,
                ),
                Tool.Parameter(
                    "unit",
                    "string",
                    "The temperature unit to use (by default, celsius)",
                    enums=["celsius", "fahrenheit"],
                    default="celsius",
                ),
            ],
        )
    )

    tool_registry.register_tool(
        Tool(
            function=get_current_time,
            description="Get the current date and time in a specific timezone",
            name="get_current_time",
            parameters=[
                Tool.Parameter(
                    "timezone",
                    "string",
                    "Timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Shanghai'). Use standard IANA timezone names.",
                    default="UTC",
                )
            ],
        )
    )

    tool_registry.register_tool(
        Tool(
            function=convert_currency,
            description="Convert an amount from one currency to another. You MUST use this tool to convert currencies in order to get the latest exchange rate.",
            name="convert_currency",
            parameters=[
                Tool.Parameter("amount", "number", "Amount to convert", required=True),
                Tool.Parameter("from_currency", "string", "Source currency code (e.g., 'USD', 'EUR')", required=True),
                Tool.Parameter("to_currency", "string", "Target currency code (e.g., 'USD', 'EUR')", required=True),
            ],
        )
    )

    tool_registry.register_tool(
        Tool(
            function=code_interpreter,
            description="Execute Python code for calculations and data processing. You MUST use this tool to perform any complex calculations or data processing.",
            name="code_interpreter",
            parameters=[Tool.Parameter("code", "string", "Python code to execute", required=True)],
        )
    )
