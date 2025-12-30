import json
from dataclasses import dataclass
from typing import Any, Callable

from builtin_tools import (
    code_interpreter,
    convert_currency,
    get_current_temperature,
    get_current_time,
)


@dataclass
class Tool:
    name: str
    function: Callable
    description: str
    parameters: dict

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry for managing available tools"""

    def __init__(self):
        self.tools: dict[str, Tool] = dict()
        self._register_default_tools()

    def register_tool(self, name: str, function: Callable, description: str, parameters: dict):
        """Register a new tool."""
        self.tools[name] = Tool(name, function, description, parameters)

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

    def get_tool_schemas(self) -> list[dict]:
        """Get OpenAI-compatible tool schemas."""
        return [tool.schema() for tool in self.tools.values()]

    def _register_default_tools(self):
        self.register_tool(
            name="get_current_temperature",
            function=get_current_temperature,
            description="Get the current temperature for a specific location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and country, e.g., 'Paris, France'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use (by default, celsius)",
                    },
                },
                "required": ["location", "unit"],
            },
        )
        self.register_tool(
            name="get_current_time",
            function=get_current_time,
            description="Get the current date and time in a specific timezone",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Shanghai'). Use standard IANA timezone names.",
                        "default": "UTC",
                    }
                },
                "required": [],
            },
        )
        self.register_tool(
            name="convert_currency",
            function=convert_currency,
            description="Convert an amount from one currency to another. You MUST use this tool to convert currencies in order to get the latest exchange rate.",
            parameters={
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "number",
                        "description": "Amount to convert",
                    },
                    "from_currency": {
                        "type": "string",
                        "description": "Source currency code (e.g., 'USD', 'EUR')",
                    },
                    "to_currency": {
                        "type": "string",
                        "description": "Target currency code (e.g., 'USD', 'EUR')",
                    },
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        )
        self.register_tool(
            name="code_interpreter",
            function=code_interpreter,
            description="Execute Python code for calculations and data processing. You MUST use this tool to perform any complex calculations or data processing.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        )
