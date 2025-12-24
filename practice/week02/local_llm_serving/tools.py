from dataclasses import dataclass
from typing import Callable

from builtin_tools.get_current_temperature import get_current_temperature


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
