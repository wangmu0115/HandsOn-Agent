from dataclasses import dataclass
from typing import Callable


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

    def register_tool(self, name: str, function: Callable, description: str, parameters: dict):
        """Register a new tool."""
        self.tools[name] = Tool(name, function, description, parameters)

    def get_tool_schemas(self) -> list[dict]:
        """Get OpenAI-compatible tool schemas."""
        return [tool.schema() for tool in self.tools.values()]








    @staticmethod
    def get_current_temperature()