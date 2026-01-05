import json
from typing import Any, Callable, Literal, Optional


class Tool:
    class Parameter:
        def __init__(
            self,
            name: str,
            type: Literal["integer", "string", "boolean", "float"],
            description: str,
            *,
            enum: Optional[list] = None,
            required: bool = False,
            default: Any = None,
        ):
            self.name = name
            self.type = type
            self.description = description
            self.enum = enum
            self.required = required
            self.default = default

        def asdict(self, with_name: bool = False):
            d = dict(type=self.type, description=self.description)
            if not self.required:
                d["default"] = self.default
            if not self.enum:
                d["enum"] = self.enum
            if with_name:
                d["name"] = self.name
            return d

    def __init__(self, *, function: Callable, description: str, name: Optional[str] = None, parameters: list[Parameter] = None):
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
                    "properties": {p.name: p.asdict(with_name=False) for p in self.parameters},
                    "required": [p.name for p in self.parameters if p.required],
                },
            },
        }


class ToolRegister:
    """Registry for managing available tools"""

    def __init__(self):
        self.tools: dict[str, Tool] = dict()

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
