import json
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional


@dataclass
class Parameter:
    name: str
    type: Literal["integer", "string", "boolean", "float", "array", "object"]

    description: Optional[str] = None
    enum: Optional[list] = None
    required: bool = False
    default: Any = None

    def asdict(self, with_name: bool = False):
        d = {"type": self.type}
        if with_name:
            d["name"] = self.name
        if self.description:
            d["description"] = self.description
        if self.enum:
            d["enum"] = self.enum
        if not self.required:
            d["default"] = self.default

        return d


class NestedParameter(Parameter):
    def __init__(self, name: str = "", **kwargs):
        super().__init__(name, "object", **kwargs)
        self.properties: list[Parameter] = []

    def add_property(self, param: Parameter):
        self.properties.append(param)
        return self

    def add_properties(self, params: list[Parameter]):
        self.properties.extend(params)
        return self

    def asdict(self, with_name: bool = False):
        d = super().asdict(with_name)
        if self.properties:
            d["properties"] = {p.name: p.asdict(with_name) for p in self.properties}
            d["required"] = [p.name for p in self.properties if p.required]
        else:
            d["properties"] = {}
            d["required"] = []

        return d


class ArrayParameter(Parameter):
    def __init__(self, name: str, item: Parameter, description: str = "", **kwargs):
        super().__init__(name, "array", description=description, **kwargs)
        self.item = item

    def asdict(self, with_name: bool = False):
        d = super().asdict(with_name)
        d["items"] = self.item.asdict(with_name)
        return d


class Tool:
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
                "parameters": NestedParameter(required=True).add_properties(self.parameters).asdict(),
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


if __name__ == "__main__":
    p1 = NestedParameter(required=True).add_properties(
        [
            Parameter("file_path", "string", "Path to the file to read (absolute or relative to current directory)", required=True),
            Parameter(
                "begin_line",
                "integer",
                "Optional: Line number to start reading from (1-based indexing). E.g., begin_line=10 starts from line 10.",
                default=1,
            ),
            Parameter(
                "number_lines",
                "integer",
                "Optional: Number of lines to read from begin_line. E.g., number_lines=50 reads 50 lines.",
                default=100,
            ),
        ]
    )

    p2 = NestedParameter(required=True).add_property(
        ArrayParameter("items", Parameter("", "string", required=True), "List of new TODO items to add as pending", required=True)
    )

    p3 = NestedParameter(required=True).add_property(
        ArrayParameter(
            "updates",
            NestedParameter(required=True).add_properties(
                [
                    Parameter("id", "integer", description="TODO item ID", required=True),
                    Parameter(
                        "status", "string", description="New status for the item", required=True, enum=["pending", "in_progress", "completed", "cancelled"]
                    ),
                ]
            ),
            "List of TODO items to update with their new status",
            required=True,
        )
    )

    import json

    print(json.dumps(p1.asdict(), indent=2))
    print("\n" + "=" * 80 + "\n")
    print(json.dumps(p2.asdict(), indent=2))
    print("\n" + "=" * 80 + "\n")
    print(json.dumps(p3.asdict(), indent=2))
