import json
from typing import Any, Callable, Optional

from builtin_tools import LocalFileTools


class Tool:
    class Parameter:
        def __init__(self, name: str, type: str, description: str, *, required: bool = False, default: Any = None):
            self.name = name
            self.type = type
            self.description = description
            self.required = required
            self.default = default

        def asdict(self, with_name: bool = False):
            d = dict(type=self.type, description=self.description)
            if not self.required:
                d["default"] = self.default
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
    def __init__(self, **kwargs):
        self.tools: dict[str, Tool] = dict()
        if "root_dir" in kwargs:  # Register local file tools
            self._register_local_file_tools(kwargs["root_dir"])

    def register_tool(self, tool: Tool):
        """Register a new tool."""
        self.tools[tool.name] = tool

    def get_tool_schemas(self):
        """Get OpenAI-compatible tool schemas."""
        return [tool.get_schema() for tool in self.tools.values()]

    def execute_tool(self, name: str, kwargs: dict[str, Any]) -> str:
        """Execute a tool by name with given arguments"""
        tool = self.tools.get(name, None)
        if tool is None:
            return json.dumps({"error": f"Tool `{name}` not found."})
        try:
            result = tool.function(**kwargs)
            return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _register_local_file_tools(self, root_dir: str):
        self.register_tool(
            Tool(
                function=LocalFileTools(root_dir).read_file,
                description="Read the contents of a file, optionally specifying a line range",
                name="read_file",
                parameters=[
                    Tool.Parameter("file_path", "string", "Path to the file relative to root directory", required=True),
                    Tool.Parameter("offset", "integer", "Line number to start reading from (0-based, default: 0)", default=0),
                    Tool.Parameter("size", "integer", "Number of lines to read (default: read all lines)", default=None),
                ],
            )
        )
        self.register_tool(
            Tool(
                function=LocalFileTools(root_dir).find,
                description="Find files matching a pattern",
                name="find",
                parameters=[
                    Tool.Parameter("pattern", "string", "File name pattern (supports wildcards like *.py)", required=True),
                    Tool.Parameter("directory", "string", "Directory to search in (default: current directory)", default="."),
                ],
            )
        )
        self.register_tool(
            Tool(
                function=LocalFileTools(root_dir).grep,
                description="Search for a pattern in files",
                name="grep",
                parameters=[
                    Tool.Parameter("pattern", "string", "Regular expression pattern to search for", required=True),
                    Tool.Parameter("file_path", "string", "Single file to search in (optional)", default=None),
                    Tool.Parameter("directory", "string", "Directory to search in (optional)", default=None),
                ],
            )
        )
