from typing import Any, Literal, Union


class ChatGenerationChunk:
    type: Literal["final", "thinking", "tool_call", "tool_result", "error"]

    content: Union[str, dict]

    def __init__(self, type, content):
        self.type = type
        self.content = content

    @classmethod
    def final(cls, content: str):
        return cls("final", content)

    @classmethod
    def thinking(cls, content: str):
        return cls("thinking", content)

    @classmethod
    def tool_call(cls, tool_name: str, tool_args: dict[str, Any]):
        return cls("tool_call", {"name": tool_name, "args": tool_args})

    @classmethod
    def tool_result(cls, tool_result: str):
        return cls("tool_result", tool_result)

    @classmethod
    def error(cls, content: str | Exception):
        return cls("error", str(content))
