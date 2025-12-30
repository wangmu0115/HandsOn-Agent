from __future__ import annotations

from typing import Literal

from libs.core.athena_core.messages import BaseMessage

# class BaseMessage:
#     content: str
#     """The string contents of the message."""

#     type: str
#     """The type of the message. Must be a string that is unique to the message type."""

#     def __init__(self, content: str):
#         self.content = content

#     def asdict(self) -> dict[str, str]:
#         return {"role": self.type, "content": self.content}

#     def __repr__(self):
#         return f"{self.__class__.__name__}({', '.join(f'{attr[0]}={repr(attr[1])}' for attr in self.__dict__.items())})"


class SystemMessage(BaseMessage):
    type: Literal["system"] = "system"

    def __init__(self, content):
        super().__init__(content=content)


class UserMessage(BaseMessage):
    type: Literal["user"] = "user"

    def __init__(self, content):
        super().__init__(content=content)


if __name__ == "__main__":
    print(SystemMessage("HelloWorld").asdict())
