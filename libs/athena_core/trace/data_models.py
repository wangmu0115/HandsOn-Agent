from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional


class TodoStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    call_numbers: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: Optional[int] = None


@dataclass
class TodoItem:
    id: int
    content: str
    status: TodoStatus = TodoStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Trajectory:
    timestamp: datetime
    iteration: int = 1
    model: Optional[str] = None  # Model name and provider
    provider: Optional[str] = None

    def to_dict(self): ...

    def to_json(self): ...


if __name__ == "__main__":
    print(Trajectory(datetime.now(), iteration=10))
    print(ToolCall("test"))
