from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from athena_core.tools import Tool


class TodoStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TodoItem:
    id: int
    content: str
    status: TodoStatus = TodoStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


def get_todo_tools(todo_list: TodoList) -> list[Tool]:
    return [
        Tool(
            function=todo_list.rewrite_todo_list,
            description="Rewrite the TODO list with new pending items (keeps completed/cancelled items)",
            name="rewrite_todo_list",
            parameters=[
                Tool.Parameter(
                    name="items",
                    type="array",
                    description="List of new TODO items to add as pending",
                    required=True,
                )
            ],
        )
    ]

    (
        {
            "type": "function",
            "function": {
                "name": "rewrite_todo_list",
                "description": "Rewrite the TODO list with new pending items (keeps completed/cancelled items)",
                "parameters": {
                    "type": "object",
                    "properties": {"items": {"type": "array", "items": {"type": "string"}, "description": "List of new TODO items to add as pending"}},
                    "required": ["items"],
                },
            },
        },
    )


class TodoList:
    def __init__(self):
        self.todo_list: list[TodoItem] = []
        self.next_todo_id = 1

    def rewrite_todo_list(self, items: list[str]) -> None:
        # Keep completed and cancelled items
        kept_items = [item for item in self.todo_list if item.status in [TodoStatus.COMPLETED, TodoStatus.CANCELLED]]
        # Create new pending items
        new_items = []
        for content in items:
            new_items.append(TodoItem(self.next_todo_id, content))
            self.next_todo_id += 1
        # Update todo list
        self.todo_list = kept_items + new_items
        return {
            "success": True,
            "kept_items": len(kept_items),
            "new_items": len(new_items),
            "total_items": len(self.todo_list),
        }

    def update_todo_status(self, updates: list[dict[str, Any]]) -> dict[str, Any]:
        updated_count = 0
        for update in updates:
            item_id = update["id"]
            for item in self.todo_list:
                if item.id == item_id:
                    item.status = TodoStatus(update["status"])
                    item.updated_at = datetime.now()
                    updated_count += 1
                    break
        return {
            "success": True,
            "updated_items": updated_count,
            "total_items": len(self.todo_list),
        }
