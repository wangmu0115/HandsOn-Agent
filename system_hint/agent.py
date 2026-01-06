"""
System-Hint Enhanced AI Agent
An agent that demonstrates advanced trajectory management with system hints,
including timestamps, tool call tracking, todo lists, and detailed error messages.
"""

import os
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional

from athena_core.loggers import setup_logger
from athena_core.messages import BaseMessage, UserMessage
from athena_core.metas import PostInitMeta
from athena_core.tools.tool_register import ToolRegister
from config import SystemHintConfig
from openai import OpenAI
from system_hint.todo_tools import TodoList
from utils import get_system_state

logger = setup_logger(__name__)


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


def get_todo_status_symbol(status: Optional[TodoStatus] = None):
    match status:
        case TodoStatus.PENDING:
            return "⏳"
        case TodoStatus.IN_PROGRESS:
            return "🔄"
        case TodoStatus.COMPLETED:
            return "✅"
        case TodoStatus.CANCELLED:
            return "❌"
        case _:
            return "❓"


class SystemHintAgent(metaclass=PostInitMeta):
    def __init__(self, api_key: str, *, model: str = "kimi-k2-0905-preview", config: Optional[SystemHintConfig] = None):
        self.api_key = api_key
        self.model = model
        self.config = config or SystemHintConfig()
        self.conversation_history: list[BaseMessage] = []
        self.tool_register = ToolRegister()

    def __post_init__(self):
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.moonshot.cn/v1")

        self.todo_list: list[TodoItem] = []

        # Track current working directory
        self.current_directory = os.getcwd()

        # Track last messages sent to LLM
        self.last_llm_messages = None

        # Register Tools
        if self.config.enable_todo_list:
            self.todo_list = TodoList()
            




    def execute_task(self, task: str, max_iterations: int = 20) -> dict[str, Any]:
        """
        Execute a task using available tools with system hints

        Args:
            task: The task to execute
            max_iterations: Maximum number of tool calls

        Returns:
            Task execution result
        """
        if self.config.enable_timestamps:  # If enabled add timestamp before user message
            task = f"[{datetime.now().strftime(self.config.timestamp_format)}] " + task
        self.conversation_history.append(UserMessage(task))

        iteration = 0
        final_answer = None
        while iteration < max_iterations:
            iteration += 1
            logger.info("Iteration: %d/%d", iteration, max_iterations)
            # TODO Save trajectory at the start of each iteration

            try:
                # system_hint =
                pass

            except Exception:
                pass

    def _get_system_hint(self) -> Optional[str]:
        """Get system hint content with current state"""
        hint_parts = []
        if self.config.enable_system_state:
            hint_parts.append("=== SYSTEM STATE ===")
            hint_parts.append(get_system_state(timestamp_format=self.config.timestamp_format))
            hint_parts.append("")
        if self.config.enable_todo_list:
            hint_parts.append("=== CURRENT TASKS ===")
            hint_parts.append(self._format_todo_list())

            ...

        # if not any([self.config.enable_system_state, self.config.enable_todo_list]):
        #     return None

        if self.config.enable_system_state:
            hint_parts.append("=== SYSTEM STATE ===")
            hint_parts.append(self._get_system_state())
            hint_parts.append("")

        if self.config.enable_todo_list and self.todo_list:
            hint_parts.append("=== CURRENT TASKS ===")
            hint_parts.append(self._format_todo_list())
            hint_parts.append("")

        if hint_parts:
            return "\n".join(hint_parts)
        return None

    def _get_timestamp(self) -> str:
        now_dt = datetime.now(tz=datetime.now().astimezone().tzinfo)
        return now_dt.strftime(self.config.timestamp_format)

    def _get_system_state(self) -> str:
        system = platform.system()
        if system == "Windows":
            shell_type = "Windows Command Prompt or PowerShell"
        elif system == "Darwin":
            shell_type = "macOS Terminal (zsh/bash)"
        else:
            shell_type = f"Linux Shell ({os.environ.get('SHELL', 'bash')})"

        state_info = [
            f"Current Time: {self._get_timestamp()}",
            f"Current Directory: {self.current_directory or os.getcwd()}",
            f"System: {system} ({platform.release()})",
            f"Shell Environment: {shell_type}",
            f"Python Version: {sys.version.split()[0]}",
        ]

        return "\n".join(state_info)

    def _format_todo_list(self) -> str:
        if not self.todo_list:
            return "TODO List: Empty"
        else:
            lines = ["TODO List:"]
            for todo_item in self.todo_list:
                status_symbol = get_todo_status_symbol(todo_item.status)
                lines.append(f"  [{todo_item.id}] {status_symbol} {todo_item.content} ({todo_item.status})")
            return "\n".join(lines)

    def _register_tools(self):

        # Add todo management tools if enabled
        if self.config.enable_todo_list:
            todo_list = 



if __name__ == "__main__":
    print(get_todo_status_symbol())
