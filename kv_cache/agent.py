import time
from dataclasses import dataclass, field
from typing import Any, Optional

from constants import KVCacheMode
from messages import SystemMessage, UserMessage
from openai import OpenAI
from tools import ToolRegistry

from kv_cache.prompt_templates import get_system_prompt


@dataclass
class AgentMetrics:
    ttft: Optional[float] = None  # Time to first token (first iteration)
    ttft_per_iteration: list[float] = field(default_factory=list)  # TTFT for each iteration
    total_time: Optional[float] = None
    iterations: Optional[int] = None
    tool_calls: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None


class KVCacheAgent:
    """
    ReAct Agent with different KV cache optimization modes
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "kimi-k2-0905-preview",
        mode: KVCacheMode = KVCacheMode.CORRECT,
        root_dir: str = ".",
        verbose: bool = True,
    ):
        self.client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")
        self.model = model
        self.mode = mode
        self.verbose = verbose
        self.tools = ToolRegistry(root_dir=root_dir).get_tool_schemas()

        self.user_credits = 100  # For dynamic profile mode

        self.conversation_history = []
        self.metrics = AgentMetrics()

    def execute_task(self, task: str, max_iterations: int = 50) -> dict[str, Any]:
        """
        Execute a task using ReAct pattern with standard OpenAI tool calling

        Args:
            task: The task to execute
            max_iterations: Maximum number of iterations

        Returns:
            Task execution result with metrics
        """
        start = time.perf_counter()
        iteration = 0
        final_answer = None
        tool_calls = []

        original_task = task

        while iteration < max_iterations:
            iteration += 1
            # Message handling for KV cache demonstration
            #
            # CORRECT mode: Build messages once on first iteration, then keep appending
            #   - Maintains stable context → KV cache works efficiently
            #
            # INCORRECT modes: Recreate entire messages list from history each iteration
            #   - Forces complete context reconstruction → KV cache invalidated
            #   - Within an iteration, we still append to messages for proper API flow
            #   - But at the start of each new iteration, we rebuild from scratch

            match self.mode:
                case KVCacheMode.CORRECT:
                    ...
                case _:
                    ...

    def _format_messages(self, task: str):
        messages = []

        messages.append(SystemMessage(get_system_prompt(self.mode)))
        if self.mode == KVCacheMode.DYNAMIC_PROFILE:
            self.user_credits -= 1
            messages.append(UserMessage(f"[User Profile: Premium user with {self.user_credits} credits remaining]"))
        elif self.mode == KVCacheMode.SLIDING_WINDOW:
            # Preserve all system and user messages, at the most recent 6 messages
            if self.conversation_history:
                # Include all system and user messages, and if there are at least 6 messages, include the last 6 messages regardless of role
                system_user_msgs = [msg for msg in self.conversation_history if msg.get("role") in ("system", "user")]
                messages.extend(system_user_msgs)
                if len(self.conversation_history) >= 6:
                    messages.extend(self.conversation_history[-6:])
