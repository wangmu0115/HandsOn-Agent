"""
System-Hint Enhanced AI Agent
An agent that demonstrates advanced trajectory management with system hints,
including timestamps, tool call tracking, TODO lists, and detailed error messages.
"""

from typing import Optional

from athena_core.loggers import setup_logger
from athena_core.metas import PostInitMeta
from config import SystemHintConfig
from openai import OpenAI

logger = setup_logger(__name__)


class SystemHintAgent(metaclass=PostInitMeta):
    def __init__(self, api_key: str, model: str = "kimi-k2-0905-preview", config: Optional[SystemHintConfig] = None):
        self.api_key = api_key
        self.model = model
        self.config = config or SystemHintConfig()

    def __post_init__(self):
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.moonshot.cn/v1")
