from enum import StrEnum


class KVCacheMode(StrEnum):
    CORRECT = "correct"  # Correct implementation with stable context
    DYNAMIC_SYSTEM = "dynamic_system"  # Changing system prompt with timestamp
    SHUFFLED_TOOLS = "shuffled_tools"  # Shuffling tool order each request
    DYNAMIC_PROFILE = "dynamic_profile"  # Changing user profile with credits
    SLIDING_WINDOW = "sliding_window"  # Only keeping recent 5 messages
    TEXT_FORMAT = "text_format"  # Formatting messages as plain text
