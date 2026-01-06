import os
import platform
import sys
from datetime import datetime


def get_current_timestamp(as_str: bool = False, dt_format: str = "%Y-%m-%d %H:%M:%S") -> datetime | str:
    now_dt = datetime.now(tz=datetime.now().astimezone().tzinfo)
    if as_str:
        return now_dt.strftime(dt_format)
    else:
        return now_dt


def get_system_state(**kwargs) -> str:
    """Get current system state information"""
    # Detect OS
    system = platform.system()
    if system == "Windows":
        shell_type = "Windows Command Prompt or PowerShell"
    elif system == "Darwin":
        shell_type = "macOS Terminal (zsh/bash)"
    else:
        shell_type = f"Linux Shell ({os.environ.get('SHELL', 'bash')})"

    current_directory = kwargs.get("current_directory", os.getcwd())
    current_timestamp = get_current_timestamp(True, kwargs.get("timestamp_format", "%Y-%m-%d %H:%M:%S"))

    state_info = [
        f"Current Time: {current_timestamp}",
        f"Current Directory: {current_directory}",
        f"System: {system} ({platform.release()})",
        f"Shell Environment: {shell_type}",
        f"Python Version: {sys.version.split()[0]}",
    ]

    return "\n".join(state_info)
