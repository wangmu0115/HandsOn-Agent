from dataclasses import dataclass


@dataclass
class SystemHintConfig:
    """Configuration for system hints"""

    enable_timestamps: bool = True
    enable_tool_counter: bool = True
    enable_todo_list: bool = True
    enable_detailed_errors: bool = True
    enable_system_state: bool = True  # Current dir, shell, etc.
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    simulate_time_delay: bool = False  # For demo purposes
    save_trajectory: bool = True  # Save conversation history to file
    trajectory_file: str = "trajectory.json"  # File to save trajectory to
