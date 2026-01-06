from argparse import Namespace
from dataclasses import dataclass


@dataclass
class SystemHintConfig:
    """Configuration for system hints"""

    enable_timestamps: bool = True
    enable_tool_counter: bool = True
    enable_todo_list: bool = True
    enable_detailed_errors: bool = True
    enable_system_state: bool = True
    simulate_time_delay: bool = False  # For demo purposes
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    save_trajectory: bool = True  # Save conversation history to file
    trajectory_file: str = "trajectory.json"  # File to save trajectory to

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(
            enable_timestamps=not args.no_timestamps,
            enable_tool_counter=not args.no_counter,
            enable_todo_list=not args.no_todo,
            enable_detailed_errors=not args.no_errors,
            enable_system_state=not args.no_state,
        )
