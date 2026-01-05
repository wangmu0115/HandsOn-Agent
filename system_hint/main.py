import argparse
import sys

from athena_core.loggers import setup_logger
from system_hint.config import SystemHintConfig

logger = setup_logger(__name__)

def execute_single_task(task: str, config: SystemHintConfig):
    

def interactive_mode():
    pass


def main():
    parser = argparse.ArgumentParser(description="System-Hint Enhanced AI Agent - Advanced trajectory management with system hints")
    parser.add_argument("--mode", choices=["single", "interactive", "demo", "sample"], default="interactive", help="Execution mode (default: interactive)")

    parser.add_argument("--task", type=str, help="Task to execute (for single mode)")

    parser.add_argument("--demo", choices=["basic", "loop", "comparison"], help="Specific demo to run (for demo mode)")

    parser.add_argument("--no-timestamps", action="store_true", help="Disable timestamp tracking")

    parser.add_argument("--no-counter", action="store_true", help="Disable tool call counter")

    parser.add_argument("--no-todo", action="store_true", help="Disable TODO list management")

    parser.add_argument("--no-errors", action="store_true", help="Disable detailed error messages")

    parser.add_argument("--no-state", action="store_true", help="Disable system state tracking")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    match args.mode:
        case "single":
            if not args.task:
                logger.error("❌ Error: --task requried for `single` mode")
                logger.info("Example: python main.py --mode single --task 'Create a hello world script'")
                sys.exit(1)
        case "sample":
            pass
        case "interactive":
            interactive_mode()
        case "demo":
            pass
        case _:
            logger.error("Unsupported mode: %s", args.mode)
            sys.exit(1)


if __name__ == "__main__":
    main()
