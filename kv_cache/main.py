import argparse
import logging
import os
import sys

from agent import KVCacheAgent, KVCacheMode

logger = logging.getLogger(__name__)


def run_single_mode(api_key: str, mode: str, task: str, root_dir: str = "/tmp"):
    """
    Run agent in a single mode

    Args:
        api_key: API key for Kimi
        mode: KV cache mode to use
        task: Custom task (optional)
        root_dir: Root directory for file operations (default: "/tmp" from kv-cache dir)
    """
    kvcache_modes = {mode.value: mode for mode in KVCacheMode}
    if mode not in kvcache_modes:
        logger.error("Invalid KVCache mode: %s. Valid modes: %s", mode, ", ".join(kvcache_modes.keys()))
        return

    agent = KVCacheAgent(api_key, mode=kvcache_modes[mode], root_dir=root_dir)


def main():
    parser = argparse.ArgumentParser(description="KV Cache Demonstration with ReAct Agent")
    parser.add_argument("--api-key", type=str, help="API key for Kimi (or use MOONSHOT_API_KEY env var)")
    parser.add_argument("--mode", type=str, help="Single mode to run (correct, dynamic_system, etc.)")
    parser.add_argument("--compare", action="store_true", help="Run comparison across all modes")
    parser.add_argument("--task", type=str, help="Custom task to execute")
    parser.add_argument("--root-dir", type=str, default="../..", help="Root directory for file operations (default: ../.. = /projects)")
    parser.add_argument("--interactive", action="store_true", default=True, help="Interactive mode selection (default: True)")
    parser.add_argument("--no-interactive", dest="interactive", action="store_false", help="Disable interactive mode")

    args = parser.parse_args()
    # Get API key
    api_key = args.api_key or os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        logger.error("Please provide API key via --api-key or MOONSHOT_API_KEY environment variable")
        sys.exit(1)

    # Run based on mode
    if args.mode:
        run_single_mode(api_key, args.mode, args.task, args.root_dir)


if __name__ == "__main__":
    main()
