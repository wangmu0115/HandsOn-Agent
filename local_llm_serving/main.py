import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Iterator, Optional

from agent import ToolCallingAgent
from ollama_native import ChatResponseChunk
from tools import ToolRegistry


def show_system_info():
    print("\n📊 System Information:")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python: {sys.version.split()[0]}")


def show_interactive_commands():
    print("\n💡 Commands:")
    print("  /reset      - Reset conversation")
    print("  /tools      - Show available tools")
    print("  /samples    - Show sample tasks")
    print("  /sample <n> - Run sample task number n")
    print("  /stream     - Toggle streaming mode")
    print("  /help       - Show this help")
    print("  /exit       - Exit the program")


def show_available_tools():
    print("\n📦 Available Tools:")
    for i, tool in enumerate(ToolRegistry().get_tool_schemas(), 1):
        func = tool["function"]
        print(f"  {i}. {func['name']}: {func['description']}")


def show_sample_tasks(sample_tasks: Optional[list] = None):
    print("\n📋 Sample tasks:")
    sample_tasks = sample_tasks or get_sample_tasks()
    for i, sample in enumerate(sample_tasks, 1):
        print(f"\n{i}. {sample['name']}")
        task_preview = sample["task"].replace("\n", " ")[:100]
        if len(sample["task"]) > 100:
            task_preview += "..."
        print(f"   {task_preview}")


def show_task_detail(task: dict[str, str]):
    print(f"\n✅ Selected: {task['name']}")
    print("\nTask details:")
    print("-" * 80)
    print(task["task"])
    print("-" * 80)


def get_sample_tasks() -> list[dict[str, str]]:
    tasks_file_path = Path(__file__).resolve().parent / "sample_tasks.json"
    with open(tasks_file_path) as f:
        return json.loads(f.read())


def run_single_task(agent: ToolCallingAgent, task: str, stream: bool = True):
    """Run a single task with optional streaming."""
    print("\n" + "=" * 80)
    print("TASK EXECUTION")
    print("=" * 80)
    if stream:
        print("\n⏳ Processing (streaming)...\n")

        response_chunks = []
        thinking_shown = False
        tools_shown = False
        response_started = False
        last_chunk_type = None

        resp_chunks = agent.chat(task, use_tools=True, stream=True)

        for chunk in resp_chunks:
            chunk_type = chunk.type
            content = chunk.content
            match chunk_type:
                case "thinking":
                    if not thinking_shown:
                        print("🧠 Thinking: ", end="", flush=True)
                        thinking_shown = True
                    # Stream thinking character by character in gray
                    print(f"\033[90m{content}\033[0m", end="", flush=True)
                case "tool_call":
                    if not tools_shown:
                        print("\n\n🔧 Tool Calls:")
                        tools_shown = True
                    # Display tool call info
                    tool_info = content
                    print(f"  → {tool_info.get('name', 'unknown')}: {tool_info.get('arguments', {})}")
                    # Reset response_started flag after tool calls
                    response_started = False
                case "tool_result":
                    # Display tool result
                    result_str = str(content)
                    print(f"    ✓ {result_str}")
                    # Reset response_started flag after tool results
                    response_started = False
                case "answer":
                    if not response_started:
                        # Check if this is content after tool execution
                        if last_chunk_type in ["tool_result", "tool_call"]:
                            print("\n🤖 Assistant: ", end="", flush=True)
                        elif thinking_shown or tools_shown:
                            print("\n\n🤖 Assistant: ", end="", flush=True)
                        else:
                            print("🤖 Assistant: ", end="", flush=True)
                        response_started = True
                    # Stream the actual response content
                    print(content, end="", flush=True)
                    response_chunks.append(content)
                case "error":
                    print(f"\n❌ Error: {content}")
            last_chunk_type = chunk_type
        print("-" * 60)

    else:
        print("\n⏳ Processing...")
        response = agent.chat(task, use_tools=True, stream=False)
        print("\n✅ Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)


def interactive_mode(agent: ToolCallingAgent, stream: bool = True):
    """Run interactive chat mode with optional streaming."""
    print("\n" + "=" * 80)
    print("💬 INTERACTIVE MODE" + (" (STREAMING)" if stream else ""))
    print("=" * 80)
    print("\nYou can now chat with the AI agent.")
    show_available_tools()  # Show available tools
    show_interactive_commands()  # Show commands
    print("-" * 80)

    streaming_enabled = stream
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            match user_input.lower():
                case "":
                    continue
                case "/exit" | "quit":
                    print("👋 Goodbye!")
                    break
                case "/reset":
                    agent.reset_conversation()
                    print("✅ Conversation reset")
                    continue
                case "/help":
                    show_interactive_commands()
                    continue
                case "/tools":
                    show_available_tools()
                    continue
                case "/stream":
                    streaming_enabled = not streaming_enabled
                    print(f"✅ Streaming {'enabled' if streaming_enabled else 'disabled'}")
                    continue
                case "/samples":
                    show_sample_tasks()
                    print("\n💡 Tip: Use /sample <n> to run a specific sample (e.g., /sample 1)")
                    continue
                case _:
                    print(123)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Universal Tool Calling Agent - Works on all platforms")
    parser.add_argument("--mode", choices=["single", "interactive"], default="interactive", help="Execution mode (default: interactive)")
    parser.add_argument("--task", type=str, help="Task to execute (for single mode)")
    parser.add_argument("--backend", choices=["vllm", "ollama", "auto"], default="auto", help="Backend to use (default: auto-detect)")
    parser.add_argument("--info", action="store_true", help="Show system information and exit")
    parser.add_argument("--stream", action="store_true", default=True, help="Enable streaming mode (default: True)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming mode")

    args = parser.parse_args()  # Get arguments

    # Header
    print("=" * 100)
    print("🚀 Universal Tool Calling Agent")
    print("=" * 100)
    # Show system info and exist
    if args.info:
        show_system_info()
        return 0
    # Initialize agent
    print("\n⚙️  Initializing agent...")
    backend = None if args.backend == "auto" else args.backend
    try:
        agent = ToolCallingAgent(backend=backend, model="qwen3:0.6b")
    except SystemExit:
        return 1
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return 1
    print(f"✅ Agent ready! Using {agent.backend_type} backend")
    # Execute based on mode
    stream_enabled = not args.no_stream if hasattr(args, "no_stream") else True
    if args.mode == "single":
        sample_tasks = get_sample_tasks()
        if not args.task:
            print("\n" + "=" * 80)
            print("SINGLE TASK MODE - No task provided")
            print("=" * 80)
            show_sample_tasks(sample_tasks)  # show all tasks
            try:
                choice = input(f"\nSelect a task number (1-{len(sample_tasks)}) or 'q' to quit: ").strip()
                task_num = int(choice)
                if task_num >= 1 and task_num <= len(sample_tasks):
                    selected_task = sample_tasks[task_num - 1]
                    show_task_detail(selected_task)  # show task
                    confirm = input("\nRun this task? (y/n): ").strip().lower()
                    if confirm == "y":
                        run_single_task(agent, selected_task["task"], stream=stream_enabled)
                    else:
                        print("Task cancelled")
                else:
                    print(f"Invalid selection. Please choose 1-{len(sample_tasks)}")
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
            return 0
        else:
            mapped_tasks = list(filter(lambda t: t["name"] == args.task, sample_tasks))
            if len(mapped_tasks) > 0:
                show_task_detail(mapped_tasks[0])  # show task
                run_single_task(agent, mapped_tasks[0]["task"], stream=stream_enabled)
            else:
                print(f"Unknown task name: {args.task}")
            return 0
    else:
        interactive_mode(agent, stream_enabled)


if __name__ == "__main__":
    sys.exit(main())
