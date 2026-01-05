from athena_core.builtin_tools import get_current_time
from athena_core.tools.tool_register import Tool

tool_get_current_time = Tool(
    function=get_current_time,
    description="Get the current date and time in a specific timezone",
    name="get_current_time",
    parameters=[
        Tool.Parameter(
            "timezone",
            "string",
            "Timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Shanghai'). Use standard IANA timezone names.",
            default="UTC",
        )
    ],
)
