import contextlib
import datetime
import io
import json
import math
import random
import re
import sys


def code_interpreter(code: str) -> dict:
    """
    Execute Python code in a full Python environment.
    This provides unrestricted access to Python's built-in functions and standard library.
    """
    try:
        # Strip markdown code blocks and other formatting
        code = re.sub(r"^```(?:python|py)?\s*\n", "", code.strip())
        code = re.sub(r"\n```\s*$", "", code)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
        # Strip any leading/trailing whitespace
        code = code.strip()
        # Convert common mathematical notation to Python syntax
        # Replace ^ with ** for exponentiation
        code = re.sub(r"\^", "**", code)
        # Create a full Python namespace with all builtins available
        # This gives the agent access to the complete Python environment
        namespace = {
            "__builtins__": __builtins__,
            "math": math,
            "random": random,
            "datetime": datetime,
            "sys": sys,
            "re": re,
            "json": json,
        }
        # Capture both stdout and stderr
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
            exec(code, namespace)
        # Get output and any error messages
        printed_output = output_buffer.getvalue()
        error_output = error_buffer.getvalue()
        # Try to get result from common variable names
        result = namespace.get("result", None)
        if result is None:
            for var_name in ["A", "total", "sum", "output", "answer", "final", "value"]:
                if var_name in namespace:
                    result = namespace[var_name]
                    break
        response = {
            "result": result,
            "output": printed_output if printed_output else None,
            "stderr": error_output if error_output else None,
            "success": True,
        }

        return response

    except SyntaxError as e:
        error_msg = f"Syntax Error on line {e.lineno}: {e.msg}\n{e.text}"
        return {
            "error": error_msg,
            "error_type": "SyntaxError",
            "success": False,
        }
    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": error_trace,
            "success": False,
        }
