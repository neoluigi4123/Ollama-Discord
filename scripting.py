"""
scripting.py
Execute Python scripts in a restricted environment.
"""
import io
import contextlib

def run_script(script: str) -> str:
    """
    Executes a Python script within a restricted environment.
    
    Args:
        script (str): Python code to execute.
    
    Returns:
        str: The output or error of the script execution.
    """
    # Redirect stdout and stderr to capture output
    output = io.StringIO()
    restricted_globals = {
        "__builtins__": {
            "print": print,
            "range": range,
            "len": len,
            "str": str,
            "int": int,
            "__import__": __import__,
        }
    }
    restricted_locals = {}

    try:
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exec(script, restricted_globals, restricted_locals)
    except Exception as e:
        return f"Error during execution: {e}"
    
    return output.getvalue()