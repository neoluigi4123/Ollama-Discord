"""
conf_module.py
Configuration loader module
"""
import importlib.util
from pathlib import Path

def load_conf(varname=None, path="config.py") -> str:
    """
    Load configuration from a Python file.

    Args:
        varname (str, optional): Specific variable name to retrieve. If None, returns all variables.
        path (str): Path to the configuration file.
    
    Returns:
        str: The value of the specified variable or a formatted string of all variables.
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location("config", path)
    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    if varname is not None:
        return getattr(conf, varname)

    # pretty dump of all values
    def format_value(v, indent=0):
        prefix = " " * indent
        if isinstance(v, (list, tuple, set)):
            lines = [f"{prefix}- {format_value(x, indent+2).lstrip()}" for x in v]
            return "\n".join(lines)
        elif isinstance(v, dict):
            lines = [f"{prefix}{k}:" for k in v]
            for k, val in v.items():
                lines.append(format_value(val, indent+2))
            return "\n".join(lines)
        else:
            return f"{prefix}{v}"

    result_lines = []
    for k, v in conf.__dict__.items():
        if not k.startswith("__"):
            if isinstance(v, (list, tuple, set, dict)):
                result_lines.append(f"{k}:")
                result_lines.append(format_value(v, 2))
            else:
                result_lines.append(f"{k}: {v}")
    return "\n".join(result_lines)