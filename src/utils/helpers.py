import yaml
from pathlib import Path
from typing import Any, Dict, Union

def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Helper to load a YAML config file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
