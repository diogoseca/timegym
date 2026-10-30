"""Compatibility entry point; the maintained checker lives in notebooks/."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "notebooks" / "check_notebooks.py"), run_name="__main__")
