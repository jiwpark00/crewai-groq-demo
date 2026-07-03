import sys
from pathlib import Path

# crewai_groq_demo is not an installed/editable package (no [build-system] in
# pyproject.toml), so it's only importable when the repo root is on sys.path.
# Running scripts (`uv run main.py`) gets this for free via Python's implicit
# script-directory insertion; pytest does not, since tests/ has no
# __init__.py and pytest's "prepend" import mode only adds tests/ itself.
# Insert the repo root explicitly so imports resolve the same way regardless
# of collection order or how pytest is invoked (bare, single file, etc.).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
