import os
from pathlib import Path

_runtime_dlls = Path(__file__).resolve().parents[1] / ".runtime-dlls"
if _runtime_dlls.is_dir() and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(str(_runtime_dlls))
    except Exception:
        pass
os.environ["PATH"] = str(_runtime_dlls) + os.pathsep + os.environ.get("PATH", "")

import pymol

pymol.__path__.append(".")
pymol.__path__.append("tests/helpers")

collect_ignore = [
    "tests/helpers",
]
