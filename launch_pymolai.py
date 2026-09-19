"""Windows launcher for PyMolAI.

Adds native dependency DLL search paths (Python 3.8+ Windows requirement),
then starts the PyMOL Qt GUI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_dll_dir(path: Path) -> None:
    if not path.is_dir():
        return
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(path))
    # Keep PATH updated for child processes / older loaders.
    os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent
    runtime_dlls = root / ".runtime-dlls"
    conda_dlls = Path(r"C:\Users\msr4c\miniforge3\envs\pymolai\Library\bin")

    _add_dll_dir(runtime_dlls)
    if conda_dlls.is_dir():
        _add_dll_dir(conda_dlls)

    # Prefer the repo checkout for data/scripts when present.
    os.environ.setdefault("PYMOL_PATH", str(root))

    # Prefer in-repo Python modules so Phase 1 provider edits apply without reinstall.
    modules = root / "modules"
    if modules.is_dir():
        sys.path.insert(0, str(modules))

    args = ["pymol", *(argv if argv is not None else sys.argv[1:])]
    import pymol

    return int(pymol.launch(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
