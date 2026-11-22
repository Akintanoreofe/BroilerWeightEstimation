"""Small helpers for the Jupyter notebooks (folder prompts, environment report)."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def ask_for_directory(message: str, env_var: str | None = None, default: str | None = None,
                      must_exist: bool = True, create: bool = False) -> Path:
    """Prompt for a folder path and validate it.

    The answer can be pasted straight from Finder ("Copy as Pathname"):
    surrounding quotes and a trailing slash are removed and ``~`` is expanded.
    Set the environment variable ``env_var`` to skip the prompt (useful when
    re-running the notebook).

    Parameters
    ----------
    message : str
        Prompt text.
    env_var : str, optional
        Environment variable that, if set, is used instead of prompting.
    default : str, optional
        Used when the answer is empty.
    must_exist : bool
        Keep asking until the folder exists.
    create : bool
        Create the folder instead of asking again (for output folders).
    """
    preset = os.environ.get(env_var, "") if env_var else ""
    while True:
        raw = preset or input(f"{message}" + (f" [{default}]" if default else "") + ": ")
        preset = ""
        raw = raw.strip().strip("'\"").strip() or (default or "")
        if not raw:
            print("Please paste a folder path.")
            continue
        path = Path(raw).expanduser()
        if path.is_dir():
            print(f"Using: {path.resolve()}")
            return path.resolve()
        if create:
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {path.resolve()}")
            return path.resolve()
        if not must_exist:
            return path
        print(f"Folder not found: {path}. Try again.")


def environment_report() -> dict:
    """Versions and accelerators, so results can be reproduced."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "cv2", "torch", "transformers", "ultralytics"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except ImportError:
            info[mod] = "not installed"
    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        info["mps"] = bool(getattr(torch.backends, "mps", None)
                           and torch.backends.mps.is_available())
    except ImportError:
        pass
    return info
