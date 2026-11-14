"""Launch the FlockLabel desktop app: ``python -m flocklabel [project_dir]``."""

import sys


def main() -> int:
    try:
        from flocklabel.ui.main_window import run
    except ImportError as err:  # PySide6 missing
        print(f"Cannot start the GUI: {err}\nInstall the requirements: pip install -r requirements.txt")
        return 1
    return run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
