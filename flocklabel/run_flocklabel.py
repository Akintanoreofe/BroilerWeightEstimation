"""Start FlockLabel without installing it: ``python run_flocklabel.py [project_dir]``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flocklabel.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
