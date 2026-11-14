import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ValidationNotebookTest(unittest.TestCase):
    def test_repository_discovery_and_override(self):
        notebook = json.loads((ROOT / "validation_walkthrough.ipynb").read_text())
        setup = "".join(notebook["cells"][1]["source"])
        with patch.object(Path, "cwd", return_value=ROOT / "src"):
            namespace = {}
            exec(setup, namespace)
            self.assertEqual(namespace["ROOT"], ROOT)
            self.assertEqual(namespace["EXPORT_DIR"], ROOT / "dataset")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(Path, "cwd", return_value=Path(tmp)):
                with self.assertRaisesRegex(FileNotFoundError, "Set REPO_ROOT"):
                    exec(setup, {})
                namespace = {}
                exec(setup.replace("REPO_ROOT = None", f"REPO_ROOT = {str(ROOT)!r}", 1), namespace)
                self.assertEqual(namespace["ROOT"], ROOT)
                with self.assertRaisesRegex(FileNotFoundError, "REPO_ROOT does not contain"):
                    exec(setup.replace("REPO_ROOT = None", f"REPO_ROOT = {tmp!r}", 1), {})


if __name__ == "__main__":
    unittest.main()
