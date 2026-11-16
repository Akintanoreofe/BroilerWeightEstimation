import json
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ValidationNotebookTest(unittest.TestCase):
    def test_full_run_cell_does_not_require_platform_polygons(self):
        notebook = json.loads((ROOT / "src" / "validation_walkthrough.ipynb").read_text())
        code = next("".join(cell["source"]) for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                    and "str(ROOT / 'src' / 'run_eval.py')" in "".join(cell["source"]))
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "results"
            commands = []

            def run_command(command):
                commands.append(command)
                (results / "bird_results.csv").write_text("frame,volume\n1,1.0\n")

            namespace = dict(ROOT=ROOT, EXPORT_DIR=Path(tmp), RESULTS_DIR=results,
                             sys=sys, csv=csv, Path=Path, DENSITY=900, SLICE_COUNT=None,
                             ALLOW_FRAME_EDGE=True,
                             names=["bird.mp4"], run_command=run_command, display=lambda rows: None)
            exec(code, namespace)
            self.assertEqual(len(commands), 1)
            self.assertNotIn("--scale-polygons", commands[0])
            self.assertNotIn("--min-scale-overlap", commands[0])
            self.assertIn("--allow-frame-edge", commands[0])
            namespace["ALLOW_FRAME_EDGE"] = False
            exec(code, namespace)
            self.assertNotIn("--allow-frame-edge", commands[-1])

    def test_repository_discovery_and_override(self):
        notebook = json.loads((ROOT / "src" / "validation_walkthrough.ipynb").read_text())
        setup = "".join(notebook["cells"][1]["source"])
        for cwd in (ROOT, ROOT / "src"):
            with self.subTest(cwd=cwd), patch.object(Path, "cwd", return_value=cwd):
                namespace = {}
                exec(setup, namespace)
                self.assertEqual(namespace["ROOT"], ROOT)
                relative_setup = setup.replace(
                    "Path('dataset')",
                    "Path('another_dataset')",
                )
                exec(relative_setup, namespace)
                self.assertEqual(namespace["EXPORT_DIR"], ROOT / "another_dataset")
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(Path, "cwd", return_value=Path(tmp)):
                with self.assertRaisesRegex(FileNotFoundError, "Set REPO_ROOT"):
                    exec(setup, {})
                namespace = {}
                exec(setup.replace("REPO_ROOT = None", f"REPO_ROOT = {str(ROOT)!r}", 1), namespace)
                self.assertEqual(namespace["ROOT"], ROOT)
                with self.assertRaisesRegex(FileNotFoundError, "REPO_ROOT does not contain"):
                    exec(setup.replace("REPO_ROOT = None", f"REPO_ROOT = {tmp!r}", 1), {})
                with self.assertRaisesRegex(FileNotFoundError, "REPO_ROOT does not contain"):
                    exec(setup.replace("REPO_ROOT = None", f"REPO_ROOT = {str(ROOT / 'src' / 'run_eval.py')!r}", 1), {})


if __name__ == "__main__":
    unittest.main()
