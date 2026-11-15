import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import matplotlib
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from SliceVolumeEstimator import Circular, SliceVolumeEstimator
import visualize


@unittest.skipUnless(HAS_MATPLOTLIB, "install requirements-visualization.txt")
class VisualizationTest(unittest.TestCase):
    def test_slice_plot_contains_six_panels_and_saves(self):
        frame = np.full((220, 280, 3), 110, np.uint8)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.ellipse(mask, (140, 110), (70, 32), 35, 0, 360, 255, -1)
        result = SliceVolumeEstimator(20, Circular()).estimate(mask, 5.0)
        figure = visualize.plot_slices(frame, mask, result, 5.0)
        self.assertEqual(len(figure.axes), 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slices.png"
            figure.savefig(path)
            self.assertGreater(path.stat().st_size, 1000)
        import matplotlib.pyplot as plt
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
