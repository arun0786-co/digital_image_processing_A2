import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.local_stats import box_mean, local_std
from problem_3.homomorphic import LOG_OFFSET, equalise, homomorphic, homomorphic_transfer


class LocalStatsTests(unittest.TestCase):
    def test_box_mean_matches_direct_sum(self):
        image = np.random.default_rng(0).uniform(size=(12, 15))
        padded = np.pad(image, 2, mode="reflect")
        direct = np.array([[padded[i:i + 5, j:j + 5].mean() for j in range(15)] for i in range(12)])
        self.assertLess(np.abs(box_mean(image, 5) - direct).max(), 1e-12)

    def test_local_std_of_constant_is_zero(self):
        self.assertLess(local_std(np.full((10, 10), 3.0), 3).max(), 1e-7)


class HomomorphicTests(unittest.TestCase):
    def test_transfer_limits(self):
        transfer = homomorphic_transfer(256, 20, 0.5, 2.0)
        self.assertAlmostEqual(transfer[128, 128], 0.5)
        self.assertAlmostEqual(transfer[0, 0], 2.0, places=6)
        self.assertAlmostEqual(transfer[128 + 20, 128], 1.5 * (1 - np.exp(-1)) + 0.5)

    def test_unit_gains_return_input(self):
        image = np.random.default_rng(1).uniform(0.05, 1.0, size=(30, 40))
        out = homomorphic(image, 10, 1.0, 1.0)
        self.assertLess(np.abs(out - image).max(), 1e-10)

    def test_low_gain_scales_a_constant_log(self):
        # a constant image only has dc, so the output is (I + offset)^gL - offset
        image = np.full((16, 16), 0.2)
        out = homomorphic(image, 5, 0.5, 2.0)
        self.assertLess(np.abs(out - ((0.2 + LOG_OFFSET) ** 0.5 - LOG_OFFSET)).max(), 1e-10)

    def test_equalise_is_monotone_and_spreads(self):
        image = np.random.default_rng(2).integers(40, 80, size=(64, 64))
        out = equalise(image)
        order = np.argsort(image.ravel(), kind="stable")
        self.assertTrue(np.all(np.diff(out.ravel()[order].astype(int)) >= 0))
        self.assertEqual(out.max(), 255)


if __name__ == "__main__":
    unittest.main()
