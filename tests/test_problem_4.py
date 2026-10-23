import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from problem_4.bilateral import bilateral
from problem_4.kuwahara import kuwahara, quadrant_stats, soft_kuwahara


def direct_quadrants(image, half, y, x):
    padded = np.pad(image, half, mode="reflect")
    py, px = y + half, x + half
    blocks = [padded[py - half:py + 1, px - half:px + 1], padded[py - half:py + 1, px:px + half + 1],
              padded[py:py + half + 1, px - half:px + 1], padded[py:py + half + 1, px:px + half + 1]]
    return [b.mean() for b in blocks], [b.var() for b in blocks]


class KuwaharaTests(unittest.TestCase):
    def test_quadrant_stats_match_direct(self):
        image = np.random.default_rng(0).uniform(0, 255, size=(14, 17))
        means, variances = quadrant_stats(image, 2)
        for y, x in [(0, 0), (5, 7), (13, 16), (3, 15)]:
            m, v = direct_quadrants(image, 2, y, x)
            self.assertTrue(np.allclose(means[:, y, x], m))
            self.assertTrue(np.allclose(variances[:, y, x], v))

    def test_step_edge_is_kept_exactly(self):
        image = np.zeros((20, 20))
        image[:, 10:] = 100.0
        self.assertTrue(np.array_equal(kuwahara(image, 5), image))

    def test_constant_and_shape(self):
        image = np.full((9, 11), 42.0)
        self.assertTrue(np.allclose(kuwahara(image, 7), 42.0))
        self.assertTrue(np.allclose(soft_kuwahara(image, 7, 10.0), 42.0))
        self.assertEqual(kuwahara(image, 3).shape, (9, 11))

    def test_soft_tends_to_hard_for_small_temperature(self):
        image = np.random.default_rng(1).uniform(0, 255, size=(16, 16))
        self.assertLess(np.abs(soft_kuwahara(image, 5, 1e-9) - kuwahara(image, 5)).max(), 1e-6)

    def test_rejects_even_window(self):
        with self.assertRaises(ValueError):
            kuwahara(np.zeros((5, 5)), 4)


class BilateralTests(unittest.TestCase):
    def test_matches_direct_formula(self):
        rng = np.random.default_rng(2)
        image = rng.uniform(0, 255, size=(10, 12))
        out = bilateral(image, 1.5, 30.0, radius=2)
        padded = np.pad(image, 2, mode="reflect")
        y, x = 4, 6
        offsets = np.arange(-2, 3)
        dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
        patch = padded[y:y + 5, x:x + 5]
        w = np.exp(-(dy ** 2 + dx ** 2) / (2 * 1.5 ** 2)) * np.exp(-(patch - image[y, x]) ** 2 / (2 * 30.0 ** 2))
        self.assertAlmostEqual(out[y, x], np.sum(w * patch) / np.sum(w))

    def test_large_range_sigma_is_a_gaussian_blur(self):
        image = np.random.default_rng(3).uniform(0, 255, size=(12, 12))
        out = bilateral(image, 1.0, 1e9, radius=3)
        padded = np.pad(image, 3, mode="reflect")
        offsets = np.arange(-3, 4)
        kernel = np.exp(-(offsets[:, None] ** 2 + offsets[None, :] ** 2) / 2.0)
        kernel /= kernel.sum()
        self.assertAlmostEqual(out[6, 6], np.sum(kernel * padded[6:13, 6:13]))

    def test_small_range_sigma_keeps_edges(self):
        image = np.zeros((16, 16))
        image[:, 8:] = 100.0
        self.assertLess(np.abs(bilateral(image, 2.0, 5.0) - image).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
