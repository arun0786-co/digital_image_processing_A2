import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.metrics import label_components, ssim


class SsimTests(unittest.TestCase):
    def test_matches_skimage_grey(self):
        from skimage.metrics import structural_similarity
        rng = np.random.default_rng(3)
        clean = rng.uniform(0, 255, size=(40, 57))
        noisy = clean + rng.normal(0, 20, clean.shape)
        expected = structural_similarity(clean, noisy, data_range=255)
        self.assertAlmostEqual(ssim(clean, noisy, 255), expected, places=12)

    def test_matches_skimage_colour(self):
        from skimage.metrics import structural_similarity
        rng = np.random.default_rng(4)
        clean = rng.uniform(0, 1, size=(33, 41, 3))
        noisy = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1)
        expected = structural_similarity(clean, noisy, data_range=1.0, channel_axis=-1)
        self.assertAlmostEqual(ssim(clean, noisy, 1.0), expected, places=12)

    def test_identical_images_score_one(self):
        image = np.random.default_rng(5).uniform(size=(20, 20))
        self.assertAlmostEqual(ssim(image, image, 1.0), 1.0, places=12)


class ComponentTests(unittest.TestCase):
    def test_matches_scipy_label(self):
        from scipy.ndimage import label
        rng = np.random.default_rng(6)
        for density in (0.2, 0.45, 0.6):
            mask = rng.uniform(size=(60, 80)) < density
            expected, count = label(mask, structure=np.ones((3, 3)))
            labels, found = label_components(mask)
            self.assertEqual(found, count)
            # same partition: each of our pieces maps to exactly one scipy piece
            pairs = set(zip(labels[mask].tolist(), expected[mask].tolist()))
            self.assertEqual(len(pairs), count)

    def test_diagonal_touch_joins_and_empty_mask(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[1, 1] = mask[2, 2] = mask[4, 0] = True
        labels, count = label_components(mask)
        self.assertEqual(count, 2)
        self.assertEqual(labels[1, 1], labels[2, 2])
        self.assertEqual(label_components(np.zeros((3, 3), dtype=bool))[1], 0)


if __name__ == "__main__":
    unittest.main()
