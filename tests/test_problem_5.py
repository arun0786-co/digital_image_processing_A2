import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from problem_4.bilateral import bilateral
from problem_5.wiener import (estimate_noise_sigma, flat_region_sigma, noise_power_units, noise_tiles,
                              padded_spectrum, radial_average, remove_plane, wiener_denoise, wiener_gain)


class WienerTests(unittest.TestCase):
    def test_noise_power_units_match_parseval(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.1, size=(64, 64))
        spectrum, P = padded_spectrum(noise)
        self.assertAlmostEqual(np.mean(np.abs(spectrum) ** 2) / noise_power_units(0.1, P), 1.0, delta=0.05)

    def test_gain_limits(self):
        self.assertTrue(np.allclose(wiener_gain(np.array([0.0, 1e12]), 1.0), [0.0, 1.0]))

    def test_radial_average_of_radial_function_is_unchanged(self):
        from common.freq import distance_grid
        radius = np.round(distance_grid(32))
        self.assertTrue(np.allclose(radial_average(radius), radius))

    def test_zero_noise_returns_input(self):
        image = np.random.default_rng(1).uniform(size=(20, 30))
        out, _ = wiener_denoise(image, 0.0)
        self.assertLess(np.abs(out - image).max(), 1e-10)

    def test_denoising_reduces_error_on_smooth_image(self):
        rows, cols = np.indices((64, 64))
        clean = 0.5 + 0.3 * np.sin(rows / 9.0) * np.cos(cols / 7.0)
        noisy = clean + np.random.default_rng(2).normal(0, 0.1, clean.shape)
        out, _ = wiener_denoise(noisy, 0.1)
        self.assertLess(np.mean((out - clean) ** 2), 0.5 * np.mean((noisy - clean) ** 2))

    def test_noise_estimate_on_flat_and_plane(self):
        rng = np.random.default_rng(3)
        rows, cols = np.indices((200, 200))
        plane = 0.2 + 0.001 * rows + 0.002 * cols
        estimate = estimate_noise_sigma(plane + rng.normal(0, 0.05, plane.shape))
        self.assertAlmostEqual(estimate, 0.05, delta=0.003)

    def test_remove_plane_leaves_nothing_of_a_plane(self):
        rows, cols = np.indices((16, 20))
        tile = np.stack([0.3 + 0.01 * rows - 0.02 * cols, 0.5 + 0.03 * cols, 0.1 * np.ones((16, 20))], axis=2)
        self.assertLess(np.abs(remove_plane(tile)).max(), 1e-12)

    def test_flat_tiles_recover_white_noise_sigma_under_shading(self):
        rng = np.random.default_rng(5)
        rows, cols = np.indices((256, 256))
        shading = (0.2 + 0.002 * rows + 0.001 * cols)[..., None]
        image = shading + rng.normal(0, 0.02, (256, 256, 3))
        _, tiles = noise_tiles(image, 64)
        sigma = flat_region_sigma(tiles, 0.5)
        self.assertTrue(np.allclose(sigma, 0.02, rtol=0.03))


class ColourBilateralTests(unittest.TestCase):
    def test_grey_rgb_matches_grey_with_scaled_range(self):
        grey = np.random.default_rng(4).uniform(size=(12, 12))
        rgb = np.repeat(grey[..., None], 3, axis=2)
        # the rgb distance of a grey difference d is sqrt(3) d
        out_rgb = bilateral(rgb, 1.0, 0.2 * np.sqrt(3), radius=2)
        out_grey = bilateral(grey, 1.0, 0.2, radius=2)
        self.assertLess(np.abs(out_rgb[..., 1] - out_grey).max(), 1e-12)


if __name__ == "__main__":
    unittest.main()
