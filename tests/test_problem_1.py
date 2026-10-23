import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from problem_1.dft import (circular_convolve, circular_convolve_fft, dft2, dft_matrix,
                           idft2, linear_convolve_fft, recover_sinusoids, sinusoid_image)


class DftTests(unittest.TestCase):
    def test_matrix_is_unitary_up_to_scale(self):
        w = dft_matrix(16)
        self.assertTrue(np.allclose(w @ np.conj(w).T, 16 * np.eye(16)))
        self.assertTrue(np.allclose(w, w.T))

    def test_matches_numpy_on_non_square_input(self):
        image = np.random.default_rng(0).normal(size=(12, 20))
        self.assertLess(np.abs(dft2(image) - np.fft.fft2(image)).max(), 1e-10)

    def test_inverse_round_trip(self):
        image = np.random.default_rng(1).uniform(size=(16, 10))
        self.assertLess(np.abs(idft2(dft2(image)) - image).max(), 1e-12)

    def test_impulse_has_flat_spectrum(self):
        image = np.zeros((8, 8))
        image[0, 0] = 1.0
        self.assertTrue(np.allclose(dft2(image), np.ones((8, 8))))

    def test_conjugate_symmetry_for_real_input(self):
        image = np.random.default_rng(2).normal(size=(9, 14))
        spectrum = dft2(image)
        mirrored = np.roll(spectrum[::-1, ::-1], 1, axis=(0, 1))
        self.assertLess(np.abs(spectrum - np.conj(mirrored)).max(), 1e-10)


class SinusoidTests(unittest.TestCase):
    def test_recovers_integer_frequency_components(self):
        truth = [(3, 5, 10.0, 0.4), (0, 7, 4.0, -1.2), (6, -2, 2.5, 2.9)]
        image = sinusoid_image(32, truth, offset=50.0)
        found, offset = recover_sinusoids(np.fft.fft2(image))
        self.assertAlmostEqual(offset, 50.0, places=9)
        self.assertEqual(len(found), 3)
        by_frequency = {(u, v): (a, p) for u, v, a, p in found}
        for u, v, amplitude, phase in truth:
            got_amplitude, got_phase = by_frequency[(u, v)]
            self.assertAlmostEqual(got_amplitude, amplitude, places=9)
            self.assertAlmostEqual(got_phase, phase, places=9)


class ConvolutionTests(unittest.TestCase):
    def test_circular_spatial_matches_fft(self):
        rng = np.random.default_rng(3)
        image = rng.normal(size=(20, 24))
        for shape in [(3, 3), (5, 5), (4, 6)]:
            kernel = rng.normal(size=shape)
            spatial = circular_convolve(image, kernel)
            self.assertLess(np.abs(spatial - circular_convolve_fft(image, kernel)).max(), 1e-10)

    def test_linear_fft_matches_scipy_same(self):
        from scipy.signal import convolve2d  # checking only
        rng = np.random.default_rng(4)
        image = rng.normal(size=(18, 22))
        for shape in [(5, 5), (4, 4), (3, 6)]:
            kernel = rng.normal(size=shape)
            reference = convolve2d(image, kernel, mode="same")
            self.assertLess(np.abs(linear_convolve_fft(image, kernel) - reference).max(), 1e-10)

    def test_circular_and_linear_agree_away_from_border(self):
        rng = np.random.default_rng(5)
        image = rng.normal(size=(32, 32))
        kernel = rng.normal(size=(5, 5))
        diff = np.abs(circular_convolve_fft(image, kernel) - linear_convolve_fft(image, kernel))
        self.assertLess(diff[2:-2, 2:-2].max(), 1e-10)
        self.assertGreater(diff[:2].max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
