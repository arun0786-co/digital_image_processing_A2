import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from problem_6.motion import (apply_otf, continuous_otf, estimate_blur, motion_psf, naive_inverse,
                              noise_for_snr, otf, wiener_deconvolve)


class PsfTests(unittest.TestCase):
    def test_horizontal_psf_is_a_box(self):
        psf = motion_psf(9, 0.0)
        centre = psf.shape[0] // 2
        row = psf[centre]
        self.assertAlmostEqual(psf.sum(), 1.0)
        self.assertAlmostEqual(row.sum(), 1.0)
        self.assertTrue(np.allclose(row[row > 0], 1 / 9))

    def test_psf_is_point_symmetric_and_has_right_direction(self):
        psf = motion_psf(15, 30.0)
        self.assertTrue(np.allclose(psf, psf[::-1, ::-1]))
        rows, cols = np.nonzero(psf > 1e-3)
        centre = psf.shape[0] // 2
        # up-right motion: upper part (smaller row) lies to the right
        upper = rows < centre
        self.assertGreater(cols[upper].mean(), centre)

    def test_vertical_psf(self):
        psf = motion_psf(7, 90.0)
        centre = psf.shape[0] // 2
        self.assertAlmostEqual(psf[:, centre].sum(), 1.0)


class OtfTests(unittest.TestCase):
    def test_horizontal_otf_zeros_at_m_over_l(self):
        shape = (64, 63 * 4)
        transfer = otf(motion_psf(9, 0.0), shape)
        # columns k with k / N = m / 9, i.e. k = 28 m for N = 252
        for m in [1, 2, 3]:
            self.assertLess(abs(transfer[0, 28 * m]), 1e-12)

    def test_discrete_close_to_continuous_at_low_frequency(self):
        shape = (128, 128)
        transfer = otf(motion_psf(11, 30.0), shape)
        model = continuous_otf(shape, 11, 30.0)
        low = (np.abs(np.fft.fftfreq(128))[:, None] < 0.1) & (np.abs(np.fft.fftfreq(128))[None, :] < 0.1)
        self.assertLess(np.abs(np.abs(transfer) - np.abs(model))[low].max(), 0.05)

    def test_inverse_is_exact_without_noise(self):
        image = np.random.default_rng(0).uniform(size=(48, 64))
        transfer = otf(motion_psf(5, 0.0), image.shape)
        blurred = apply_otf(image, transfer)
        restored, _ = naive_inverse(blurred, transfer)
        # width 64 and L = 5 put no dft sample on a zero of the otf
        self.assertGreater(np.abs(transfer).min(), 1e-3)
        self.assertLess(np.abs(restored - image).max(), 1e-9)

    def test_wiener_with_zero_k_matches_inverse(self):
        image = np.random.default_rng(1).uniform(size=(32, 32))
        transfer = otf(motion_psf(3, 45.0), image.shape)
        blurred = apply_otf(image, transfer)
        self.assertLess(np.abs(wiener_deconvolve(blurred, transfer, 0.0) - naive_inverse(blurred, transfer)[0]).max(), 1e-8)


class NoiseAndEstimateTests(unittest.TestCase):
    def test_snr_definition(self):
        clean = np.random.default_rng(2).uniform(size=(200, 200))
        noise, _ = noise_for_snr(clean, 30.0, np.random.default_rng(3))
        snr = 10 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2))
        self.assertAlmostEqual(snr, 30.0, delta=0.1)

    def test_cepstrum_finds_synthetic_blur(self):
        rng = np.random.default_rng(4)
        image = np.kron(rng.uniform(size=(64, 64)), np.ones((4, 4)))
        image = image + 0.2 * rng.uniform(size=image.shape)
        transfer = otf(motion_psf(17, 30.0), image.shape)
        angle, length, _ = estimate_blur(apply_otf(image, transfer))
        self.assertAlmostEqual(angle, 30.0, delta=4.0)
        self.assertAlmostEqual(length, 17.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
