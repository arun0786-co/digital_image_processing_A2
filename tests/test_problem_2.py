import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common.freq import filter_image
from problem_2.filters import apply_filter, make_filter, matched_gaussian_d0, slice_kernel


def value_at(transfer, offset):
    centre = transfer.shape[0] // 2
    return transfer[centre + offset, centre]


class TransferFunctionTests(unittest.TestCase):
    def test_values_at_cutoff(self):
        self.assertEqual(value_at(make_filter("ideal", 128, 20), 20), 1.0)
        self.assertEqual(value_at(make_filter("ideal", 128, 20), 21), 0.0)
        self.assertAlmostEqual(value_at(make_filter("butterworth", 128, 20, n=3), 20), 0.5)
        self.assertAlmostEqual(value_at(make_filter("gaussian", 128, 20), 20), np.exp(-0.5))

    def test_dc_and_highpass(self):
        for kind, n in [("ideal", None), ("butterworth", 2), ("gaussian", None)]:
            low = make_filter(kind, 64, 10, n=n)
            high = make_filter(kind, 64, 10, n=n, highpass=True)
            self.assertEqual(low[32, 32], 1.0)
            self.assertEqual(high[32, 32], 0.0)
            self.assertTrue(np.allclose(low + high, 1.0))
            self.assertEqual(low.shape, (64, 64))

    def test_matched_gaussian_half_amplitude(self):
        transfer = make_filter("gaussian", 256, matched_gaussian_d0(30))
        self.assertAlmostEqual(value_at(transfer, 30), 0.5)

    def test_inverse_design_answer(self):
        transfer = make_filter("butterworth", 256, 20, n=2)
        self.assertAlmostEqual(value_at(transfer, 10), 16 / 17)
        self.assertAlmostEqual(value_at(transfer, 40), 1 / 17)

    def test_radial_symmetry(self):
        transfer = make_filter("butterworth", 64, 9, n=2)
        self.assertTrue(np.allclose(transfer[1:, 1:], transfer[1:, 1:][::-1, ::-1]))
        self.assertTrue(np.allclose(transfer, transfer.T))

    def test_needs_order_and_known_kind(self):
        with self.assertRaises(ValueError):
            make_filter("butterworth", 32, 5)
        with self.assertRaises(ValueError):
            make_filter("box", 32, 5)


class PipelineTests(unittest.TestCase):
    def test_all_pass_returns_input(self):
        image = np.random.default_rng(0).uniform(0, 255, size=(20, 30))
        out = filter_image(image, np.ones((60, 60)))
        self.assertEqual(out.shape, image.shape)
        self.assertLess(np.abs(out - image).max(), 1e-10)

    def test_lowpass_keeps_constant_away_from_padding(self):
        image = np.full((64, 64), 100.0)
        # D0 = 10 keeps H(Nyquist) ~ 1e-9; a wider gaussian is cut off at the grid edge
        out = apply_filter(image, "gaussian", 10)
        self.assertLess(np.abs(out[24:40, 24:40] - 100).max(), 1e-6)

    def test_highpass_padded_output_has_zero_mean(self):
        image = np.random.default_rng(1).uniform(0, 255, size=(32, 32))
        transfer = make_filter("butterworth", 64, 8, n=4, highpass=True)
        padded = np.zeros((64, 64))
        padded[:32, :32] = image
        out = filter_image(padded, transfer)
        self.assertLess(abs(out.mean()), 1e-9)

    def test_gaussian_kernel_is_positive_and_ideal_is_not(self):
        gaussian = slice_kernel(make_filter("gaussian", 512, 20))
        ideal = slice_kernel(make_filter("ideal", 512, 20))
        self.assertGreater(gaussian.min(), -1e-15)
        self.assertLess(ideal.min(), -1e-3)
        self.assertAlmostEqual(gaussian.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
