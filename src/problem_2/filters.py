import numpy as np

from common.freq import distance_grid, filter_image, padded_size

# the handout fixes this signature, including the capitalised P and D0
def make_filter(kind, P, D0, n=None, highpass=False):
    """P x P centred transfer function.

    ideal:        1 for D <= D0, else 0
    butterworth:  1 / (1 + (D / D0)^(2n))
    gaussian:     exp(-D^2 / (2 D0^2))
    highpass returns 1 - lowpass."""
    distance = distance_grid(P)
    if kind == "ideal":
        lowpass = (distance <= D0).astype(np.float64)
    elif kind == "butterworth":
        if n is None:
            raise ValueError("a butterworth filter needs an order n")
        lowpass = 1.0 / (1.0 + (distance / D0) ** (2 * n))
    elif kind == "gaussian":
        lowpass = np.exp(-distance ** 2 / (2.0 * D0 ** 2))
    else:
        raise ValueError(f"unknown filter kind '{kind}'. choose ideal, butterworth or gaussian")
    return 1.0 - lowpass if highpass else lowpass


def matched_gaussian_d0(radius):
    """gaussian D0' whose half-amplitude point (H = 0.5) sits at the given radius."""
    return radius / np.sqrt(2.0 * np.log(2.0))


def apply_filter(image, kind, D0, n=None, highpass=False, P=None):
    """the problem 2 pipeline: zero-pad to P = 2 max(rows, cols), filter, crop."""
    P = padded_size(image) if P is None else P
    return filter_image(image, make_filter(kind, P, D0, n=n, highpass=highpass))


def radial_slice(transfer):
    """H(u, 0) along the row through the centre, ordered from u = -P/2 to P/2 - 1."""
    return transfer[:, transfer.shape[1] // 2]


def slice_kernel(transfer):
    """1-d spatial kernel h(x) = ift{H(u, 0)}, returned with x = 0 at index 0."""
    return np.real(np.fft.ifft(np.fft.ifftshift(radial_slice(transfer))))
