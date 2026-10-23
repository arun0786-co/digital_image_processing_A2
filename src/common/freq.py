import numpy as np


def pad_image(image, rows, cols, mode="constant"):
    """pad the first two axes at the bottom and right up to rows x cols."""
    extra_rows = rows - image.shape[0]
    extra_cols = cols - image.shape[1]
    if extra_rows < 0 or extra_cols < 0:
        raise ValueError("padded size is smaller than the image")
    pad = [(0, extra_rows), (0, extra_cols)] + [(0, 0)] * (image.ndim - 2)
    return np.pad(image, pad, mode=mode)


def padded_size(image):
    """P = 2 max(rows, cols), the square grid used by the problem 2 pipeline."""
    return 2 * max(image.shape[:2])


def distance_grid(rows, cols=None):
    """D(u, v) = sqrt((u - P/2)^2 + (v - Q/2)^2) on a centred rows x cols grid."""
    cols = rows if cols is None else cols
    u = np.arange(rows) - rows / 2
    v = np.arange(cols) - cols / 2
    return np.sqrt(u[:, None] ** 2 + v[None, :] ** 2)


def centred_spectrum(image):
    return np.fft.fftshift(np.fft.fft2(image, axes=(0, 1)), axes=(0, 1))


def filter_image(image, transfer, pad_mode="constant"):
    """pad, centre the spectrum, multiply by a real centred transfer function,
    invert, keep the real part and crop. works per channel for rgb input."""
    rows, cols = image.shape[:2]
    padded = pad_image(image, transfer.shape[0], transfer.shape[1], pad_mode)
    spectrum = centred_spectrum(padded)
    if spectrum.ndim == 3:
        transfer = transfer[..., None]
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * transfer, axes=(0, 1)), axes=(0, 1))
    return np.real(filtered)[:rows, :cols]


def log_magnitude(spectrum):
    return np.log1p(np.abs(spectrum))
