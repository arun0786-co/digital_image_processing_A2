import numpy as np

from common.freq import centred_spectrum, distance_grid, pad_image
from common.io_utils import luminance
from common.local_stats import box_mean


def padded_spectrum(image):
    """mirror-pad to a square P = 2 max(rows, cols) and return the centred spectrum
    (per channel for rgb) and P. mirroring avoids the dark border that zero padding
    would leave after any low-pass filter."""
    P = 2 * max(image.shape[:2])
    return centred_spectrum(pad_image(image, P, P, mode="symmetric")), P


def invert(spectrum, shape):
    rows, cols = shape[:2]
    full = np.fft.ifft2(np.fft.ifftshift(spectrum, axes=(0, 1)), axes=(0, 1))
    return np.real(full)[:rows, :cols]


def per_channel(fn, power):
    if power.ndim == 2:
        return fn(power)
    return np.stack([fn(power[..., c]) for c in range(power.shape[2])], axis=2)


def radial_average(power):
    """replace each value by the mean over its ring of unit width around the centre."""
    radius = np.round(distance_grid(power.shape[0], power.shape[1])).astype(int)
    sums = np.bincount(radius.ravel(), weights=power.ravel())
    counts = np.bincount(radius.ravel())
    return (sums / np.maximum(counts, 1))[radius]


def smoothed_periodogram(power, size=15):
    """local box average of |G|^2 in the frequency plane, which keeps anisotropy."""
    return box_mean(power, size)


def signal_power(noisy_spectrum, noise_power, method):
    """S_ff estimate: smoothed noisy periodogram minus the assumed flat noise power,
    floored at zero. noise_power is in the same units as |G|^2."""
    power = np.abs(noisy_spectrum) ** 2
    smoother = radial_average if method == "radial" else smoothed_periodogram
    return np.maximum(per_channel(smoother, power) - noise_power, 0.0)


def wiener_gain(signal, noise_power):
    """W = S_ff / (S_ff + S_nn)."""
    return signal / (signal + noise_power)


def noise_power_units(sigma, P):
    """E|N(u, v)|^2 for white noise of per-pixel variance sigma^2 on an unnormalised
    P x P dft: by parseval the mean of |N|^2 over frequencies is P^2 sigma^2."""
    return P ** 2 * sigma ** 2


def wiener_denoise(noisy, sigma, method="radial", noise_scale=1.0):
    """white-noise wiener denoiser for an image in [0, 1]. sigma is the noise std in
    the same units; noise_scale multiplies the assumed noise power (for 5.2)."""
    spectrum, P = padded_spectrum(noisy)
    assumed = noise_scale * noise_power_units(sigma, P)
    gain = wiener_gain(signal_power(spectrum, assumed, method), assumed)
    return invert(gain * spectrum, noisy.shape), gain


def oracle_wiener(noisy, clean, sigma):
    """wiener gain built from the clean image's own periodogram: an upper bound that
    needs the reference, so it is labelled oracle."""
    spectrum, P = padded_spectrum(noisy)
    clean_spectrum, _ = padded_spectrum(clean)
    gain = wiener_gain(np.abs(clean_spectrum) ** 2, noise_power_units(sigma, P))
    return invert(gain * spectrum, noisy.shape)


def remove_plane(tile):
    """tile minus its least-squares plane a + b x + c y per channel, which takes out
    slow shading so that what is left is noise plus any fine texture."""
    rows, cols = tile.shape[:2]
    y, x = np.indices((rows, cols))
    design = np.stack([np.ones(rows * cols), x.ravel(), y.ravel()], axis=1)
    values = tile.reshape(rows * cols, -1)
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    return (values - design @ coefficients).reshape(tile.shape)


def noise_tiles(image, size):
    """split an rgb image into non-overlapping size x size tiles, remove a plane from
    each, and return (mean luminance of each tile, the plane-free tiles)."""
    means, tiles = [], []
    for top in range(0, image.shape[0] - size + 1, size):
        for left in range(0, image.shape[1] - size + 1, size):
            tile = image[top:top + size, left:left + size]
            means.append(float(luminance(tile).mean()))
            tiles.append(remove_plane(tile))
    return np.array(means), np.array(tiles)


def flat_region_sigma(tiles, share):
    """noise std per channel from flat regions: the plane-free std of each tile, taking
    the `share` quantile over tiles. edges and texture only make a tile's std larger,
    so the low end of the distribution comes from the flat tiles."""
    return np.quantile(tiles.std(axis=(1, 2)), share, axis=0)


def estimate_noise_sigma(image):
    """immerkaer (1996) estimate of white-noise std from the response to
    [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], which cancels local planes:
    sigma = sqrt(pi / 2) / (6 (H - 2) (W - 2)) * sum |response|."""
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    rows, cols = image.shape
    response = np.zeros((rows - 2, cols - 2))
    for i in range(3):
        for j in range(3):
            response += mask[i, j] * image[i:i + rows - 2, j:j + cols - 2]
    return float(np.sqrt(np.pi / 2) * np.abs(response).sum() / (6 * (rows - 2) * (cols - 2)))
