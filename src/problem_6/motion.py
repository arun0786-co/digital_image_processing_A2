import numpy as np

from problem_1.dft import centred_kernel_grid

# angles are measured counter-clockwise from the +x (column) axis with y pointing up,
# so a point at distance t along the motion sits at column +t cos(theta) and row
# -t sin(theta) from the centre.


def motion_psf(length, angle_deg, samples_per_pixel=50):
    """discrete version of h(x, y) = (1/L) int_{-L/2}^{L/2} delta(x - t cos, y - t sin) dt,
    integrated over each pixel: a pixel's weight is the length of the segment inside
    its unit square divided by L. computed by sampling the segment densely at
    midpoints and counting the samples that land in each pixel."""
    theta = np.deg2rad(angle_deg)
    count = max(int(np.ceil(length * samples_per_pixel)), 1)
    t = (np.arange(count) + 0.5) / count * length - length / 2
    size = int(np.ceil(length)) + 3
    size += (size + 1) % 2
    centre = size // 2
    cols = np.rint(centre + t * np.cos(theta)).astype(int)
    rows = np.rint(centre - t * np.sin(theta)).astype(int)
    psf = np.zeros((size, size))
    np.add.at(psf, (rows, cols), 1.0)
    return psf / psf.sum()


def otf(psf, shape):
    """dft of the psf placed with its centre at index (0, 0) of an image-sized grid."""
    return np.fft.fft2(centred_kernel_grid(psf, shape[0], shape[1]))


def frequency_grid(shape):
    """(fx, fy) in cycles per pixel for an uncentred dft grid, fy pointing up."""
    fy_rows = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    return np.broadcast_to(fx, shape), np.broadcast_to(-fy_rows, shape)


def continuous_otf(shape, length, angle_deg):
    """H(fx, fy) = sinc(L (fx cos theta + fy sin theta)), sinc(z) = sin(pi z) / (pi z)."""
    theta = np.deg2rad(angle_deg)
    fx, fy = frequency_grid(shape)
    return np.sinc(length * (fx * np.cos(theta) + fy * np.sin(theta)))


def apply_otf(image, transfer):
    """circular convolution in the frequency domain, per channel for rgb."""
    if image.ndim == 3:
        transfer = transfer[..., None]
    return np.real(np.fft.ifft2(np.fft.fft2(image, axes=(0, 1)) * transfer, axes=(0, 1)))


def noise_for_snr(clean, snr_db, rng):
    """white gaussian noise scaled so that 10 log10(sum f^2 / sum n^2) = snr_db
    on average, with f the clean image."""
    sigma = np.sqrt(np.mean(clean ** 2) / 10 ** (snr_db / 10))
    return sigma * rng.standard_normal(clean.shape), sigma


def naive_inverse(blurred, transfer, floor=1e-12):
    """F = G / H. only values with |H| below floor are replaced, by floor with the
    phase of H, to avoid dividing by an exact zero."""
    small = np.abs(transfer) < floor
    safe = np.where(small, floor * np.exp(1j * np.angle(transfer)), transfer)
    if blurred.ndim == 3:
        safe = safe[..., None]
    spectrum = np.fft.fft2(blurred, axes=(0, 1)) / safe
    return np.real(np.fft.ifft2(spectrum, axes=(0, 1))), spectrum


def wiener_deconvolve(blurred, transfer, k):
    """F = conj(H) G / (|H|^2 + K) with a scalar noise-to-signal ratio K."""
    gain = np.conj(transfer) / (np.abs(transfer) ** 2 + k)
    return apply_otf(blurred, gain)


def cepstrum(image):
    """real cepstrum ift{log |F|}, centred. a motion blur of length L at angle theta
    puts a negative spike at distance about L along the motion direction."""
    return np.fft.fftshift(np.real(np.fft.ifft2(np.log(np.abs(np.fft.fft2(image)) + 1e-8))))


def estimate_blur(image, min_length=3, max_length=80):
    """(angle_deg, length) from the most negative cepstrum value outside a small
    disc around the centre, searched in one half plane (the cepstrum is even)."""
    c = cepstrum(image)
    rows, cols = c.shape
    y, x = np.indices(c.shape)
    dy = rows // 2 - y          # up is positive
    dx = x - cols // 2
    radius = np.hypot(dx, dy)
    search = (radius >= min_length) & (radius <= max_length) & ((dy > 0) | ((dy == 0) & (dx > 0)))
    row, col = np.unravel_index(np.argmin(np.where(search, c, np.inf)), c.shape)
    # three-point parabola along each axis for a sub-pixel position of the spike
    peak_dy = rows // 2 - (row + parabola_offset(c[row - 1, col], c[row, col], c[row + 1, col]))
    peak_dx = col + parabola_offset(c[row, col - 1], c[row, col], c[row, col + 1]) - cols // 2
    angle = np.rad2deg(np.arctan2(peak_dy, peak_dx)) % 180.0
    return float(angle), float(np.hypot(peak_dx, peak_dy)), c


def parabola_offset(left, centre, right):
    """offset in (-0.5, 0.5) of the extremum of the parabola through three samples."""
    curvature = left - 2 * centre + right
    return 0.0 if curvature == 0 else float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
