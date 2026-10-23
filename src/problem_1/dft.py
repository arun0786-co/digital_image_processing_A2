import numpy as np


def dft_matrix(n):
    """W[u, x] = exp(-j 2 pi u x / n), unnormalised, same convention as numpy.fft."""
    index = np.arange(n)
    # reduce u x mod n first so the exponent stays small and exact
    return np.exp(-2j * np.pi * (np.outer(index, index) % n) / n)


def dft2(image):
    """2-d dft by row-column separability, F = W_M f W_N^T."""
    rows, cols = image.shape
    return dft_matrix(rows) @ image @ dft_matrix(cols).T


def idft2(spectrum):
    """inverse of dft2, f = conj(W_M) F conj(W_N)^T / (M N)."""
    rows, cols = spectrum.shape
    return np.conj(dft_matrix(rows)) @ spectrum @ np.conj(dft_matrix(cols)).T / (rows * cols)


def signed_index(index, n):
    """map a dft index in [0, n) to the signed frequency in [-n/2, n/2)."""
    return index - n if index >= n / 2 else index


def sinusoid_image(size, components, offset=0.0):
    """sum of A cos(2 pi (u x + v y) / size + phi), x the row and y the column index.
    components is a list of (u, v, amplitude, phase) in cycles per image."""
    x = np.arange(size)[:, None]
    y = np.arange(size)[None, :]
    image = np.full((size, size), float(offset))
    for u, v, amplitude, phase in components:
        image += amplitude * np.cos(2 * np.pi * (u * x + v * y) / size + phase)
    return image


def recover_sinusoids(spectrum, relative_threshold=1e-3):
    """read (u, v, amplitude, phase) of every cosine from the spectral peaks.

    each real cosine gives a conjugate pair at (u, v) and (-u, -v) with value
    (A / 2) M N exp(+-j phi). the pair member in the half plane u > 0, or u = 0
    and v > 0, is reported. returns the components and the recovered dc offset."""
    rows, cols = spectrum.shape
    magnitude = np.abs(spectrum).copy()
    offset = float(np.real(spectrum[0, 0])) / (rows * cols)
    magnitude[0, 0] = 0.0
    threshold = relative_threshold * magnitude.max()

    components = []
    while magnitude.max() > threshold:
        u, v = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        value = spectrum[u, v]
        magnitude[u, v] = 0.0
        magnitude[(-u) % rows, (-v) % cols] = 0.0

        su, sv = signed_index(u, rows), signed_index(v, cols)
        if su < 0 or (su == 0 and sv < 0):
            su, sv, value = -su, -sv, np.conj(value)
        # a nyquist-only frequency is its own conjugate, so it is not split in two
        self_conjugate = (2 * u) % rows == 0 and (2 * v) % cols == 0
        scale = 1.0 if self_conjugate else 2.0
        components.append((int(su), int(sv), scale * abs(value) / (rows * cols),
                           float(np.angle(value))))
    return components, offset


def centred_kernel_grid(kernel, rows, cols):
    """embed a small kernel in a rows x cols array with its origin, index
    ((kh - 1) // 2, (kw - 1) // 2) as in scipy mode="same", at (0, 0),
    wrapping negative offsets, which is the layout circular convolution expects."""
    grid = np.zeros((rows, cols))
    kh, kw = kernel.shape
    ch, cw = (kh - 1) // 2, (kw - 1) // 2
    for i in range(kh):
        for j in range(kw):
            grid[(i - ch) % rows, (j - cw) % cols] += kernel[i, j]
    return grid


def circular_convolve(image, kernel):
    """direct circular convolution, g[x, y] = sum h[m, n] f[x - m, y - n] (mod size),
    with the kernel centre as its origin. one shifted copy of the image per tap."""
    kh, kw = kernel.shape
    ch, cw = (kh - 1) // 2, (kw - 1) // 2
    out = np.zeros_like(image, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * np.roll(image, (i - ch, j - cw), axis=(0, 1))
    return out


def linear_convolve_fft(image, kernel, size=None):
    """zero-padded fft convolution, cropped to the 'same' size of the image.
    size is the padded grid (rows, cols); by default the full N + M - 1, which
    makes it exactly linear. smaller grids wrap around, so the crop is taken mod size."""
    rows, cols = image.shape
    kh, kw = kernel.shape
    if size is None:
        size = (rows + kh - 1, cols + kw - 1)
    padded_image = np.zeros(size)
    padded_image[:rows, :cols] = image
    padded_kernel = np.zeros(size)
    padded_kernel[:kh, :kw] = kernel
    full = np.real(np.fft.ifft2(np.fft.fft2(padded_image) * np.fft.fft2(padded_kernel)))
    top, left = (kh - 1) // 2, (kw - 1) // 2
    return np.roll(full, (-top, -left), axis=(0, 1))[:rows, :cols]


def circular_convolve_fft(image, kernel):
    """unpadded fft convolution: the image grid itself is the period."""
    grid = centred_kernel_grid(kernel, *image.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(grid)))
