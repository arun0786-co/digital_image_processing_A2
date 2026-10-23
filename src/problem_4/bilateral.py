import numpy as np


def bilateral(image, sigma_s, sigma_r, radius=None):
    """bilateral filter over a (2 radius + 1)^2 window, reflected edges.

    g(p) = sum_q w(p, q) f(q) / sum_q w(p, q),
    w(p, q) = exp(-|p - q|^2 / (2 sigma_s^2)) exp(-(f(p) - f(q))^2 / (2 sigma_r^2)).
    one shifted copy of the image per window offset, so the cost is (2r + 1)^2
    whole-image passes. for colour input (rows, cols, 3) the range distance is the
    euclidean distance between rgb vectors, so all channels share one weight."""
    radius = int(np.ceil(3 * sigma_s)) if radius is None else int(radius)
    rows, cols = image.shape[:2]
    colour = image.ndim == 3
    pad = [(radius, radius), (radius, radius)] + [(0, 0)] * (image.ndim - 2)
    padded = np.pad(image, pad, mode="reflect")
    numerator = np.zeros_like(image, dtype=np.float64)
    denominator = np.zeros((rows, cols) + ((1,) if colour else ()), dtype=np.float64)
    range_scale = -0.5 / sigma_r ** 2
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial = np.exp(-(dy * dy + dx * dx) / (2.0 * sigma_s ** 2))
            neighbour = padded[radius + dy:radius + dy + rows, radius + dx:radius + dx + cols]
            squared = (neighbour - image) ** 2
            if colour:
                squared = squared.sum(axis=2, keepdims=True)
            weight = spatial * np.exp(range_scale * squared)
            numerator += weight * neighbour
            denominator += weight
    return numerator / denominator
