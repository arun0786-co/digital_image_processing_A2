import numpy as np

from common.local_stats import integral_image


def quadrant_stats(image, half):
    """means and variances of the four (half + 1) x (half + 1) sub-windows of the
    (2 half + 1) x (2 half + 1) window around each pixel. the sub-windows share the
    centre row and column. returns two arrays of shape (4, rows, cols), in the order
    top-left, top-right, bottom-left, bottom-right."""
    rows, cols = image.shape
    size = half + 1
    padded = np.pad(image, half, mode="reflect")
    sums = integral_image(padded)
    squares = integral_image(padded ** 2)
    means = np.empty((4, rows, cols))
    variances = np.empty((4, rows, cols))
    for k, (oy, ox) in enumerate([(0, 0), (0, half), (half, 0), (half, half)]):
        def window(total):
            return (total[oy + size:oy + size + rows, ox + size:ox + size + cols]
                    - total[oy:oy + rows, ox + size:ox + size + cols]
                    - total[oy + size:oy + size + rows, ox:ox + cols]
                    + total[oy:oy + rows, ox:ox + cols])
        mean = window(sums) / size ** 2
        means[k] = mean
        variances[k] = np.maximum(window(squares) / size ** 2 - mean ** 2, 0.0)
    return means, variances


def kuwahara(image, window):
    """classical kuwahara: the mean of the sub-window with the smallest variance."""
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be an odd size of at least 3")
    means, variances = quadrant_stats(image, window // 2)
    best = np.argmin(variances, axis=0)
    return np.take_along_axis(means, best[None], axis=0)[0]


def soft_kuwahara(image, window, temperature):
    """modified kuwahara: a softmax over -variance / temperature replaces the hard
    argmin, so the output moves smoothly between sub-window means instead of
    jumping when the lowest-variance sub-window changes."""
    if window % 2 == 0 or window < 3:
        raise ValueError("window must be an odd size of at least 3")
    means, variances = quadrant_stats(image, window // 2)
    # subtract the minimum first so the exponentials cannot underflow to all zeros
    logits = -(variances - variances.min(axis=0)) / temperature
    weights = np.exp(logits)
    return np.sum(weights * means, axis=0) / np.sum(weights, axis=0)
