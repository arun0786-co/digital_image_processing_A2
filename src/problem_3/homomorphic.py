import numpy as np

from common.freq import distance_grid, filter_image

# offset inside the log so that black pixels stay finite: ln(I + 1/255)
LOG_OFFSET = 1.0 / 255.0


def homomorphic_transfer(P, D0, gamma_low, gamma_high, c=1.0):
    """H = (gH - gL) (1 - exp(-c D^2 / D0^2)) + gL on a centred P x P grid."""
    distance = distance_grid(P)
    return (gamma_high - gamma_low) * (1.0 - np.exp(-c * distance ** 2 / D0 ** 2)) + gamma_low


def homomorphic(image, D0, gamma_low, gamma_high, c=1.0):
    """image in [0, 1]. log, high-frequency emphasis in the log domain, exp back.
    the log image is padded by mirroring to P = 2 max(rows, cols): zero padding a
    log image would put a large artificial step at the border."""
    P = 2 * max(image.shape)
    log_image = np.log(image + LOG_OFFSET)
    transfer = homomorphic_transfer(P, D0, gamma_low, gamma_high, c)
    return np.exp(filter_image(log_image, transfer, pad_mode="symmetric")) - LOG_OFFSET


def equalise(image):
    """global histogram equalisation, the assignment 1 implementation:
    T(v) = round(255 F(v)) on the 8-bit image, F the inclusive cdf."""
    image = np.asarray(image, dtype=np.uint8)
    counts = np.bincount(image.ravel(), minlength=256).astype(np.float64)
    lut = np.round(255.0 * np.cumsum(counts) / counts.sum()).astype(np.uint8)
    return lut[image]
