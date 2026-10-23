import numpy as np


def integral_image(image):
    """I[i, j] = sum of image[:i, :j], with a leading row and column of zeros."""
    return np.pad(image.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def box_mean(image, size):
    """mean over a size x size window centred on each pixel (size odd), edges reflected."""
    radius = size // 2
    padded = np.pad(image, radius, mode="reflect")
    total = integral_image(padded)
    rows, cols = image.shape
    window = (total[size:size + rows, size:size + cols] - total[:rows, size:size + cols]
              - total[size:size + rows, :cols] + total[:rows, :cols])
    return window / size ** 2


def local_std(image, size):
    mean = box_mean(image, size)
    return np.sqrt(np.maximum(box_mean(image ** 2, size) - mean ** 2, 0.0))
