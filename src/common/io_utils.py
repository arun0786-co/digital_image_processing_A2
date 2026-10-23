import numpy as np
from PIL import Image, ImageOps


def load_gray(path):
    """8-bit greyscale image as float64 in [0, 255]."""
    image = ImageOps.exif_transpose(Image.open(path))
    if image.mode not in ("L", "I;16", "I", "F"):
        return luminance(np.asarray(image.convert("RGB"), dtype=np.float64))
    return np.asarray(image, dtype=np.float64)


def load_rgb(path):
    """rgb image as float64 in [0, 1]."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return np.asarray(image, dtype=np.float64) / 255.0


def luminance(rgb):
    # rec. 601 weights, the same ones used in assignment 0
    return rgb @ np.array([0.299, 0.587, 0.114])


def downsample(image, factor):
    """box average over factor x factor blocks, dropping any remainder rows or columns."""
    rows = image.shape[0] // factor * factor
    cols = image.shape[1] // factor * factor
    image = image[:rows, :cols]
    shape = (rows // factor, factor, cols // factor, factor) + image.shape[2:]
    return image.reshape(shape).mean(axis=(1, 3))


def to_uint8(image, peak=255.0):
    return np.clip(np.round(np.asarray(image) / peak * 255.0), 0, 255).astype(np.uint8)


def save_image(image, path, peak=255.0):
    Image.fromarray(to_uint8(image, peak)).save(path)


def normalise_for_display(image, low=0.0, high=100.0):
    """map the [low, high] percentile range to [0, 1] and clip."""
    lo, hi = np.percentile(image, [low, high])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float64)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)
