import numpy as np
from skimage.metrics import structural_similarity

# metrics only. ssim comes from skimage (the handout lets libraries check
# answers, and ssim is never something a problem asks us to implement).
SSIM_NOTE = ("skimage.metrics.structural_similarity, 7x7 uniform window, "
             "K1 = 0.01, K2 = 0.03, colour images averaged over channels")


def psnr(reference, estimate, data_range):
    diff = np.asarray(reference, dtype=np.float64) - np.asarray(estimate, dtype=np.float64)
    mse = np.mean(diff ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))


def ssim(reference, estimate, data_range):
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    channel_axis = -1 if reference.ndim == 3 else None
    return float(structural_similarity(reference, estimate, data_range=data_range,
                                       channel_axis=channel_axis))


def evaluate(reference, estimate, data_range):
    """clip the estimate to [0, data_range], then psnr and ssim against the reference."""
    clipped = np.clip(estimate, 0.0, data_range)
    return psnr(reference, clipped, data_range), ssim(reference, clipped, data_range)
