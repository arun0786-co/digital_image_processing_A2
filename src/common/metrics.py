import numpy as np

# metrics, written with numpy like everything else. the unit tests check ssim against
# skimage.metrics.structural_similarity and the component count against scipy.ndimage.label.
SSIM_WINDOW = 7
SSIM_NOTE = ("own numpy ssim (wang et al. 2004), 7x7 uniform window, sample covariance, "
             "K1 = 0.01, K2 = 0.03, colour images averaged over channels")


def psnr(reference, estimate, data_range):
    diff = np.asarray(reference, dtype=np.float64) - np.asarray(estimate, dtype=np.float64)
    mse = np.mean(diff ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))


def window_mean(image, size):
    """mean over every size x size window that lies fully inside the image."""
    rows, cols = image.shape
    column_sums = sum(image[i:rows - size + 1 + i] for i in range(size))
    return sum(column_sums[:, j:cols - size + 1 + j] for j in range(size)) / size ** 2


def ssim(reference, estimate, data_range):
    """mean of the ssim map
    (2 mu_x mu_y + C1)(2 s_xy + C2) / ((mu_x^2 + mu_y^2 + C1)(s_x^2 + s_y^2 + C2))
    over all 7 x 7 windows inside the image, C1 = (0.01 L)^2, C2 = (0.03 L)^2."""
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(estimate, dtype=np.float64)
    if x.ndim == 3:
        return float(np.mean([ssim(x[..., c], y[..., c], data_range) for c in range(x.shape[2])]))
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    n = SSIM_WINDOW ** 2
    unbiased = n / (n - 1)   # sample (not population) variances, as in the paper's code
    mu_x, mu_y = window_mean(x, SSIM_WINDOW), window_mean(y, SSIM_WINDOW)
    var_x = unbiased * (window_mean(x * x, SSIM_WINDOW) - mu_x ** 2)
    var_y = unbiased * (window_mean(y * y, SSIM_WINDOW) - mu_y ** 2)
    cov = unbiased * (window_mean(x * y, SSIM_WINDOW) - mu_x * mu_y)
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(np.mean(numerator / denominator))


def evaluate(reference, estimate, data_range):
    """clip the estimate to [0, data_range], then psnr and ssim against the reference."""
    clipped = np.clip(estimate, 0.0, data_range)
    return psnr(reference, clipped, data_range), ssim(reference, clipped, data_range)


def label_components(mask):
    """8-connected pieces of a boolean mask, returned as (labels, count) with labels
    1..count on the pieces and 0 elsewhere. union-find: every pixel points at a parent,
    starting with itself. for every pair of neighbouring mask pixels the larger of the
    two roots is pointed at the smaller one, then every pointer is followed to the end
    (path compression). when no pair has two different roots left, each piece has
    exactly one root."""
    mask = np.asarray(mask, dtype=bool)
    rows, cols = mask.shape
    index = np.arange(rows * cols).reshape(rows, cols)
    firsts, seconds = [], []
    # right, down and the two diagonals cover every 8-neighbour pair once
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        left, right = max(0, -dc), cols - max(0, dc)
        both = mask[:rows - dr, left:right] & mask[dr:, left + dc:right + dc]
        firsts.append(index[:rows - dr, left:right][both])
        seconds.append(index[dr:, left + dc:right + dc][both])
    first, second = np.concatenate(firsts), np.concatenate(seconds)
    parent = np.arange(rows * cols)
    while True:
        root_a, root_b = parent[first], parent[second]
        low, high = np.minimum(root_a, root_b), np.maximum(root_a, root_b)
        joining = low < high
        if not joining.any():
            break
        np.minimum.at(parent, high[joining], low[joining])
        while True:
            grand = parent[parent]
            if np.array_equal(grand, parent):
                break
            parent = grand
    roots, numbered = np.unique(parent.reshape(rows, cols)[mask], return_inverse=True)
    labels = np.zeros((rows, cols), dtype=int)
    labels[mask] = numbered + 1
    return labels, len(roots)
