import time
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from common.io_utils import load_gray
from common.metrics import SSIM_NOTE, evaluate
from common.paths import input_path, results_dir
from common.plotting import FIGSIZE, panel_grid, save_figure, write_table
from problem_4.bilateral import bilateral
from problem_4.kuwahara import kuwahara, soft_kuwahara

NOISE_SIGMA = 20.0
NOISE_SEED = 2026
MATCH_WINDOWS = [5, 7, 11]
MAIN_WINDOW = 7
CALIBRATION_SIZE = 256
CALIBRATION_SEED = 11
BLOCK_WINDOWS = [11, 15]
SOFT_TEMPERATURE = NOISE_SIGMA ** 2
EDGE_STEP = 5.0
EDGE_MARGIN = 3

# (name, rows, cols) of the crops used in the comparison figures
CROPS = [("bright rectangle corner", slice(56, 120), slice(56, 120)),
         ("ramp over the disc", slice(276, 340), slice(240, 304)),
         ("disc edge (contrast 40)", slice(392, 456), slice(392, 456))]
RAMP_CROP = ("ramp interior", slice(316, 412), slice(48, 176))

# 4.3 synthetic step edges
STEP_ROWS = 256
STEP_COLS = 128
STEP_LOW = 100.0
STEP_HIGH = 150.0
NCR_VALUES = np.round(np.arange(0.05, 1.501, 0.05), 2)
NCR_SEEDS = 8
NCR_SIGMA_S = 2.0
NCR_RADIUS = 6
RANGE_FACTOR = 2.0


def noisy_q4():
    clean = load_gray(input_path("q4"))
    noisy = clean + np.random.default_rng(NOISE_SEED).normal(0.0, NOISE_SIGMA, clean.shape)
    return clean, noisy


def flat_field():
    rng = np.random.default_rng(CALIBRATION_SEED)
    return 128.0 + rng.normal(0.0, NOISE_SIGMA, (CALIBRATION_SIZE, CALIBRATION_SIZE))


def variance_ratio(output, field):
    margin = 8
    inner = (slice(margin, -margin), slice(margin, -margin))
    return float(output[inner].var() / field[inner].var())


def match_bilateral(window, field, target):
    """sigma_s = half / 2 and radius = half fix the footprint to the kuwahara window;
    sigma_r is then found by bisection so the flat-field variance ratio equals target."""
    half = window // 2
    sigma_s = half / 2.0
    low, high = 1.0, 1000.0
    for _ in range(40):
        middle = np.sqrt(low * high)
        ratio = variance_ratio(bilateral(field, sigma_s, middle, radius=half), field)
        if ratio > target:
            low = middle
        else:
            high = middle
    sigma_r = np.sqrt(low * high)
    return sigma_s, sigma_r, variance_ratio(bilateral(field, sigma_s, sigma_r, radius=half), field)


@cache
def matched_parameters():
    field = flat_field()
    rows = []
    for window in MATCH_WINDOWS:
        target = variance_ratio(kuwahara(field, window), field)
        sigma_s, sigma_r, achieved = match_bilateral(window, field, target)
        rows.append((window, target, sigma_s, sigma_r, achieved))
    return tuple(rows)


def matching(out):
    rows = matched_parameters()
    lines = ["4.1a matched smoothing strength",
             "definition: same footprint (bilateral radius = kuwahara half-window a, sigma_s = a / 2)",
             "and the same noise-variance reduction var(out) / var(in) on a flat field",
             f"flat field {CALIBRATION_SIZE}x{CALIBRATION_SIZE}, value 128, gaussian noise sigma {NOISE_SIGMA}, "
             f"seed {CALIBRATION_SEED}; sigma_r found by bisection", "",
             f"{'window':>7s}{'kuwahara var ratio':>20s}{'sigma_s':>9s}{'sigma_r':>10s}"
             f"{'bilateral var ratio':>21s}{'1/(a+1)^2':>11s}"]
    for window, target, sigma_s, sigma_r, achieved in rows:
        lines.append(f"{window:7d}{target:20.5f}{sigma_s:9.2f}{sigma_r:10.3f}{achieved:21.5f}"
                     f"{1 / (window // 2 + 1) ** 2:11.5f}")
    write_table(out / "matching.txt", lines)


def comparison(out):
    clean, noisy = noisy_q4()
    rows = matched_parameters()
    lines = [f"4.1b q4.png + gaussian noise sigma {NOISE_SIGMA}, seed {NOISE_SEED}; matched pairs from matching.txt",
             f"psnr and ssim against the clean image, outputs clipped to [0, 255], data range 255; ssim: {SSIM_NOTE}", "",
             f"{'method':40s}{'psnr dB':>9s}{'ssim':>8s}{'edge-band rmse':>16s}"]
    edges = edge_band(clean)
    psnr_noisy, ssim_noisy = evaluate(clean, noisy, 255)
    lines.append(f"{'noisy input':40s}{psnr_noisy:9.3f}{ssim_noisy:8.4f}{band_rmse(clean, noisy, edges):16.3f}")
    outputs = {}
    for window, _, sigma_s, sigma_r, _ in rows:
        kuwahara_out = kuwahara(noisy, window)
        bilateral_out = bilateral(noisy, sigma_s, sigma_r, radius=window // 2)
        for name, result in [(f"kuwahara {window}x{window}", kuwahara_out),
                             (f"bilateral s={sigma_s:.1f} r={sigma_r:.1f} ({window}x{window})", bilateral_out)]:
            score = evaluate(clean, result, 255)
            lines.append(f"{name:40s}{score[0]:9.3f}{score[1]:8.4f}{band_rmse(clean, result, edges):16.3f}")
            if window == MAIN_WINDOW:
                outputs[name] = result
    lines += ["", f"edge band: pixels within {EDGE_MARGIN} px of a ground-truth jump larger than {EDGE_STEP}"]
    write_table(out / "edge_preserving_table.txt", lines)

    # the whole images with the crops numbered, then one figure per crop, two to a row
    panels = [("clean", clean), ("noisy", noisy)] + list(outputs.items())
    fig, axes = panel_grid(len(panels))
    for ax, (name, data) in zip(axes, panels):
        ax.imshow(np.clip(data, 0, 255), cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=10)
        for number, (_, rows_, cols_) in enumerate(CROPS, start=1):
            ax.add_patch(Rectangle((cols_.start, rows_.start), 64, 64, fill=False, ec="tab:red", lw=0.8))
            ax.text(cols_.start, rows_.start - 6, str(number), color="tab:red", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"P4.1b matched {MAIN_WINDOW}x{MAIN_WINDOW} kuwahara and bilateral, noise sigma {NOISE_SIGMA:g}",
                 fontsize=11)
    save_figure(fig, out / "edge_preservation_comparison.png")

    for number, (crop_name, rows_, cols_) in enumerate(CROPS, start=1):
        fig, axes = panel_grid(len(panels), width=2.8, height=2.9)
        for ax, (name, data) in zip(axes, panels):
            ax.imshow(np.clip(data[rows_, cols_], 0, 255), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(name, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"P4.1b crop {number}: {crop_name}", fontsize=11)
        save_figure(fig, out / f"edge_preservation_crop_{number}.png")


def edge_band(clean):
    jump = np.zeros(clean.shape, dtype=bool)
    jump[:, :-1] |= np.abs(np.diff(clean, axis=1)) > EDGE_STEP
    jump[:-1, :] |= np.abs(np.diff(clean, axis=0)) > EDGE_STEP
    band = jump.copy()
    for _ in range(EDGE_MARGIN):
        grown = band.copy()
        grown[1:] |= band[:-1]
        grown[:-1] |= band[1:]
        grown[:, 1:] |= band[:, :-1]
        grown[:, :-1] |= band[:, 1:]
        band = grown
    return band


def band_rmse(clean, result, band):
    return float(np.sqrt(np.mean((np.clip(result, 0, 255)[band] - clean[band]) ** 2)))


def blockiness(result, mask):
    """total variation and the 99th percentile of the gradient magnitude, both
    measured away from the true edges. hard sub-window switching shows up as
    jumps, which raise both numbers."""
    dx = np.zeros_like(result)
    dy = np.zeros_like(result)
    dx[:, :-1] = np.diff(result, axis=1)
    dy[:-1, :] = np.diff(result, axis=0)
    gradient = np.hypot(dx, dy)[mask]
    return float(np.mean(np.abs(dx[mask]) + np.abs(dy[mask]))), float(np.percentile(gradient, 99))


def kuwahara_blockiness(out):
    clean, noisy = noisy_q4()
    smooth = ~edge_band(clean)
    smooth[:2, :] = smooth[-2:, :] = smooth[:, :2] = smooth[:, -2:] = False
    tv_clean, p99_clean = blockiness(clean, smooth)
    lines = [f"4.2b blockiness of classical vs soft kuwahara, noise sigma {NOISE_SIGMA}, seed {NOISE_SEED}",
             f"soft kuwahara: softmax weights exp(-(var_k - min var) / T), T = sigma^2 = {SOFT_TEMPERATURE:g}",
             f"measured on {int(smooth.sum())} px at least {EDGE_MARGIN + 1} px from any true edge "
             "(flat regions and the ramp)",
             f"clean image on the same pixels: tv {tv_clean:.3f}, p99 |grad| {p99_clean:.3f}", "",
             f"{'window':>7s}{'method':>11s}{'tv':>9s}{'p99 |grad|':>12s}{'psnr dB':>9s}{'ssim':>8s}{'edge rmse':>11s}"]
    edges = edge_band(clean)
    results = {}
    for window in BLOCK_WINDOWS:
        for name, result in [("classical", kuwahara(noisy, window)),
                             ("soft", soft_kuwahara(noisy, window, SOFT_TEMPERATURE))]:
            results[(window, name)] = result
            tv, p99 = blockiness(result, smooth)
            score = evaluate(clean, result, 255)
            lines.append(f"{window:7d}{name:>11s}{tv:9.3f}{p99:12.3f}{score[0]:9.3f}{score[1]:8.4f}"
                         f"{band_rmse(clean, result, edges):11.3f}")
    write_table(out / "blockiness.txt", lines)

    # one figure per crop: classical and soft side by side for each window, the noisy input last
    for key, (crop_name, rows_, cols_) in [("ramp", RAMP_CROP), ("corner", CROPS[0])]:
        low, high = (40, 130) if key == "ramp" else (0, 255)
        fig, axes = panel_grid(len(results) + 1, width=3.2, height=2.9)
        for ax, ((window, name), result) in zip(axes, results.items()):
            ax.imshow(np.clip(result[rows_, cols_], 0, 255), cmap="gray", vmin=low, vmax=high,
                      interpolation="nearest")
            ax.set_title(f"{name} {window}x{window}", fontsize=10)
        axes[-1].imshow(noisy[rows_, cols_], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[-1].set_title("noisy input (grey range 0-255)", fontsize=10)
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"P4.2b {crop_name}, outputs shown with grey range {low}-{high}", fontsize=11)
        save_figure(fig, out / f"blockiness_{key}.png")


def op_counts(window):
    """per-pixel operation counts. kuwahara is counted for the integral-image version
    used here and for a direct version that sums each sub-window."""
    half = window // 2
    sub = (half + 1) ** 2
    taps = window ** 2
    rows = [
        # integral images: 1 square, 2 cumulative sums per image (2 adds each)
        ("kuwahara, integral images", 4 + 4 * 7 + 3, 1 + 4 * 3, 0),
        ("kuwahara, direct sums", 4 * (2 * (sub - 1)) + 4 + 3, 4 * sub + 4 * 3, 0),
        ("soft kuwahara, integral images", 4 + 4 * 7 + 3 + 4 + 3 + 3, 1 + 4 * 3 + 4 + 4 + 1, 4),
        ("bilateral", 3 * taps, 4 * taps + 1, taps),
    ]
    return rows


def median_runtime(fn, *args, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def complexity(out):
    clean, noisy = noisy_q4()
    matched = {window: (sigma_s, sigma_r) for window, _, sigma_s, sigma_r, _ in matched_parameters()}
    lines = ["4.2c per-pixel operation counts (adds/subs/compares, multiplies/divides, exponentials)",
             "kuwahara integral-image cost per pixel: 2 integral images (1 square, 4 adds), 4 sub-windows x "
             "(6 adds/subs for the two box sums, 1 sub for the variance, 3 mult/div), 3 compares for the argmin",
             "soft kuwahara adds 4 subs (minus min variance), 4 divides by T, 4 exp, 3 adds (weight sum), "
             "4 mults + 3 adds (weighted mean), 1 divide",
             "bilateral per window tap: 1 sub, 1 square, 1 scale, 1 exp, 1 mult by the spatial weight, "
             "1 mult by the value, 2 adds; plus 1 divide per pixel", "",
             f"{'window':>7s}{'filter':>34s}{'add/sub/cmp':>13s}{'mult/div':>10s}{'exp':>6s}{'runtime s':>11s}"]
    for window in [5, 7, 11, 15]:
        timings = {
            "kuwahara, integral images": median_runtime(kuwahara, noisy, window),
            "soft kuwahara, integral images": median_runtime(soft_kuwahara, noisy, window, SOFT_TEMPERATURE),
        }
        if window in matched:
            sigma_s, sigma_r = matched[window]
        else:
            sigma_s, sigma_r = window // 2 / 2.0, 2 * NOISE_SIGMA
        timings["bilateral"] = median_runtime(bilateral, noisy, sigma_s, sigma_r, window // 2)
        for name, adds, mults, exps in op_counts(window):
            runtime = f"{timings[name]:11.4f}" if name in timings else f"{'-':>11s}"
            lines.append(f"{window:7d}{name:>34s}{adds:13d}{mults:10d}{exps:6d}{runtime}")
    lines += ["", f"runtimes: median of 3 runs on the 512x512 noisy q4 image"]
    write_table(out / "complexity.txt", lines)


def step_image():
    image = np.full((STEP_ROWS, STEP_COLS), STEP_LOW)
    image[:, STEP_COLS // 2:] = STEP_HIGH
    return image


def gradient_kept(profile):
    """largest jump between neighbouring columns of a row-averaged profile near the
    edge, as a fraction of the true step height."""
    centre = STEP_COLS // 2
    window = profile[centre - 8:centre + 8]
    return float(np.abs(np.diff(window)).max() / (STEP_HIGH - STEP_LOW))


def mean_field_gradient(ncr):
    """prediction from the expected range weights. a same-side neighbour differs by
    noise only, d ~ N(0, 2 sigma^2); a cross-edge neighbour by d ~ N(dI, 2 sigma^2).
    averaging exp(-d^2 / (2 sigma_r^2)) over those gives
        same side:  c = sigma_r / sqrt(sigma_r^2 + 2 sigma^2)
        cross edge: c rho,  rho = exp(-dI^2 / (2 (sigma_r^2 + 2 sigma^2))).
    the centre pixel itself always has weight 1 (d = 0), which matters once noise
    is large, because c < 1 while the self weight is not reduced."""
    step = STEP_HIGH - STEP_LOW
    sigma = ncr * step
    sigma_r = RANGE_FACTOR * sigma
    spread = sigma_r ** 2 + 2 * sigma ** 2
    same = sigma_r / np.sqrt(spread)
    rho = np.exp(-step ** 2 / (2 * spread))
    offsets = np.arange(-NCR_RADIUS, NCR_RADIUS + 1)
    spatial = np.exp(-(offsets[:, None] ** 2 + offsets[None, :] ** 2) / (2 * NCR_SIGMA_S ** 2))
    centre = STEP_COLS // 2
    profile = []
    for x in range(centre - 8, centre + 8):
        side = (x + offsets) >= centre
        values = np.where(side, STEP_HIGH, STEP_LOW)
        expected = np.where(side == (x >= centre), same, same * rho)[None, :] * np.ones_like(spatial)
        expected[NCR_RADIUS, NCR_RADIUS] = 1.0
        weight = spatial * expected
        profile.append(np.sum(weight * values[None, :]) / np.sum(weight))
    return float(np.abs(np.diff(profile)).max() / step), float(rho)


def crossing(x, y, level):
    """first x where y falls through level, by linear interpolation."""
    for i in range(len(y) - 1):
        if y[i] >= level > y[i + 1]:
            return float(x[i] + (y[i] - level) * (x[i + 1] - x[i]) / (y[i] - y[i + 1]))
    return float("nan")


def ncr_breakdown(out):
    clean = step_image()
    step = STEP_HIGH - STEP_LOW
    gaussian_level = gradient_kept(bilateral(clean, NCR_SIGMA_S, 1e9, radius=NCR_RADIUS).mean(axis=0))
    midpoint = 0.5 * (1.0 + gaussian_level)
    band = slice(STEP_COLS // 2 - 3, STEP_COLS // 2 + 3)
    measured, spread, edge_rmse, predicted, rhos = [], [], [], [], []
    for index, ncr in enumerate(NCR_VALUES):
        sigma = ncr * step
        profiles, errors = [], []
        for seed in range(NCR_SEEDS):
            rng = np.random.default_rng(1000 * index + seed)
            noisy = clean + rng.normal(0.0, sigma, clean.shape)
            result = bilateral(noisy, NCR_SIGMA_S, RANGE_FACTOR * sigma, radius=NCR_RADIUS)
            profiles.append(result.mean(axis=0))
            errors.append(np.sqrt(np.mean((result[:, band] - clean[:, band]) ** 2)))
        # averaging the profiles first keeps the max-of-differences from picking up noise
        measured.append(gradient_kept(np.mean(profiles, axis=0)))
        spread.append(np.std([gradient_kept(p) for p in profiles]))
        edge_rmse.append(np.mean(errors))
        g, rho = mean_field_gradient(ncr)
        predicted.append(g)
        rhos.append(rho)

    measured = np.array(measured)
    ncr_star = crossing(NCR_VALUES, measured, midpoint)
    # the curve levels off above the gaussian level, so also measure against that plateau
    plateau = float(measured[NCR_VALUES >= 1.0].mean())
    ncr_plateau = crossing(NCR_VALUES, measured, 0.5 * (1.0 + plateau))
    ncr_pred = crossing(NCR_VALUES, np.array(predicted), midpoint)
    ncr_half = 1.0 / np.sqrt(2 * np.log(2) * (RANGE_FACTOR ** 2 + 2))
    lines = [f"4.3 bilateral breakdown on synthetic steps {STEP_ROWS}x{STEP_COLS}, I1 = {STEP_LOW}, I2 = {STEP_HIGH}, "
             f"|dI| = {step}",
             f"bilateral sigma_s = {NCR_SIGMA_S}, radius {NCR_RADIUS}, sigma_r = {RANGE_FACTOR} sigma_noise; "
             f"{NCR_SEEDS} fresh noise images per NCR (seed = 1000 * index + k)",
             "gradient kept = largest neighbouring-column jump of the output averaged over rows and seeds, "
             "near the edge, / |dI|; kept std = spread of the single-seed values",
             f"pure spatial gaussian (sigma_r -> inf) keeps {gaussian_level:.4f}; breakdown level = midpoint "
             f"(1 + {gaussian_level:.4f}) / 2 = {midpoint:.4f}", "",
             f"{'NCR':>6s}{'kept':>11s}{'kept std':>10s}{'edge rmse':>11s}{'mean-field':>12s}{'rho':>9s}"]
    for ncr, m, s, e, p, r in zip(NCR_VALUES, measured, spread, edge_rmse, predicted, rhos):
        lines.append(f"{ncr:6.2f}{m:11.4f}{s:10.4f}{e:11.3f}{p:12.4f}{r:9.4f}")
    lines += ["", f"empirical NCR* (measured curve crosses the midpoint) = {ncr_star:.4f}",
              f"mean-field NCR* (prediction crosses the midpoint)     = {ncr_pred:.4f}",
              f"high-noise plateau (mean kept for NCR >= 1)          = {plateau:.4f}; the curve crosses "
              f"(1 + plateau) / 2 = {0.5 * (1 + plateau):.4f} at NCR = {ncr_plateau:.4f}",
              f"rho = 1/2 at NCR = 1 / sqrt(2 ln 2 (k^2 + 2)), k = sigma_r / sigma = {RANGE_FACTOR}: {ncr_half:.4f}"]
    write_table(out / "ncr_breakdown.txt", lines)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(NCR_VALUES, measured, yerr=spread, fmt="o-", ms=3, capsize=2, label=f"measured ({NCR_SEEDS} seeds)")
    ax.plot(NCR_VALUES, predicted, "--", color="tab:red", label="mean-field prediction")
    ax.axhline(midpoint, color="gray", lw=0.8, label=f"breakdown level {midpoint:.3f}")
    ax.axhline(gaussian_level, color="gray", ls=":", lw=0.8, label=f"pure gaussian blur {gaussian_level:.3f}")
    ax.axvline(ncr_star, color="tab:green", lw=1.0)
    ax.annotate(f"NCR* = {ncr_star:.3f}", (ncr_star, midpoint), xytext=(ncr_star + 0.08, midpoint + 0.12),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
    ax.set_xlabel("NCR = sigma_noise / |dI|")
    ax.set_ylabel("edge gradient kept (fraction of |dI|)")
    ax.set_title(f"P4.3 bilateral edge preservation vs noise, sigma_s = {NCR_SIGMA_S}, sigma_r = {RANGE_FACTOR} sigma")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, out / "ncr_breakdown.png")

    # one row per noise level: the noisy step next to the bilateral output
    examples = [0.1, 0.3, 0.6, 1.0]
    fig, axes = plt.subplots(len(examples), 2, figsize=(5.6, 4.6 * len(examples)))
    for row, ncr in enumerate(examples):
        sigma = ncr * step
        noisy = clean + np.random.default_rng(7).normal(0.0, sigma, clean.shape)
        result = bilateral(noisy, NCR_SIGMA_S, RANGE_FACTOR * sigma, radius=NCR_RADIUS)
        axes[row, 0].imshow(noisy, cmap="gray", vmin=0, vmax=250)
        axes[row, 0].set_title(f"noisy, NCR = {ncr}", fontsize=10)
        axes[row, 1].imshow(result, cmap="gray", vmin=0, vmax=250)
        axes[row, 1].set_title(f"bilateral output, NCR = {ncr}", fontsize=10)
    for ax in axes.ravel():
        ax.axis("off")
    save_figure(fig, out / "ncr_examples.png")


EXPERIMENTS = {
    "matching": matching,
    "comparison": comparison,
    "blockiness": kuwahara_blockiness,
    "complexity": complexity,
    "ncr_breakdown": ncr_breakdown,
}


def run(names=None):
    out = results_dir(4)
    for name in names or EXPERIMENTS:
        print(f"\n== problem 4: {name}")
        start = time.perf_counter()
        EXPERIMENTS[name](out)
        print(f"  done in {time.perf_counter() - start:.2f} s")
