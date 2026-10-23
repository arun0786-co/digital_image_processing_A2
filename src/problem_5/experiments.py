import time
from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from common.freq import pad_image
from common.io_utils import downsample, load_rgb, luminance, normalise_for_display
from common.metrics import SSIM_NOTE, evaluate, psnr
from common.paths import find_photo, input_path, results_dir
from common.plotting import FIGSIZE, save_figure, write_table
from problem_2.filters import make_filter
from problem_4.bilateral import bilateral
from problem_5.wiener import estimate_noise_sigma, oracle_wiener, wiener_denoise

REFERENCE_FACTOR = 2
SIGMAS = [5, 10, 20, 40]
NOISE_SEED = 2026
BUTTERWORTH_ORDER = 2
COARSE_RADII = np.geomspace(40, 2400, 20)
BILATERAL_SIGMA_S = [0.5, 1.0, 1.5, 2.0, 3.0]
BILATERAL_RANGE = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
TUNING_CROP = (slice(256, 768), slice(512, 1024))
MISSPEC_SIGMA = 20
MISSPEC_SCALES = [0.5, 1.0, 2.0]
EDGE_SHARE = 10
FLAT_SHARE = 50
REAL_CROP = (1024, 1536)

CROPS = [("roof edge", slice(400, 528), slice(1056, 1312)),
         ("ridge and sky", slice(192, 320), slice(640, 896)),
         ("foliage", slice(640, 768), slice(384, 640))]


@cache
def reference():
    """the instructor reference, box-averaged 2x2 to 1024 x 1536, rgb in [0, 1]."""
    return downsample(load_rgb(input_path("reference")), REFERENCE_FACTOR)


@cache
def unit_noise():
    # one standard-normal field, scaled for every sigma, so all methods and all sigmas
    # see the same realisation
    return np.random.default_rng(NOISE_SEED).standard_normal(reference().shape)


def noisy_image(sigma):
    """clean + sigma/255 * z, not clipped."""
    return reference() + sigma / 255.0 * unit_noise()


def rfft_padded(image):
    P = 2 * max(image.shape[:2])
    return np.fft.rfft2(pad_image(image, P, P, mode="symmetric"), axes=(0, 1)), P


def lowpass_from_rfft(spectrum, P, transfer, shape):
    """apply a centred P x P transfer function to an rfft2 spectrum and crop."""
    half = np.fft.ifftshift(transfer)[:, :P // 2 + 1]
    out = np.fft.irfft2(spectrum * half[..., None], s=(P, P), axes=(0, 1))
    return out[:shape[0], :shape[1]]


@cache
def tuned_lowpass(kind, sigma):
    """oracle tuning of D0 by psnr: a 20-point log grid, then 9 points between the
    best grid point's neighbours."""
    clean = reference()
    noisy = noisy_image(sigma)
    spectrum, P = rfft_padded(noisy)
    n = BUTTERWORTH_ORDER if kind == "butterworth" else None

    def score(radius):
        out = lowpass_from_rfft(spectrum, P, make_filter(kind, P, radius, n=n), clean.shape)
        return psnr(clean, np.clip(out, 0, 1), 1.0)

    coarse = [score(r) for r in COARSE_RADII]
    best = int(np.argmax(coarse))
    low = COARSE_RADII[max(best - 1, 0)]
    high = COARSE_RADII[min(best + 1, len(COARSE_RADII) - 1)]
    fine_radii = np.geomspace(low, high, 9)
    fine = [score(r) for r in fine_radii]
    radius = float(fine_radii[int(np.argmax(fine))])
    output = lowpass_from_rfft(spectrum, P, make_filter(kind, P, radius, n=n), clean.shape)
    return radius, output


@cache
def tuned_bilateral(sigma):
    """oracle tuning of (sigma_s, k) with sigma_r = k sigma on a 512 x 512 crop, then
    one run on the full image."""
    clean = reference()
    noisy = noisy_image(sigma)
    best = None
    for sigma_s in BILATERAL_SIGMA_S:
        for k in BILATERAL_RANGE:
            out = bilateral(noisy[TUNING_CROP], sigma_s, k * sigma / 255.0)
            value = psnr(clean[TUNING_CROP], np.clip(out, 0, 1), 1.0)
            if best is None or value > best[0]:
                best = (value, sigma_s, k)
    _, sigma_s, k = best
    return sigma_s, k, bilateral(noisy, sigma_s, k * sigma / 255.0)


def denoising_table(out):
    clean = reference()
    lines = [f"5.1 reference_image.jpg, box-averaged 2x2 to {clean.shape[0]}x{clean.shape[1]} rgb, scaled to [0, 1]",
             f"noise: clean + (sigma / 255) z, one z ~ N(0, 1) (seed {NOISE_SEED}) shared by every method and sigma; "
             "the noisy input is not clipped",
             f"metrics on outputs clipped to [0, 1], data range 1.0; ssim: {SSIM_NOTE}",
             "all frequency-domain filters mirror-pad to P = 3072 x 3072",
             "wiener S_nn = known injected variance; S_ff = smoothed noisy periodogram - S_nn, floored at 0",
             "oracle = a parameter or spectrum taken from the clean reference", "",
             f"{'sigma':>6s}  {'method':44s}{'psnr dB':>9s}{'ssim':>8s}"]
    for sigma in SIGMAS:
        noisy = noisy_image(sigma)
        rows = [("noisy input", noisy)]
        rows.append(("wiener, radial S_ff (not oracle)", wiener_denoise(noisy, sigma / 255.0, "radial")[0]))
        rows.append(("wiener, 15x15 smoothed S_ff (not oracle)", wiener_denoise(noisy, sigma / 255.0, "smoothed")[0]))
        rows.append(("wiener, clean periodogram (oracle)", oracle_wiener(noisy, clean, sigma / 255.0)))
        radius, gaussian = tuned_lowpass("gaussian", sigma)
        rows.append((f"gaussian LPF D0={radius:.0f} (oracle-tuned)", gaussian))
        radius, butterworth = tuned_lowpass("butterworth", sigma)
        rows.append((f"butterworth LPF n={BUTTERWORTH_ORDER} D0={radius:.0f} (oracle-tuned)", butterworth))
        sigma_s, k, smoothed = tuned_bilateral(sigma)
        rows.append((f"bilateral s_s={sigma_s:g} s_r={k:g}sigma (oracle-tuned)", smoothed))
        for name, result in rows:
            value_psnr, value_ssim = evaluate(clean, result, 1.0)
            lines.append(f"{sigma:6d}  {name:44s}{value_psnr:9.3f}{value_ssim:8.4f}")
        lines.append("")
    lines.append(f"bilateral tuned on rows {TUNING_CROP[0].start}-{TUNING_CROP[0].stop}, cols "
                 f"{TUNING_CROP[1].start}-{TUNING_CROP[1].stop} over s_s in {BILATERAL_SIGMA_S}, "
                 f"k in {BILATERAL_RANGE}; colour range distance, radius ceil(3 s_s)")
    write_table(out / "denoising_table.txt", lines)

    sigma = 20
    clean_crops, panels = [], []
    noisy = noisy_image(sigma)
    panels = [("clean", clean), (f"noisy sigma={sigma}", noisy),
              ("wiener radial", wiener_denoise(noisy, sigma / 255.0, "radial")[0]),
              ("gaussian LPF (oracle)", tuned_lowpass("gaussian", sigma)[1]),
              ("bilateral (oracle)", tuned_bilateral(sigma)[2])]
    fig, axes = plt.subplots(len(CROPS), len(panels), figsize=(3.4 * len(panels), 2.0 * len(CROPS)))
    for row, (crop_name, rows_, cols_) in enumerate(CROPS):
        for column, (name, image) in enumerate(panels):
            axes[row, column].imshow(np.clip(image[rows_, cols_], 0, 1))
            axes[row, column].set_title(f"{name}: {crop_name}", fontsize=8)
            axes[row, column].axis("off")
    save_figure(fig, out / "denoising_crops.png")


def radial_profile(transfer):
    P = transfer.shape[0]
    return transfer[P // 2, P // 2:]


def wiener_gain_plot(out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for sigma in SIGMAS:
        _, gain = wiener_denoise(noisy_image(sigma), sigma / 255.0, "radial")
        profile = radial_profile(gain[..., 1])
        frequency = np.arange(profile.size) / gain.shape[0]
        axes[0].semilogx(frequency[1:], profile[1:], label=f"sigma = {sigma}")
        if sigma == MISSPEC_SIGMA:
            axes[1].semilogx(frequency[1:], profile[1:], label="wiener, radial S_ff", color="black")
            P = gain.shape[0]
            for kind, colour in [("gaussian", "tab:green"), ("butterworth", "tab:orange")]:
                radius, _ = tuned_lowpass(kind, sigma)
                n = BUTTERWORTH_ORDER if kind == "butterworth" else None
                lp = radial_profile(make_filter(kind, P, radius, n=n))
                axes[1].semilogx(frequency[1:], lp[1:], color=colour, label=f"{kind} D0={radius:.0f} (oracle)")
    axes[0].set_title("P5.1 wiener gain W(rho) (green channel) for each noise level")
    axes[1].set_title(f"P5.1 wiener gain vs fixed-cutoff low-pass, sigma = {MISSPEC_SIGMA}")
    for ax in axes:
        ax.set_xlabel("radial frequency (cycles/pixel)")
        ax.set_ylabel("gain")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    save_figure(fig, out / "wiener_gain.png")


def band_masks(clean):
    gy, gx = np.gradient(luminance(clean))
    strength = np.hypot(gx, gy)
    edge = strength >= np.percentile(strength, 100 - EDGE_SHARE)
    flat = strength <= np.percentile(strength, FLAT_SHARE)
    return edge, flat


def band_rmse(clean, result, mask):
    return float(np.sqrt(np.mean((np.clip(result, 0, 1)[mask] - clean[mask]) ** 2)) * 255)


def misspecification(out):
    clean = reference()
    sigma = MISSPEC_SIGMA
    noisy = noisy_image(sigma)
    edge, flat = band_masks(clean)
    lines = [f"5.2 wiener at sigma = {sigma} with assumed noise power scale * S_nn (radial S_ff estimate)",
             "S_ff is re-estimated with the assumed S_nn, as a user of the filter would do",
             f"band rmse in 8-bit levels: edge = top {EDGE_SHARE} % clean gradient, flat = bottom {FLAT_SHARE} %", "",
             f"{'scale':>6s}{'psnr dB':>9s}{'ssim':>8s}{'flat rmse':>11s}{'edge rmse':>11s}{'mean gain':>11s}"]
    results = {}
    for scale in MISSPEC_SCALES:
        result, gain = wiener_denoise(noisy, sigma / 255.0, "radial", noise_scale=scale)
        results[scale] = result
        value_psnr, value_ssim = evaluate(clean, result, 1.0)
        lines.append(f"{scale:6.1f}{value_psnr:9.3f}{value_ssim:8.4f}{band_rmse(clean, result, flat):11.3f}"
                     f"{band_rmse(clean, result, edge):11.3f}{gain.mean():11.4f}")
    write_table(out / "misspecification.txt", lines)

    fig, axes = plt.subplots(2, len(MISSPEC_SCALES) + 1, figsize=(3.6 * (len(MISSPEC_SCALES) + 1), 4.6))
    for row, (crop_name, rows_, cols_) in enumerate(CROPS[:2]):
        axes[row, 0].imshow(np.clip(noisy[rows_, cols_], 0, 1))
        axes[row, 0].set_title(f"noisy: {crop_name}", fontsize=8)
        for column, scale in enumerate(MISSPEC_SCALES, start=1):
            axes[row, column].imshow(np.clip(results[scale][rows_, cols_], 0, 1))
            axes[row, column].set_title(f"assumed {scale:g} S_nn", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    save_figure(fig, out / "misspecification.png")


def edge_blur(out):
    clean = reference()
    sigma = MISSPEC_SIGMA
    noisy = noisy_image(sigma)
    edge, flat = band_masks(clean)
    methods = [("noisy input", noisy),
               ("wiener radial", wiener_denoise(noisy, sigma / 255.0, "radial")[0]),
               ("wiener oracle", oracle_wiener(noisy, clean, sigma / 255.0)),
               ("gaussian LPF (oracle)", tuned_lowpass("gaussian", sigma)[1]),
               ("bilateral (oracle)", tuned_bilateral(sigma)[2])]
    lines = [f"5.2 where the error sits, sigma = {sigma}; rmse in 8-bit levels",
             f"edge band = top {EDGE_SHARE} % of the clean luminance gradient ({int(edge.sum())} px), "
             f"flat = bottom {FLAT_SHARE} % ({int(flat.sum())} px)", "",
             f"{'method':26s}{'flat rmse':>11s}{'edge rmse':>11s}{'edge / flat':>13s}"]
    for name, result in methods:
        f_rmse, e_rmse = band_rmse(clean, result, flat), band_rmse(clean, result, edge)
        lines.append(f"{name:26s}{f_rmse:11.3f}{e_rmse:11.3f}{e_rmse / f_rmse:13.2f}")
    write_table(out / "edge_blur.txt", lines)

    column = 1180
    rows_ = slice(420, 520)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, result in [("clean", clean)] + methods[1:]:
        ax.plot(np.arange(rows_.start, rows_.stop), luminance(np.clip(result, 0, 1))[rows_, column], label=name)
    ax.set_xlabel(f"row (column {column}, across the roof edge)")
    ax.set_ylabel("luminance")
    ax.set_title(f"P5.2 profile across the roof edge, sigma = {sigma}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, out / "edge_profile.png")


def autocorrelation(residual, lag):
    a = residual[:, :-lag].ravel()
    b = residual[:, lag:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def residual_spectrum(residual):
    spectrum = np.fft.fftshift(np.fft.fft2(residual - residual.mean()))
    return np.log1p(np.abs(spectrum))


def real_noise(out):
    clean = reference()
    lines = ["5.2c noise estimate check on the synthetic images (immerkaer 1996, per channel, averaged)",
             f"{'injected sigma':>15s}{'estimated sigma':>17s}"]
    for sigma in SIGMAS:
        noisy = noisy_image(sigma)
        estimate = np.mean([estimate_noise_sigma(noisy[..., c]) for c in range(3)]) * 255
        lines.append(f"{sigma:15d}{estimate:17.3f}")
    clean_estimate = np.mean([estimate_noise_sigma(clean[..., c]) for c in range(3)]) * 255
    lines.append(f"{'clean ref':>15s}{clean_estimate:17.3f}   (texture floor of the method on this image)")

    photo = find_photo("noisy")
    if photo is None:
        lines += ["", "BLOCKED: no photo found in photos/ whose name starts with 'noisy'.",
                  "the real-noise case study needs a genuinely noisy photo taken by the student."]
        write_table(out / "real_noise.txt", lines)
        return

    rgb = load_rgb(photo)
    rows, cols = rgb.shape[:2]
    crop_rows, crop_cols = min(REAL_CROP[0], rows), min(REAL_CROP[1], cols)
    top, left = (rows - crop_rows) // 2, (cols - crop_cols) // 2
    image = rgb[top:top + crop_rows, left:left + crop_cols]
    estimates = [estimate_noise_sigma(image[..., c]) for c in range(3)]
    sigma = float(np.mean(estimates))
    filtered, _ = wiener_denoise(image, sigma, "radial")
    residual = luminance(image - filtered)

    synthetic_sigma = sigma * 255
    synthetic_noisy = clean + sigma * unit_noise()
    synthetic_residual = luminance(synthetic_noisy - wiener_denoise(synthetic_noisy, sigma, "radial")[0])

    level = luminance(filtered)
    bins = np.quantile(level, np.linspace(0, 1, 6))
    local = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (level >= low) & (level <= high)
        local.append((low, high, float(residual[mask].std()) * 255))

    lines += ["", f"photo: {photo.name}, {rows}x{cols}, centre crop {crop_rows}x{crop_cols} at full resolution",
              f"estimated sigma per channel (8-bit levels): "
              + ", ".join(f"{e * 255:.2f}" for e in estimates) + f"; used sigma = {synthetic_sigma:.2f}",
              "residual = noisy - wiener output (luminance)",
              f"residual autocorrelation lag 1 / 2: real {autocorrelation(residual, 1):.3f} / "
              f"{autocorrelation(residual, 2):.3f}, synthetic white at the same sigma "
              f"{autocorrelation(synthetic_residual, 1):.3f} / {autocorrelation(synthetic_residual, 2):.3f}",
              "residual std by output brightness quintile (8-bit levels), signal dependence check:"]
    lines += [f"  level {low:.3f}-{high:.3f}: {value:.3f}" for low, high, value in local]
    lines.append("no psnr or ssim: there is no clean, registered reference for this photo")
    write_table(out / "real_noise.txt", lines)

    crop = (slice(crop_rows // 2 - 128, crop_rows // 2 + 128), slice(crop_cols // 2 - 192, crop_cols // 2 + 192))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    axes[0, 0].imshow(np.clip(image[crop], 0, 1))
    axes[0, 0].set_title(f"real noisy photo (crop), sigma est. {synthetic_sigma:.1f}", fontsize=9)
    axes[0, 1].imshow(np.clip(filtered[crop], 0, 1))
    axes[0, 1].set_title("white-noise wiener output", fontsize=9)
    axes[0, 2].imshow(normalise_for_display(residual[crop], 0.5, 99.5), cmap="gray")
    axes[0, 2].set_title("residual (noisy - output)", fontsize=9)
    axes[1, 0].imshow(residual_spectrum(residual), cmap="magma")
    axes[1, 0].set_title("log |DFT| of the real residual", fontsize=9)
    axes[1, 1].imshow(residual_spectrum(synthetic_residual), cmap="magma")
    axes[1, 1].set_title("log |DFT| of a synthetic white-noise residual", fontsize=9)
    axes[1, 2].plot([0.5 * (a + b) for a, b, _ in local], [v for _, _, v in local], "o-")
    axes[1, 2].set_xlabel("output luminance")
    axes[1, 2].set_ylabel("residual std (8-bit levels)")
    axes[1, 2].set_title("noise level vs brightness", fontsize=9)
    axes[1, 2].grid(True, alpha=0.25)
    for ax in list(axes[0]) + list(axes[1, :2]):
        ax.axis("off")
    save_figure(fig, out / "real_noise.png")


EXPERIMENTS = {
    "denoising_table": denoising_table,
    "wiener_gain": wiener_gain_plot,
    "misspecification": misspecification,
    "edge_blur": edge_blur,
    "real_noise": real_noise,
}


def run(names=None):
    out = results_dir(5)
    for name in names or EXPERIMENTS:
        print(f"\n== problem 5: {name}")
        start = time.perf_counter()
        EXPERIMENTS[name](out)
        print(f"  done in {time.perf_counter() - start:.2f} s")
