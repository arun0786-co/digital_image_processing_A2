import time
from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from common.io_utils import downsample, load_rgb, luminance, normalise_for_display
from common.metrics import SSIM_NOTE, evaluate, psnr
from common.paths import find_photo, results_dir
from common.plotting import FIGSIZE, save_figure, write_table
from problem_5.experiments import reference
from problem_6.motion import (apply_otf, continuous_otf, estimate_blur, frequency_grid, motion_psf,
                              naive_inverse, otf, wiener_deconvolve)

MOTION_LENGTH = 21
MOTION_ANGLE = 30.0
SNR_LEVELS = [50, 40, 30, 20]
NOISE_SEED = 2026
INVERSE_FLOOR = 1e-12
K_VALUES = np.logspace(-7, 0, 36)
ZERO_ORDERS = [1, 2, 3, 4, 5]
SHOW_SNR = 30
REAL_LONG_SIDE = 1536
REAL_K_VALUES = [3e-3, 1e-2, 3e-2]
REAL_K = 1e-2
CROP = (slice(400, 656), slice(1056, 1440))


@cache
def blur_setup():
    clean = reference()
    psf = motion_psf(MOTION_LENGTH, MOTION_ANGLE)
    transfer = otf(psf, clean.shape[:2])
    return clean, psf, transfer, apply_otf(clean, transfer)


@cache
def unit_noise():
    return np.random.default_rng(NOISE_SEED).standard_normal(reference().shape)


@cache
def observed(snr_db):
    """blurred + white noise at the stated snr, not clipped. the same standard-normal
    field is scaled for every snr."""
    clean, _, _, blurred = blur_setup()
    sigma = np.sqrt(np.mean(clean ** 2) / 10 ** (snr_db / 10))
    noise = sigma * unit_noise()
    realised = 10 * np.log10(np.sum(clean ** 2) / np.sum(noise ** 2))
    return blurred + noise, sigma, realised


def projection(shape):
    theta = np.deg2rad(MOTION_ANGLE)
    fx, fy = frequency_grid(shape)
    return fx * np.cos(theta) + fy * np.sin(theta)


def zero_line_points(shape, m, limit=0.5):
    """for each sampled fy, the sampled fx nearest the line fx cos + fy sin = m / L.
    only points with fx < 0 are returned, so a plot can mark the left half and leave
    the stripes themselves visible in the right half."""
    theta = np.deg2rad(MOTION_ANGLE)
    rows, cols = shape
    fy = -np.fft.fftfreq(rows)
    fx_exact = (m / MOTION_LENGTH - fy * np.sin(theta)) / np.cos(theta)
    fx = np.round(fx_exact * cols) / cols
    keep = (np.abs(fx) < limit) & (np.abs(fy) < limit) & (fx < 0)
    return fx[keep], fy[keep]


def psf_otf(out):
    clean, psf, transfer, _ = blur_setup()
    shape = clean.shape[:2]
    model = continuous_otf(shape, MOTION_LENGTH, MOTION_ANGLE)
    p = projection(shape)
    fx, fy = frequency_grid(shape)
    radius = np.hypot(fx, fy)
    lines = [f"6.1 motion psf, L = {MOTION_LENGTH} px, theta = {MOTION_ANGLE} deg (counter-clockwise from +x, y up)",
             f"discretisation: segment length inside each pixel / L (dense midpoint sampling), sum = {psf.sum():.12f}, "
             f"kernel {psf.shape[0]}x{psf.shape[1]}, {int(np.sum(psf > 0))} non-zero taps",
             f"boundary convention: circular, for both the forward blur and every deconvolution; image {shape[0]}x{shape[1]}",
             f"max | |H_dft| - |sinc| | over all frequencies         = {np.abs(np.abs(transfer) - np.abs(model)).max():.4f}",
             f"max | |H_dft| - |sinc| | for radius < 0.1 cyc/px     = {np.abs(np.abs(transfer) - np.abs(model))[radius < 0.1].max():.4f}",
             "",
             "zero lines fx cos(theta) + fy sin(theta) = m / L, spacing 1 / L = "
             f"{1 / MOTION_LENGTH:.5f} cycles/pixel along the motion direction",
             "samples within half a bin of the line: |p - m/L| < 0.5 / max(rows, cols), p the projection",
             f"{'m':>3s}{'m/L':>10s}{'samples':>9s}{'min |H|':>11s}{'median |H|':>12s}{'|sinc| at (m+1/2)/L':>21s}"]
    half_bin = 0.5 / max(shape)
    for m in ZERO_ORDERS:
        near = np.abs(p - m / MOTION_LENGTH) < half_bin
        lines.append(f"{m:3d}{m / MOTION_LENGTH:10.5f}{int(near.sum()):9d}{np.abs(transfer)[near].min():11.2e}"
                     f"{np.median(np.abs(transfer)[near]):12.2e}{abs(np.sinc(m + 0.5)):21.4f}")
    write_table(out / "otf_zeros.txt", lines)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(psf, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"P6.1 psf, L = {MOTION_LENGTH}, theta = {MOTION_ANGLE:g} deg", fontsize=10)
    axes[0].axis("off")
    centred = np.fft.fftshift(np.log(np.abs(transfer) + 1e-6))
    axes[1].imshow(centred, cmap="gray", extent=(-0.5, 0.5, -0.5, 0.5))
    for m in ZERO_ORDERS:
        for sign in (1, -1):
            zx, zy = zero_line_points(shape, sign * m)
            axes[1].plot(zx, zy, ".", ms=0.6, color="tab:red")
    axes[1].set_xlim(-0.3, 0.3)
    axes[1].set_ylim(-0.3, 0.3)
    axes[1].set_xlabel("fx (cycles/pixel)")
    axes[1].set_ylabel("fy (cycles/pixel)")
    axes[1].set_title("log |H|; red (left half): samples nearest the predicted zeros", fontsize=10)
    order = np.argsort(p.ravel())
    on_axis = np.abs(fx * -np.sin(np.deg2rad(MOTION_ANGLE)) + fy * np.cos(np.deg2rad(MOTION_ANGLE))).ravel()[order] < 1.5 / max(shape)
    pp = p.ravel()[order][on_axis]
    axes[2].plot(pp, np.abs(transfer).ravel()[order][on_axis], ".", ms=2, label="|H| sampled near the motion axis")
    dense = np.linspace(-0.5, 0.5, 4001)
    axes[2].plot(dense, np.abs(np.sinc(MOTION_LENGTH * dense)), lw=0.8, color="tab:red", label="|sinc(L p)|")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-4, 1.5)
    axes[2].set_xlabel("p = fx cos + fy sin (cycles/pixel)")
    axes[2].set_title("P6.1 otf along the motion direction", fontsize=10)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)
    save_figure(fig, out / "psf_otf.png")


def inverse_filter(out):
    clean, _, transfer, blurred = blur_setup()
    shape = clean.shape[:2]
    p = projection(shape)
    near_zero = np.abs(p * MOTION_LENGTH - np.round(p * MOTION_LENGTH)) < 0.05
    near_zero &= np.round(p * MOTION_LENGTH) != 0
    away = np.abs(p * MOTION_LENGTH - np.round(p * MOTION_LENGTH)) > 0.4
    clean_spectrum = np.fft.fft2(clean, axes=(0, 1))

    restored, _ = naive_inverse(blurred, transfer, INVERSE_FLOOR)
    lines = [f"6.2 naive inverse F = G / H, floor {INVERSE_FLOOR:g} on |H| (values below it take the floor with H's phase)",
             f"noise-free blurred image: psnr of G / H = {psnr(clean, np.clip(restored, 0, 1), 1.0):.2f} dB, "
             f"min |H| on the grid = {np.abs(transfer).min():.3e}",
             "error spectrum |F_hat - F| (green channel), mean near the zero lines (|pL - m| < 0.05, m != 0) "
             "vs away from them (|pL - m| > 0.4)", "",
             f"{'snr dB':>7s}{'realised':>10s}{'sigma':>9s}{'psnr G':>9s}{'psnr G/H':>10s}{'err near zeros':>16s}"
             f"{'err away':>11s}{'ratio':>9s}"]
    for snr in SNR_LEVELS:
        noisy, sigma, realised = observed(snr)
        estimate, spectrum = naive_inverse(noisy, transfer, INVERSE_FLOOR)
        error = np.abs(spectrum[..., 1] - clean_spectrum[..., 1])
        lines.append(f"{snr:7d}{realised:10.3f}{sigma:9.5f}{psnr(clean, np.clip(noisy, 0, 1), 1.0):9.2f}"
                     f"{psnr(clean, np.clip(estimate, 0, 1), 1.0):10.2f}{error[near_zero].mean():16.3e}"
                     f"{error[away].mean():11.3e}{error[near_zero].mean() / error[away].mean():9.1f}")
    write_table(out / "inverse_filter.txt", lines)

    noisy, _, _ = observed(SHOW_SNR)
    estimate, spectrum = naive_inverse(noisy, transfer, INVERSE_FLOOR)
    g_log = np.fft.fftshift(np.log(np.abs(np.fft.fft2(luminance(noisy))) + 1e-12))
    f_log = np.fft.fftshift(np.log(np.abs(luminance(spectrum)) + 1e-12))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, data, title in [(axes[0], g_log, f"log |G|, snr {SHOW_SNR} dB"),
                            (axes[1], f_log, "log |G / H| before any clipping")]:
        shown = ax.imshow(data, cmap="magma", extent=(-0.5, 0.5, -0.5, 0.5))
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.02)
        for m in ZERO_ORDERS:
            for sign in (1, -1):
                zx, zy = zero_line_points(shape, sign * m, limit=0.25)
                ax.plot(zx, zy, ".", ms=0.8, color="cyan")
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.25, 0.25)
        ax.set_xlabel("fx (cycles/pixel)")
        ax.set_ylabel("fy (cycles/pixel)")
        ax.set_title(f"P6.2 {title}; cyan (left half): samples nearest m/L", fontsize=9)
    axes[2].imshow(normalise_for_display(luminance(estimate), 1, 99), cmap="gray")
    axes[2].set_title("naive inverse output (luminance, 1-99 % stretch)", fontsize=9)
    axes[2].axis("off")
    save_figure(fig, out / "inverse_spectrum.png")


def wiener_sweep(out):
    clean, _, transfer, _ = blur_setup()
    curves = {}
    lines = [f"6.3 wiener deconvolution F = conj(H) G / (|H|^2 + K), image in [0, 1], circular model, "
             f"L = {MOTION_LENGTH}, theta = {MOTION_ANGLE:g}",
             f"K swept over {len(K_VALUES)} log-spaced values from {K_VALUES[0]:.0e} to {K_VALUES[-1]:.0e}; "
             f"psnr and ssim on outputs clipped to [0, 1], data range 1; ssim: {SSIM_NOTE}",
             "nsr = 10^(-snr/10) = noise power / mean signal power, the flat-spectrum guess for K", "",
             f"{'snr dB':>7s}{'best K':>10s}{'nsr':>10s}{'psnr dB':>9s}{'ssim':>8s}{'psnr K=1e-7':>13s}{'psnr K=1':>10s}"]
    for snr in SNR_LEVELS:
        noisy, _, _ = observed(snr)
        values = np.array([psnr(clean, np.clip(wiener_deconvolve(noisy, transfer, k), 0, 1), 1.0) for k in K_VALUES])
        curves[snr] = values
        best = int(np.argmax(values))
        score = evaluate(clean, wiener_deconvolve(noisy, transfer, K_VALUES[best]), 1.0)
        lines.append(f"{snr:7d}{K_VALUES[best]:10.2e}{10 ** (-snr / 10):10.2e}{score[0]:9.3f}{score[1]:8.4f}"
                     f"{values[0]:13.3f}{values[-1]:10.3f}")
    write_table(out / "k_sweep.txt", lines)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for snr, values in curves.items():
        best = int(np.argmax(values))
        line, = ax.semilogx(K_VALUES, values, label=f"snr {snr} dB")
        ax.plot(K_VALUES[best], values[best], "o", color=line.get_color())
        ax.axvline(10 ** (-snr / 10), color=line.get_color(), ls=":", lw=0.8)
    ax.set_xlabel("K")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"P6.3 wiener K sweep (dotted: K = 10^(-snr/10))")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    save_figure(fig, out / "k_sweep.png")

    noisy, _, _ = observed(SHOW_SNR)
    best = K_VALUES[int(np.argmax(curves[SHOW_SNR]))]
    panels = [("clean", clean), (f"blurred + noise, snr {SHOW_SNR} dB", noisy),
              ("K = 1e-6 (too small)", wiener_deconvolve(noisy, transfer, 1e-6)),
              (f"K = {best:.1e} (best psnr)", wiener_deconvolve(noisy, transfer, best)),
              ("K = 1e-1 (too large)", wiener_deconvolve(noisy, transfer, 1e-1))]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.0))
    for ax, (name, image) in zip(axes, panels):
        ax.imshow(np.clip(image[CROP], 0, 1))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    save_figure(fig, out / "deblurred_comparison.png")


def cepstrum_check(lines, image_rows):
    for name, image in image_rows:
        angle, length, _ = estimate_blur(image)
        lines.append(f"{name:28s}{angle:10.1f}{length:10.1f}")


def real_blur(out):
    lines = ["6.4 blur estimate from the cepstrum ift{log|G|}: a motion blur of length L at angle theta",
             "leaves a negative spike at distance about L along the motion direction",
             f"validation on the synthetic case (true theta = {MOTION_ANGLE:g}, L = {MOTION_LENGTH}), luminance:",
             f"{'image':28s}{'theta':>10s}{'L':>10s}"]
    cepstrum_check(lines, [(f"blurred, snr {snr} dB", luminance(observed(snr)[0])) for snr in SNR_LEVELS])

    photo = find_photo("blur")
    if photo is None:
        lines += ["", "BLOCKED: no photo found in photos/ whose name starts with 'blur'.",
                  "the case study needs a genuinely motion-blurred photo taken by the student."]
        write_table(out / "real_blur.txt", lines)
        return

    rgb = load_rgb(photo)
    factor = max(1, int(np.ceil(max(rgb.shape[:2]) / REAL_LONG_SIDE)))
    image = downsample(rgb, factor)
    grey = luminance(image)
    rows, cols = grey.shape
    window = np.outer(np.hanning(rows), np.hanning(cols))
    angle, length, ceps = estimate_blur(grey * window)
    psf = motion_psf(length, angle)
    pad = int(2 * np.ceil(length))
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="symmetric")
    transfer = otf(psf, padded.shape[:2])
    results = {k: wiener_deconvolve(padded, transfer, k)[pad:-pad, pad:-pad] for k in REAL_K_VALUES}

    lines += ["", f"photo: {photo.name}, {rgb.shape[0]}x{rgb.shape[1]}, box-downsampled x{factor} to {rows}x{cols}",
              "estimate on the hann-windowed luminance (the window suppresses the border cross in the spectrum)",
              f"estimated theta = {angle:.1f} deg, L = {length:.1f} px at this scale ({length * factor:.1f} px at full size)",
              f"stripe spacing in the spectrum 1 / L = {1 / length:.4f} cycles/pixel",
              f"deconvolution: mirror padding of {pad} px, wiener K in {REAL_K_VALUES}, chosen K = {REAL_K}",
              "no psnr or ssim: the sharp original does not exist"]
    write_table(out / "real_blur.txt", lines)

    spectrum = np.fft.fftshift(np.log(np.abs(np.fft.fft2(grey * window)) + 1e-8))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(spectrum, cmap="gray", extent=(-0.5, 0.5, -0.5, 0.5))
    theta = np.deg2rad(angle)
    for m in range(1, 6):
        for sign in (1, -1):
            centre = sign * m / length * np.array([np.cos(theta), np.sin(theta)])
            direction = np.array([-np.sin(theta), np.cos(theta)])
            ends = np.array([centre - 0.5 * direction, centre + 0.5 * direction])
            axes[0].plot(ends[:, 0], ends[:, 1], color="tab:red", lw=0.6)
    axes[0].set_xlim(-0.25, 0.25)
    axes[0].set_ylim(-0.25, 0.25)
    axes[0].set_title(f"log |G| with lines at m / L, L = {length:.1f}, theta = {angle:.1f} deg", fontsize=9)
    c_rows, c_cols = ceps.shape
    view = ceps[c_rows // 2 - 60:c_rows // 2 + 61, c_cols // 2 - 60:c_cols // 2 + 61]
    axes[1].imshow(view, cmap="gray", extent=(-60.5, 60.5, -60.5, 60.5), vmin=np.percentile(view, 1),
                   vmax=np.percentile(view, 99))
    axes[1].set_title("cepstrum near the origin (dark spikes at +-L along the motion)", fontsize=9)
    save_figure(fig, out / "real_blur_spectrum.png")

    crop = (slice(rows // 2 - 192, rows // 2 + 192), slice(cols // 2 - 256, cols // 2 + 256))
    fig, axes = plt.subplots(1, 1 + len(REAL_K_VALUES), figsize=(4.4 * (1 + len(REAL_K_VALUES)), 3.8))
    axes[0].imshow(np.clip(image[crop], 0, 1))
    axes[0].set_title("real blurred photo (crop)", fontsize=9)
    for ax, k in zip(axes[1:], REAL_K_VALUES):
        ax.imshow(np.clip(results[k][crop], 0, 1))
        ax.set_title(f"wiener K = {k:g}" + (" (chosen)" if k == REAL_K else ""), fontsize=9)
    for ax in axes:
        ax.axis("off")
    save_figure(fig, out / "real_blur_deblurred.png")


EXPERIMENTS = {
    "psf_otf": psf_otf,
    "inverse_filter": inverse_filter,
    "wiener_sweep": wiener_sweep,
    "real_blur": real_blur,
}


def run(names=None):
    out = results_dir(6)
    for name in names or EXPERIMENTS:
        print(f"\n== problem 6: {name}")
        start = time.perf_counter()
        EXPERIMENTS[name](out)
        print(f"  done in {time.perf_counter() - start:.2f} s")
