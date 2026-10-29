import time
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from common.io_utils import downsample, load_rgb, luminance, normalise_for_display
from common.metrics import SSIM_NOTE, evaluate, psnr
from common.paths import find_photo, results_dir
from common.plotting import FIGSIZE, panel_grid, save_figure, write_table
from problem_5.experiments import reference
from problem_5.wiener import flat_region_sigma, noise_tiles
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
REAL_K_SWEEP = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
REAL_NOISE_GAIN = 2.0
REAL_TILE = 32
REAL_FLAT_SHARE = 0.1
REAL_ECHO_K = [1e-2, 3e-2, 1e-1]
REAL_ECHO_ROWS = (14, 94)
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

    # the two images side by side, the otf profile across the full width below
    fig = plt.figure(figsize=(9, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
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
    axes[1].set_title("log |H|, red (left half): samples nearest the zeros", fontsize=10)
    order = np.argsort(p.ravel())
    on_axis = np.abs(fx * -np.sin(np.deg2rad(MOTION_ANGLE)) + fy * np.cos(np.deg2rad(MOTION_ANGLE))).ravel()[order] < 1.5 / max(shape)
    pp = p.ravel()[order][on_axis]
    axes[2].plot(pp, np.abs(transfer).ravel()[order][on_axis], ".", ms=2, label="|H| sampled near the motion axis")
    dense = np.linspace(-0.5, 0.5, 4001)
    axes[2].plot(dense, np.abs(np.sinc(MOTION_LENGTH * dense)), lw=0.8, color="tab:red", label="|sinc(L p)|")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-4, 1.5)
    axes[2].set_xlabel("p = fx cos + fy sin (cycles/pixel)")
    axes[2].set_title("P6.1 otf along the motion direction", fontsize=11)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=9)
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
    # the two spectra side by side, the output below
    fig = plt.figure(figsize=(10, 9))
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
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
        ax.set_title(f"P6.2 {title}", fontsize=10)
    axes[2].imshow(normalise_for_display(luminance(estimate), 1, 99), cmap="gray")
    axes[2].set_title("naive inverse output (luminance, 1-99 % stretch)", fontsize=10)
    axes[2].axis("off")
    fig.suptitle("cyan, in the left half of each spectrum: the samples nearest the zero lines m/L", fontsize=10)
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
    fig, axes = panel_grid(len(panels), width=3.7, height=2.8)
    for ax, (name, image) in zip(axes, panels):
        ax.imshow(np.clip(image[CROP], 0, 1))
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    save_figure(fig, out / "deblurred_comparison.png")


def cepstrum_check(lines, image_rows):
    for name, image in image_rows:
        angle, length, _ = estimate_blur(image)
        lines.append(f"{name:28s}{angle:10.1f}{length:10.1f}")


def prepare_photo(path):
    """box-downsample so the long side is at most REAL_LONG_SIDE, returning (rgb, factor)."""
    rgb = load_rgb(path)
    factor = max(1, int(np.ceil(max(rgb.shape[:2]) / REAL_LONG_SIDE)))
    return downsample(rgb, factor), factor, rgb


def windowed_estimate(grey):
    """cepstrum estimate on the hann-windowed image; the window removes the cross that
    the image border would otherwise put in the spectrum."""
    rows, cols = grey.shape
    return estimate_blur(grey * np.outer(np.hanning(rows), np.hanning(cols)))


def cepstrum_at(ceps, angle, length):
    """cepstrum value at distance length along angle from the centre."""
    theta = np.deg2rad(angle)
    row = ceps.shape[0] // 2 - int(round(length * np.sin(theta)))
    col = ceps.shape[1] // 2 + int(round(length * np.cos(theta)))
    return float(ceps[row, col])


def deblur(image, length, angle, k):
    """wiener deconvolution after mirror padding by 2L, since a real blur is not circular."""
    pad = int(2 * np.ceil(length))
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="symmetric")
    transfer = otf(motion_psf(length, angle), padded.shape[:2])
    return wiener_deconvolve(padded, transfer, k)[pad:-pad, pad:-pad]


def noise_gain(blurred, restored):
    """how much the deconvolution raises the plane-free std in the flattest tiles of
    the blurred photo (median ratio over those tiles)."""
    _, before = noise_tiles(blurred, REAL_TILE)
    _, after = noise_tiles(restored, REAL_TILE)
    before, after = luminance(before).std(axis=(1, 2)), luminance(after).std(axis=(1, 2))
    flat = before <= np.quantile(before, REAL_FLAT_SHARE)
    return float(np.median(after[flat] / before[flat]))


def near_motion(mask, length, angle):
    """pixels within `length` of a true pixel of mask, measured along the motion direction."""
    footprint = motion_psf(2 * length, angle) > 0
    centre = footprint.shape[0] // 2
    padded = np.pad(mask, centre)
    rows, cols = mask.shape
    near = np.zeros_like(mask)
    for dr, dc in zip(*np.nonzero(footprint)):
        near |= padded[dr:dr + rows, dc:dc + cols]
    return near


def strongest_falling_edge(grey, first_row):
    """row of the largest bright-to-dark step, going down, in the profile of the central
    half of the columns, searched from first_row to the bottom."""
    cols = grey.shape[1]
    profile = grey[first_row:, cols // 4:3 * cols // 4].mean(axis=1)
    return first_row + int(np.argmin(np.diff(profile)))


def edge_step(grey, edge):
    """height of a horizontal edge: mean level 10 to 30 rows above it minus 20 to 40 below."""
    columns = slice(grey.shape[1] // 4, 3 * grey.shape[1] // 4)
    return float(grey[edge - 30:edge - 10, columns].mean() - grey[edge + 20:edge + 40, columns].mean())


def bands_below(grey, edge):
    """row profile REAL_ECHO_ROWS below a horizontal edge (central half of the columns,
    straight line removed). with a vertical motion, bands that repeat along the motion
    show up here."""
    columns = slice(grey.shape[1] // 4, 3 * grey.shape[1] // 4)
    profile = grey[edge + REAL_ECHO_ROWS[0]:edge + REAL_ECHO_ROWS[1], columns].mean(axis=1)
    t = np.arange(len(profile))
    return profile - np.polyval(np.polyfit(t, profile, 1), t)


def echo_below(grey, edge):
    """rms of the banding below a horizontal edge."""
    return float(bands_below(grey, edge).std())


def band_extremes(grey, edge):
    """rows below the edge of the darkest and the brightest band."""
    bands = bands_below(grey, edge)
    return int(np.argmin(bands)) + REAL_ECHO_ROWS[0], int(np.argmax(bands)) + REAL_ECHO_ROWS[0]


def exact_model_control(sharp, image, length, angle):
    """the sharp control photo blurred with the estimated psf plus white noise at the
    blurred photo's flat-tile level: a case where the blur model holds exactly."""
    sigma = float(flat_region_sigma(noise_tiles(image, REAL_TILE)[1], REAL_FLAT_SHARE).mean())
    pad = int(2 * np.ceil(length))
    padded = np.pad(sharp, ((pad, pad), (pad, pad), (0, 0)), mode="symmetric")
    blurred = apply_otf(padded, otf(motion_psf(length, angle), padded.shape[:2]))[pad:-pad, pad:-pad]
    return blurred + sigma * np.random.default_rng(NOISE_SEED).standard_normal(blurred.shape), sigma


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

    image, factor, full = prepare_photo(photo)
    grey = luminance(image)
    rows, cols = grey.shape
    angle, length, ceps = windowed_estimate(grey)
    clipped = float(np.mean(full.max(axis=2) >= 254 / 255))
    lines += ["", f"photo: {photo.name}, {full.shape[0]}x{full.shape[1]} after exif rotation, box-downsampled "
                  f"x{factor} to {rows}x{cols}; {clipped:.3%} of full-size pixels have a channel at 254 or 255",
              f"estimated theta = {angle:.1f} deg, L = {length:.1f} px at this scale ({length * factor:.1f} px at full size)",
              f"stripe spacing in the spectrum 1 / L = {1 / length:.4f} cycles/pixel",
              f"cepstrum at the spike: {cepstrum_at(ceps, angle, length):.4f}", "",
              "the same estimate on parts of the photo (a single PSF assumes these agree):",
              f"{'region':16s}{'theta':>8s}{'L':>8s}"]
    regions = {"top third": (slice(0, rows // 3), slice(None)),
               "middle third": (slice(rows // 3, 2 * rows // 3), slice(None)),
               "bottom third": (slice(2 * rows // 3, rows), slice(None)),
               "left half": (slice(None), slice(0, cols // 2)),
               "right half": (slice(None), slice(cols // 2, cols))}
    regional = {}
    for name, region in regions.items():
        regional[name] = windowed_estimate(grey[region])[:2]
        lines.append(f"{name:16s}{regional[name][0]:8.1f}{regional[name][1]:8.1f}")

    sharp = find_photo("reference_for_blur")
    if sharp is not None:
        sharp_rgb = prepare_photo(sharp)[0]
        sharp_grey = luminance(sharp_rgb)
        sharp_angle, sharp_length, sharp_ceps = windowed_estimate(sharp_grey)
        lines += ["", f"control: {sharp.name}, a sharp photo of the same laptop keyboard taken by holding the camera still",
                  f"  strongest negative cepstrum value at theta = {sharp_angle:.1f} deg, distance {sharp_length:.1f} "
                  f"(value {sharp_ceps.min():.4f}); at the blurred photo's spike position the value is "
                  f"{cepstrum_at(sharp_ceps, angle, length):.4f}"]

    lines += ["", f"wiener K sweep (mirror padding 2L). noise gain = median, over the {REAL_FLAT_SHARE:.0%} flattest "
                  f"{REAL_TILE}x{REAL_TILE} tiles of the blurred photo, of restored std / blurred std",
              f"{'K':>8s}{'noise gain':>12s}{'outside [0, 1]':>16s}{'min':>8s}{'max':>8s}"]
    results = {}
    for k in REAL_K_SWEEP:
        restored = deblur(image, length, angle, k)
        gain = noise_gain(image, restored)
        outside = np.mean((restored < 0) | (restored > 1))
        lines.append(f"{k:8.0e}{gain:12.2f}{outside:16.2%}{restored.min():8.2f}{restored.max():8.2f}")
        results[k] = (restored, gain)
    chosen = min(k for k in REAL_K_SWEEP if results[k][1] <= REAL_NOISE_GAIN)
    restored = results[chosen][0]
    below = restored < 0
    outside = below | (restored > 1)
    # where do the out-of-range values come from: dark areas, or clipped highlights?
    clipped_here = downsample((full.max(axis=2) >= 254 / 255).astype(float), factor) > 0
    near_clipping = near_motion(clipped_here, length, angle)
    outside_pixels = outside.any(axis=2)
    lines += [f"rule: the smallest K whose noise gain is at most {REAL_NOISE_GAIN:g}: chosen K = {chosen:g}",
              f"at K = {chosen:g}: {below.sum() / outside.sum():.1%} of the values outside [0, 1] are below 0, and "
              f"{below[image < 0.1].sum() / below.sum():.1%} of those are where the blurred photo is below 0.1; "
              f"{(outside_pixels & near_clipping).sum() / outside_pixels.sum():.1%} of the pixels with a value outside "
              "[0, 1] lie within L, along the motion, of a pixel that is clipped at full size"]

    # bands about L apart next to the strongest edge: measured on the photo and on a control
    # where the model holds exactly, scaled by the edge height so the two can be compared
    lines += ["", f"echo = rms of the row profile {REAL_ECHO_ROWS[0]} to {REAL_ECHO_ROWS[1]} rows below the strongest "
                  "bright-to-dark edge in the bottom third (central half of the columns, straight line removed)",
              f"{'image':30s}{'edge row':>9s}{'step':>8s}{'blurred':>9s}"
              + "".join(f"{'K = ' + format(k, 'g'):>11s}" for k in REAL_ECHO_K)
              + f"{'echo/step':>11s}{'dark band':>11s}{'bright band':>13s}   (last three at K = {chosen:g}; "
                "band positions in rows below the edge)"]
    edge = strongest_falling_edge(grey, 2 * rows // 3)
    cases = [("real photo", grey, edge, {k: luminance(results[k][0]) for k in REAL_ECHO_K})]
    if sharp is not None:
        control, sigma = exact_model_control(sharp_rgb, image, length, angle)
        control_grey = luminance(control)
        control_edge = strongest_falling_edge(control_grey, 2 * rows // 3)
        control_out = {k: deblur(control, length, angle, k) for k in REAL_ECHO_K}
        cases.append(("control (exact blur)", control_grey, control_edge,
                      {k: luminance(v) for k, v in control_out.items()}))
    for name, before, at, restored_by_k in cases:
        step = edge_step(before, at)
        dark, bright = band_extremes(restored_by_k[chosen], at)
        lines.append(f"{name:30s}{at:9d}{step:8.3f}{echo_below(before, at):9.4f}"
                     + "".join(f"{echo_below(restored_by_k[k], at):11.4f}" for k in REAL_ECHO_K)
                     + f"{echo_below(restored_by_k[chosen], at) / step:11.3f}{dark:11d}{bright:13d}")
    if sharp is not None:
        lines.append(f"control noise: white, sigma = {sigma * 255:.2f} (8-bit), the blurred photo's flat-tile level")

    lines += ["", f"each third restored with its own estimated psf instead of the global one (K = {chosen:g}):"]
    local = {}
    for name in ("top third", "middle third", "bottom third"):
        local_angle, local_length = regional[name]
        local[name] = deblur(image, local_length, local_angle, chosen)
    lines += [f"echo below row {edge}: global psf {echo_below(luminance(results[chosen][0]), edge):.4f}, "
              f"bottom-third psf {echo_below(luminance(local['bottom third']), edge):.4f}",
              "no psnr or ssim: the sharp original does not exist"]
    write_table(out / "real_blur.txt", lines)

    spectrum = np.fft.fftshift(np.log(np.abs(np.fft.fft2(grey * np.outer(np.hanning(rows), np.hanning(cols)))) + 1e-8))
    cepstra = [("blurred photo", ceps)] + ([("sharp control photo", sharp_ceps)] if sharp is not None else [])
    # the spectrum centred on top, the cepstra side by side below
    fig, axes = panel_grid(1 + len(cepstra), width=4.6, height=4.5, rows=[1, len(cepstra)])
    axes[0].imshow(spectrum, cmap="gray", extent=(-0.5, 0.5, -0.5, 0.5))
    theta = np.deg2rad(angle)
    for m in range(1, 6):
        for sign in (1, -1):
            centre = sign * m / length * np.array([np.cos(theta), np.sin(theta)])
            direction = np.array([-np.sin(theta), np.cos(theta)])
            ends = np.array([centre - 0.5 * direction, centre + 0.5 * direction])
            axes[0].plot(ends[:, 0], ends[:, 1], color="tab:red", lw=1.0)
    axes[0].set_xlim(-0.25, 0.25)
    axes[0].set_ylim(-0.25, 0.25)
    axes[0].set_xlabel("fx (cycles/pixel)")
    axes[0].set_ylabel("fy (cycles/pixel)")
    axes[0].set_title(f"log |G| with lines at m / L, L = {length:.1f}, theta = {angle:.1f} deg", fontsize=10)
    for ax, (name, values) in zip(axes[1:], cepstra):
        c_rows, c_cols = values.shape
        view = values[c_rows // 2 - 60:c_rows // 2 + 61, c_cols // 2 - 60:c_cols // 2 + 61]
        ax.imshow(view, cmap="gray", extent=(-60.5, 60.5, -60.5, 60.5), vmin=np.percentile(view, 1),
                  vmax=np.percentile(view, 99))
        ax.plot([length * np.cos(theta), -length * np.cos(theta)], [length * np.sin(theta), -length * np.sin(theta)],
                "o", mfc="none", color="tab:red", ms=14)
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
        ax.set_title(f"cepstrum, {name} (circles at +-L)", fontsize=10)
    save_figure(fig, out / "real_blur_spectrum.png")

    index = REAL_K_SWEEP.index(chosen)
    shown = REAL_K_SWEEP[max(index - 1, 0):index + 2]
    crop = (slice(rows // 2 - 192, rows // 2 + 192), slice(cols // 2 - 256, cols // 2 + 256))
    fig, axes = panel_grid(1 + len(shown), width=3.7, height=3.0)
    axes[0].imshow(np.clip(image[crop], 0, 1))
    axes[0].set_title("real blurred photo (centre crop)", fontsize=10)
    for ax, k in zip(axes[1:], shown):
        ax.imshow(np.clip(results[k][0][crop], 0, 1))
        ax.set_title(f"wiener K = {k:g}, noise gain {results[k][1]:.2f}" + (" (chosen)" if k == chosen else ""),
                     fontsize=10)
    for ax in axes:
        ax.axis("off")
    save_figure(fig, out / "real_blur_deblurred.png")

    third = rows // 3
    band = slice(max(cols // 2 - 175, 0), min(cols // 2 + 175, cols))
    parts = {name: (slice(third * i + third // 2 - 100, third * i + third // 2 + 100), band)
             for i, name in enumerate(["top third", "middle third", "bottom third"])}

    # the whole photo with the three parts boxed, next to the sharp control photo
    photos = [(image, "blurred photo, the three parts boxed")]
    photos += [(sharp_rgb, "sharp control photo")] if sharp is not None else []
    fig, axes = panel_grid(len(photos), width=3.3, height=7.0)
    for ax, (data, title) in zip(axes, photos):
        ax.imshow(np.clip(data, 0, 1))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for name, (rows_, cols_) in parts.items():
        axes[0].add_patch(Rectangle((cols_.start, rows_.start), cols_.stop - cols_.start, rows_.stop - rows_.start,
                                    fill=False, ec="tab:red", lw=1.5))
        axes[0].text(cols_.start, rows_.start - 10, f"{name}: theta {regional[name][0]:.1f}, L {regional[name][1]:.1f}",
                     color="tab:red", fontsize=10, bbox=dict(facecolor="white", alpha=0.85, lw=0, pad=1.5))
    save_figure(fig, out / "real_blur_photo.png")

    # one row per part: the global psf next to the part's own psf
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 7.6))
    for row, (name, part) in enumerate(parts.items()):
        local_angle, local_length = regional[name]
        axes[row, 0].imshow(np.clip(restored[part], 0, 1))
        axes[row, 0].set_title(f"{name}, one PSF (theta {angle:.1f}, L {length:.1f})", fontsize=10)
        axes[row, 1].imshow(np.clip(local[name][part], 0, 1))
        axes[row, 1].set_title(f"{name}, own PSF (theta {local_angle:.1f}, L {local_length:.1f})", fontsize=10)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"P6.4 each part restored with K = {chosen:g}", fontsize=11)
    save_figure(fig, out / "real_blur_regions.png")

    if sharp is not None:
        fig, axes = panel_grid(4, width=3.7, height=2.8)
        panels = [(image, edge, "real photo, blurred"), (restored, edge, f"real photo, restored (K = {chosen:g})"),
                  (control, control_edge, "control: sharp photo + exact blur + noise"),
                  (control_out[chosen], control_edge, f"control, restored (K = {chosen:g})")]
        for ax, (data, at, title) in zip(axes, panels):
            ax.imshow(np.clip(data[at - 100:at + 100, cols // 4:3 * cols // 4], 0, 1))
            ax.axhline(100, color="tab:red", lw=0.6)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        save_figure(fig, out / "real_blur_echo.png")


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
