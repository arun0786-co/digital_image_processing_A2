import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from common.freq import centred_spectrum, filter_image, pad_image
from common.io_utils import downsample, load_rgb, luminance, normalise_for_display
from common.local_stats import box_mean, local_std
from common.paths import input_path, results_dir
from common.plotting import FIGSIZE, image_grid, panel_grid, save_figure, write_table
from problem_3.homomorphic import equalise, homomorphic, homomorphic_transfer

TUNNEL_FACTOR = 4
GAMMA_LOWS = [0.25, 0.5, 0.75]
GAMMA_HIGHS = [1.25, 1.5, 2.0]
SWEEP_D0 = [10, 40, 160]
SHARPNESS_C = 1.0
CONTRAST_WINDOW = 15
SHADOW_PERCENTILE = 25
BRIGHT_PERCENTILE = 99
SATURATION_LEVEL = 0.98
DISPLAY_PERCENTILES = (0.1, 99.9)
REVERSAL_SCALE = 15
REVERSAL_KEEP = 75

# a setting counts as halo-free when under 2 % of strong coarse gradients reverse;
# no setting in the sweep reaches 1 % without leaving the light saturated
HALO_LIMIT = 0.02
BLOWOUT_LIMIT = 0.10

SYNTHETIC_SIZE = 512
SYNTHETIC_SEED = 3


def tunnel_luminance():
    rgb = load_rgb(input_path("tunnel"))
    return downsample(luminance(rgb), TUNNEL_FACTOR)


def to_display(image):
    """map the 0.1-99.9 percentile range to [0, 1]: the common output scaling for every method."""
    return normalise_for_display(image, *DISPLAY_PERCENTILES)


def regions(original):
    """shadow = darkest 25 % of the 31x31 smoothed luminance, bright = top 1 %."""
    smooth = box_mean(original, 31)
    shadow = smooth <= np.percentile(smooth, SHADOW_PERCENTILE)
    bright = smooth >= np.percentile(smooth, BRIGHT_PERCENTILE)
    return shadow, bright


def gradient_reversal(output, original):
    """share of strong coarse-scale input gradients whose direction the output reverses.
    both images are box-smoothed first, so this sees halos (bands of the wrong sign
    beside large edges) and not pixel noise. a monotone tone curve gives about 0."""
    smooth_in = box_mean(original, REVERSAL_SCALE)
    smooth_out = box_mean(output, REVERSAL_SCALE)
    in_y, in_x = np.gradient(smooth_in)
    out_y, out_x = np.gradient(smooth_out)
    strength = np.hypot(in_x, in_y)
    strong = strength >= np.percentile(strength, REVERSAL_KEEP)
    return float(np.mean((in_x * out_x + in_y * out_y)[strong] < 0))


def contrast_metrics(display, original, shadow, bright):
    contrast = local_std(display, CONTRAST_WINDOW)
    levels = len(np.unique(np.round(display[shadow] * 255)))
    return {"shadow contrast": float(contrast[shadow].mean()),
            "bright contrast": float(contrast[bright].mean()),
            "bright saturated": float(np.mean(display[bright] >= SATURATION_LEVEL)),
            "blown out": float(np.mean(display >= SATURATION_LEVEL)),
            "shadow levels": levels,
            "shadow mean": float(display[shadow].mean()),
            "reversal": gradient_reversal(display, original)}


METRIC_HEADER = (f"{'shadow contr':>13s}{'bright contr':>13s}{'bright sat':>11s}{'blown out':>10s}"
                 f"{'shadow lvls':>12s}{'shadow mean':>12s}{'reversal':>10s}")


def metric_text(m):
    return (f"{m['shadow contrast']:13.4f}{m['bright contrast']:13.4f}{m['bright saturated']:11.3f}"
            f"{m['blown out']:10.4f}{m['shadow levels']:12d}{m['shadow mean']:12.4f}{m['reversal']:10.4f}")


def synthetic_scene():
    size = SYNTHETIC_SIZE
    rows, cols = np.indices((size, size))
    illumination = 0.03 + 0.97 * np.exp(-((rows - size / 2) ** 2 + (cols - size / 2) ** 2) / (2 * 90.0 ** 2))
    blocks = np.random.default_rng(SYNTHETIC_SEED).random((size // 4, size // 4)) > 0.5
    reflectance = 0.4 + 0.5 * np.kron(blocks, np.ones((4, 4)))
    return illumination, reflectance


def log_magnitude_filter(image, transfer):
    """the wrong order: filter log|F(I)| with H and rebuild with the original phase,
    i.e. |G| = |F|^H. this is the 'log-magnitude spectrum' alternative in 3.1."""
    rows, cols = image.shape
    P = transfer.shape[0]
    spectrum = centred_spectrum(pad_image(image, P, P, mode="symmetric"))
    magnitude = np.exp(transfer * np.log(np.abs(spectrum) + 1e-12))
    rebuilt = magnitude * np.exp(1j * np.angle(spectrum))
    return np.real(np.fft.ifft2(np.fft.ifftshift(rebuilt)))[:rows, :cols]


def log_domain_check(out):
    illumination, reflectance = synthetic_scene()
    image = illumination * reflectance
    D0, gamma_low, gamma_high = 20, 0.5, 1.5
    P = 2 * SYNTHETIC_SIZE
    transfer = homomorphic_transfer(P, D0, gamma_low, gamma_high, SHARPNESS_C)

    outputs = {
        "input I = L R": image,
        "filter I directly": filter_image(image, transfer, pad_mode="symmetric"),
        "filter log|F(I)|": log_magnitude_filter(image, transfer),
        "filter log I (homomorphic)": homomorphic(image, D0, gamma_low, gamma_high, SHARPNESS_C),
    }
    dark = illumination < 0.06
    lit = illumination > 0.9
    lines = ["3.1 synthetic check: I = L R with a gaussian light (L from 0.03 to 1) and a block texture R in {0.4, 0.9}",
             f"H: D0 = {D0}, gamma_L = {gamma_low}, gamma_H = {gamma_high}, c = {SHARPNESS_C}, P = {P}, mirrored padding",
             "detail = mean 9x9 local std; lit region L > 0.9, dark region L < 0.06",
             f"L_lit / L_dark (mean) = {illumination[lit].mean() / illumination[dark].mean():.2f}", "",
             f"{'method':30s}{'detail lit':>12s}{'detail dark':>13s}{'lit / dark':>12s}"]
    ratios = {}
    for name, result in outputs.items():
        detail = local_std(result, 9)
        ratio = detail[lit].mean() / detail[dark].mean()
        ratios[name] = ratio
        lines.append(f"{name:30s}{detail[lit].mean():12.4f}{detail[dark].mean():13.4f}{ratio:12.2f}")
    lines += ["", "homomorphic prediction: output ~ L^gL R^gH, so the ratio becomes "
                  f"(input ratio)^gL = {ratios['input I = L R'] ** gamma_low:.2f}"]
    write_table(out / "log_domain_check.txt", lines)

    fig, _ = image_grid([to_display(v) for v in outputs.values()], list(outputs), cell=3.4, aspect=1.25,
                        vmin=0, vmax=1)
    save_figure(fig, out / "log_domain_check.png")


def sweep_settings():
    return [(D0, gl, gh) for D0 in SWEEP_D0 for gl in GAMMA_LOWS for gh in GAMMA_HIGHS]


def chosen_settings(table):
    """good: the most shadow contrast among halo-free settings that do not blow out the
    light. halo example: the setting with the most gradient reversal."""
    clean = [s for s, m in table.items() if m["reversal"] <= HALO_LIMIT and m["bright saturated"] <= BLOWOUT_LIMIT]
    good = max(clean, key=lambda s: table[s]["shadow contrast"])
    halo = max(table, key=lambda s: table[s]["reversal"])
    return good, halo


def sweep_table(original, shadow, bright):
    return {s: contrast_metrics(to_display(homomorphic(original, *s, SHARPNESS_C)), original, shadow, bright)
            for s in sweep_settings()}


def sweep(out):
    original = tunnel_luminance()
    shadow, bright = regions(original)
    base = contrast_metrics(to_display(original), original, shadow, bright)
    table = {}
    lines = [f"3.2 homomorphic sweep on tunnel_image.jpg luminance, downsampled x{TUNNEL_FACTOR} "
             f"to {original.shape[0]}x{original.shape[1]}, P = {2 * max(original.shape)}, c = {SHARPNESS_C}",
             f"outputs mapped to [0, 1] by their {DISPLAY_PERCENTILES[0]}-{DISPLAY_PERCENTILES[1]} percentile range",
             f"shadow = darkest {SHADOW_PERCENTILE} % of the 31x31 smoothed input ({int(shadow.sum())} px), "
             f"bright = top {100 - BRIGHT_PERCENTILE} % ({int(bright.sum())} px)",
             f"contrast = mean {CONTRAST_WINDOW}x{CONTRAST_WINDOW} local std; saturated = share >= {SATURATION_LEVEL}; "
             "shadow lvls = distinct 8-bit output levels in the shadow",
             f"reversal = share of the strongest {100 - REVERSAL_KEEP} % of {REVERSAL_SCALE}x{REVERSAL_SCALE}-smoothed "
             "input gradients that point the other way in the output (halo measure)", "",
             f"{'D0':>4s}{'gL':>6s}{'gH':>6s}" + METRIC_HEADER,
             f"{'input':>16s}" + metric_text(base)]
    displays = {}
    for D0 in SWEEP_D0:
        images, titles = [], []
        for gamma_low in GAMMA_LOWS:
            for gamma_high in GAMMA_HIGHS:
                display = to_display(homomorphic(original, D0, gamma_low, gamma_high, SHARPNESS_C))
                metrics = contrast_metrics(display, original, shadow, bright)
                table[(D0, gamma_low, gamma_high)] = metrics
                displays[(D0, gamma_low, gamma_high)] = display
                lines.append(f"{D0:4d}{gamma_low:6.2f}{gamma_high:6.2f}" + metric_text(metrics))
                images.append(display)
                titles.append(f"gL={gamma_low}, gH={gamma_high}, reversal {100 * metrics['reversal']:.1f}%")
        # the full 3 x 3 grid for each D0, kept as an overview
        fig, _ = image_grid(images, titles, cols=len(GAMMA_HIGHS), cell=3.3, vmin=0, vmax=1)
        fig.suptitle(f"P3.2 homomorphic sweep, D0 = {D0} (rows gL, columns gH)", fontsize=11)
        save_figure(fig, out / f"sweep_grid_d0_{D0}.png")
    good, halo = chosen_settings(table)

    # two to a row for the report: the strongest gamma pair at each D0 next to the good
    # setting, and the four gamma corners at the good setting's D0
    strongest = (GAMMA_LOWS[0], GAMMA_HIGHS[-1])
    picks = {"sweep_d0.png": [(D0,) + strongest for D0 in SWEEP_D0] + [good],
             "sweep_gamma.png": [(good[0], gl, gh) for gl in (GAMMA_LOWS[-1], GAMMA_LOWS[0])
                                 for gh in (GAMMA_HIGHS[0], GAMMA_HIGHS[-1])]}
    for name, settings in picks.items():
        titles = [f"D0 {D0}, gL {gl}, gH {gh}: reversal {100 * table[(D0, gl, gh)]['reversal']:.1f}%"
                  + (" (good)" if (D0, gl, gh) == good else "") for D0, gl, gh in settings]
        fig, _ = image_grid([displays[s] for s in settings], titles, cell=3.4, aspect=1.25, vmin=0, vmax=1)
        save_figure(fig, out / name)
    lines += ["", f"rule for a good setting: reversal <= {HALO_LIMIT}, bright saturated <= {BLOWOUT_LIMIT}, "
                  "then the largest shadow contrast",
              f"good setting  (D0, gL, gH) = {good}",
              f"strongest halo (D0, gL, gH) = {halo}"]
    write_table(out / "sweep_metrics.txt", lines)


def comparison(out):
    original = tunnel_luminance()
    shadow, bright = regions(original)
    good, halo = chosen_settings(sweep_table(original, shadow, bright))
    results = {"input": to_display(original),
               "global histogram equalisation": equalise(np.round(original * 255)) / 255.0,
               f"homomorphic {good} (good)": to_display(homomorphic(original, *good, SHARPNESS_C)),
               f"homomorphic {halo} (halo)": to_display(homomorphic(original, *halo, SHARPNESS_C))}

    lines = ["3.2 homomorphic vs global histogram equalisation (assignment 1 equalise), settings (D0, gL, gH)",
             f"shadow {int(shadow.sum())} px, bright {int(bright.sum())} px; metrics as in sweep_metrics.txt", "",
             f"{'method':44s}" + METRIC_HEADER]
    for name, display in results.items():
        lines.append(f"{name:44s}" + metric_text(contrast_metrics(display, original, shadow, bright)))
    rgb = load_rgb(input_path("tunnel"))
    bright_full = np.kron(bright, np.ones((TUNNEL_FACTOR, TUNNEL_FACTOR), dtype=bool))
    bright_full = np.pad(bright_full, [(0, rgb.shape[i] - bright_full.shape[i]) for i in range(2)])
    clipped = np.mean((rgb[bright_full] >= 1.0).any(axis=1))
    lines += ["", f"source check: {100 * clipped:.1f} % of the full-resolution bright-region pixels have a "
                  "channel at 255, so the light is clipped in the jpeg itself"]
    write_table(out / "local_contrast.txt", lines)

    rows, cols = np.nonzero(shadow)
    centre_row = int(np.percentile(rows, 70))
    centre_col = int(np.percentile(cols, 30))
    crop = (slice(centre_row - 90, centre_row + 90), slice(centre_col - 90, centre_col + 90))
    # two figures, two to a row: the whole outputs (crop boxed in green), then the crops
    fig, axes = panel_grid(len(results), width=3.4, height=4.3)
    for ax, (name, display) in zip(axes, results.items()):
        ax.imshow(display, cmap="gray", vmin=0, vmax=1)
        ax.contour(shadow, levels=[0.5], colors="tab:blue", linewidths=0.6)
        ax.contour(bright, levels=[0.5], colors="tab:red", linewidths=0.6)
        ax.axvline(original.shape[1] // 2, color="tab:orange", lw=0.6)
        ax.add_patch(Rectangle((crop[1].start, crop[0].start), 180, 180, fill=False, ec="tab:green", lw=0.9))
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    save_figure(fig, out / "homomorphic_comparison.png")

    fig, axes = panel_grid(len(results), width=3.4, height=3.6)
    for ax, (name, display) in zip(axes, results.items()):
        ax.imshow(display[crop], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle("P3.2 shadow crop (180x180, green box in the comparison figure)", fontsize=11)
    save_figure(fig, out / "homomorphic_crops.png")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    column = original.shape[1] // 2
    for name, display in results.items():
        ax.plot(display[:, column], label=name, lw=1.0)
    ax.set_xlabel("row (centre column, orange line in the comparison figure)")
    ax.set_ylabel("output level")
    ax.set_title("P3.2 vertical profile through the light and the lower tunnel rim")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, out / "profiles.png")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, display in results.items():
        ax.hist(display.ravel(), bins=128, range=(0, 1), histtype="step", label=name, density=True)
    ax.set_yscale("log")
    ax.set_xlabel("output level")
    ax.set_ylabel("density")
    ax.set_title("P3.2 output histograms")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    save_figure(fig, out / "histograms.png")


EXPERIMENTS = {
    "log_domain_check": log_domain_check,
    "sweep": sweep,
    "comparison": comparison,
}


def run(names=None):
    out = results_dir(3)
    for name in names or EXPERIMENTS:
        print(f"\n== problem 3: {name}")
        start = time.perf_counter()
        EXPERIMENTS[name](out)
        print(f"  done in {time.perf_counter() - start:.2f} s")
