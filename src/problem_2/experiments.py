import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.special import sici  # sine integral, only to draw the predicted profile

from common.freq import centred_spectrum, distance_grid, filter_image, pad_image, padded_size
from common.io_utils import load_gray
from common.metrics import label_components, psnr
from common.paths import input_path, results_dir
from common.plotting import FIGSIZE, image_grid, panel_grid, save_figure, write_table
from problem_2.filters import (apply_filter, make_filter, matched_gaussian_d0, radial_slice,
                               slice_kernel)

# synthetic step image for 2.2
STEP_SIZE = 512
STEP_SIDE = 256
STEP_INSIDE = 200.0
STEP_OUTSIDE = 50.0
PROFILE_ROW = 255
LEFT_EDGE = 127.5
RIGHT_EDGE = 383.5
RINGING_RADII = [10, 20, 40, 80]
BUTTERWORTH_RADIUS = 40
BUTTERWORTH_ORDERS = [1, 2, 3, 5, 10, 20]

POWER_TARGETS = [90.0, 95.0, 99.0, 99.5]
GRID_ORDER = 2
EQUIVALENCE_RADII = [10, 40, 160]

NOISE_SIGMA = 20.0
NOISE_SEED = 2026
SWEEP_RADII = np.arange(5, 151)
EXTENDED_RADII = np.arange(155, 401, 5)
SWEEP_ORDER = 2

TEXT_RADII = np.arange(20, 201, 10)
TEXT_JOIN = 0.9
TEXT_MERGE = 0.1
HIGHPASS_RADIUS = 50
HIGHPASS_ORDER = 4
HFE_K1 = 0.5
HFE_K2 = 0.75

SI_PI = sici(np.pi)[0]
PREDICTED_OVERSHOOT = 100 * (SI_PI / np.pi - 0.5)

# regions of the characters pattern used for the ringing insets, (rows, cols)
INSETS = [("large square", slice(45, 185), slice(500, 660)),
          ("vertical bars", slice(380, 560), slice(30, 200))]


def step_image():
    image = np.full((STEP_SIZE, STEP_SIZE), STEP_OUTSIDE)
    start = (STEP_SIZE - STEP_SIDE) // 2
    image[start:start + STEP_SIDE, start:start + STEP_SIDE] = STEP_INSIDE
    return image


def overshoot(profile):
    return 100 * (profile.max() - STEP_INSIDE) / (STEP_INSIDE - STEP_OUTSIDE)


def first_peak(profile, edge, inside=STEP_INSIDE, minimum_excess=1e-3):
    """distance from a left-hand edge to the first local maximum that rises above the
    plateau by more than minimum_excess grey levels, as (integer-sample distance,
    parabolic sub-pixel distance). (nan, nan) when there is no overshoot."""
    start = int(np.ceil(edge))
    for x in range(start + 1, len(profile) - 1):
        if (profile[x] >= profile[x - 1] and profile[x] > profile[x + 1]
                and profile[x] > inside + minimum_excess):
            left, centre, right = profile[x - 1:x + 2]
            offset = 0.5 * (left - right) / (left - 2 * centre + right)
            return x - edge, x + offset - edge
    return np.nan, np.nan


def filter_row(row, transfer):
    """filter one row as if the image were constant along y: zero-pad it to P and
    multiply by the central slice H(u, 0). this is the 1-d model of 2.2."""
    size = transfer.shape[0]
    padded = np.zeros(size)
    padded[:row.size] = row
    spectrum = np.fft.fft(padded) * np.fft.ifftshift(radial_slice(transfer))
    return np.real(np.fft.ifft(spectrum))[:row.size]


def isolated_edge(radius, P, scale=16):
    """overshoot and peak distance of a single edge on a grid `scale` times longer,
    with the cutoff scaled so f_c = radius / P is unchanged. the other edge is
    P * scale / 2 samples away, so its ripple is negligible."""
    length = P * scale
    signal = np.full(length, STEP_OUTSIDE)
    signal[length // 4:3 * length // 4] = STEP_INSIDE
    keep = np.abs(np.fft.fftfreq(length) * length) <= radius * scale
    filtered = np.real(np.fft.ifft(np.fft.fft(signal) * keep))
    edge = length // 4 - 0.5
    window = filtered[length // 4 - 1:length // 2]
    return overshoot(window), first_peak(filtered, edge)[1]


def predicted_row(columns, radius, P):
    """1-d continuous prediction for the two edges of the square on the profile row."""
    fc = radius / P
    rise = sici(2 * np.pi * fc * (columns - LEFT_EDGE))[0]
    fall = sici(2 * np.pi * fc * (columns - RIGHT_EDGE))[0]
    return STEP_OUTSIDE + (STEP_INSIDE - STEP_OUTSIDE) * (rise - fall) / np.pi


def transfer_functions(out):
    P, D0 = 1024, 40
    distance = np.arange(0, 161)
    families = [("ideal", None, "tab:blue"), ("butterworth", 2, "tab:orange"),
                ("gaussian", None, "tab:green")]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.6))
    for kind, n, colour in families:
        label = f"{kind}" + (f" n={n}" if n else "")
        for highpass, style in [(False, "-"), (True, "--")]:
            profile = radial_slice(make_filter(kind, P, D0, n=n, highpass=highpass))[P // 2 + distance]
            axes[0].plot(distance, profile, style, color=colour,
                         label=f"{label} {'HP' if highpass else 'LP'}")
    axes[0].axvline(D0, color="gray", lw=0.8)
    axes[0].set_title(f"P2.1 all six transfer functions, same D0 = {D0}")

    matched = matched_gaussian_d0(D0)
    pairs = [("ideal", D0, None, "tab:blue", f"ideal D0={D0}"),
             ("butterworth", D0, 2, "tab:orange", f"butterworth n=2 D0={D0}"),
             ("gaussian", matched, None, "tab:green", f"gaussian D0'={matched:.2f}")]
    for kind, radius, n, colour, label in pairs:
        profile = radial_slice(make_filter(kind, P, radius, n=n))[P // 2 + distance]
        axes[1].plot(distance, profile, color=colour, label=label)
    axes[1].axhline(0.5, color="gray", lw=0.8)
    axes[1].axvline(D0, color="gray", lw=0.8)
    axes[1].set_title("P2.1a matched pairing: H = 0.5 at D = 40 for butterworth and gaussian")
    for ax in axes:
        ax.set_xlabel("D (frequency bins from the centre, P = 1024)")
        ax.set_ylabel("H(D)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
    save_figure(fig, out / "transfer_functions.png")


def cutoff_semantics(out):
    P, D0 = 1024, 40
    centre = P // 2
    rows = [("ideal", None, 1.0, "1 (D <= D0 is inclusive)"),
            ("butterworth", 1, 0.5, "1/2"),
            ("butterworth", 2, 0.5, "1/2"),
            ("butterworth", 10, 0.5, "1/2"),
            ("gaussian", None, np.exp(-0.5), "exp(-1/2)")]
    lines = [f"2.1a low-pass value at D = D0 = {D0}, P = {P}",
             f"{'filter':18s}{'analytic':>28s}{'make_filter':>14s}"]
    for kind, n, value, text in rows:
        measured = make_filter(kind, P, D0, n=n)[centre + D0, centre]
        name = kind + (f" n={n}" if n else "")
        lines.append(f"{name:18s}{text + f' = {value:.6f}':>28s}{measured:14.6f}")
    matched = matched_gaussian_d0(D0)
    lines += ["", "matched gaussian: exp(-D0^2 / (2 D0'^2)) = 1/2  ->  D0' = D0 / sqrt(2 ln 2)",
              f"sqrt(2 ln 2) = {np.sqrt(2 * np.log(2)):.6f}, D0' = {matched:.4f} for D0 = {D0}",
              f"make_filter('gaussian', {P}, {matched:.4f}) at D = {D0}: "
              f"{make_filter('gaussian', P, matched)[centre + D0, centre]:.6f}"]
    write_table(out / "cutoff_semantics.txt", lines)


def power_curve(image):
    P = padded_size(image)
    power = np.abs(centred_spectrum(pad_image(image, P, P))) ** 2
    distance = distance_grid(P).ravel()
    order = np.argsort(distance, kind="stable")
    return distance[order], np.cumsum(power.ravel()[order]) / power.sum(), power, P


def power_radii(image):
    radius, alpha, _, _ = power_curve(image)
    return [float(radius[np.searchsorted(alpha, target / 100.0)]) for target in POWER_TARGETS]


def power_retention(out):
    image = load_gray(input_path("characters"))
    radius, alpha, power, P = power_curve(image)
    lines = [f"2.1b spectral power within radius r, characters pattern zero-padded to P = {P}",
             f"dc share of total power = {100 * power[P // 2, P // 2] / power.sum():.3f} %",
             f"{'alpha target %':>15s}{'radius r':>12s}{'alpha(r) %':>13s}"]
    for target in POWER_TARGETS:
        index = np.searchsorted(alpha, target / 100.0)
        lines.append(f"{target:15.1f}{radius[index]:12.2f}{100 * alpha[index]:13.4f}")
    lines += ["", "alpha at fixed radii"]
    for r in [5, 10, 20, 40, 80, 160, 320, 460]:
        lines.append(f"  r = {r:4d}: alpha = {100 * alpha[np.searchsorted(radius, r, side='right') - 1]:.4f} %")
    write_table(out / "power_retention.txt", lines)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogx(radius[1:], 100 * alpha[1:], color="tab:blue")
    for target in POWER_TARGETS:
        r = radius[np.searchsorted(alpha, target / 100.0)]
        ax.plot(r, target, "o", color="tab:red")
        ax.annotate(f"{target}% at r={r:.1f}", (r, target), textcoords="offset points",
                    xytext=(6, -12), fontsize=8)
    ax.set_xlabel("radius r (frequency bins)")
    ax.set_ylabel("alpha(r) (%)")
    ax.set_title("P2.1b enclosed spectral power, characters test pattern")
    ax.grid(True, which="both", alpha=0.25)
    save_figure(fig, out / "power_retention.png")


def inverse_design(out):
    n = np.log(256.0) / (2 * np.log(4.0))
    D0 = 10.0 / (1.0 / 16.0) ** (1.0 / (2 * n))
    P = 256
    centre = P // 2
    transfer = make_filter("butterworth", P, D0, n=int(round(n)))
    lines = ["2.1c butterworth inverse design",
             "H(10) = 16/17  ->  (10/D0)^(2n) = 1/16",
             "H(40) = 1/17   ->  (40/D0)^(2n) = 16",
             "divide: 4^(2n) = 256 = 4^4  ->  n = 2;  (10/D0)^4 = 1/16  ->  D0 = 20",
             f"numeric solve: n = {n:.6f}, D0 = {D0:.6f}",
             f"make_filter('butterworth', {P}, {D0:g}, n=2): H(10) = {transfer[centre + 10, centre]:.6f} "
             f"(16/17 = {16 / 17:.6f}), H(40) = {transfer[centre + 40, centre]:.6f} (1/17 = {1 / 17:.6f})",
             "", "attenuation at D = 2 D0: H = 1 / (1 + 4^n) <= 0.01  ->  4^n >= 99  ->  n >= "
             f"{np.log(99) / np.log(4):.4f}  ->  n = 4",
             f"{'n':>3s}{'analytic H(2 D0)':>18s}{'make_filter H(2 D0)':>22s}{'<= 1 %':>8s}"]
    for order in range(1, 7):
        measured = make_filter("butterworth", P, 20, n=order)[centre + 40, centre]
        lines.append(f"{order:3d}{1 / (1 + 4 ** order):18.6f}{measured:22.6f}{'yes' if measured <= 0.01 else 'no':>8s}")
    write_table(out / "inverse_design.txt", lines)


def ringing_ideal(out):
    step = step_image()
    P = 2 * STEP_SIZE
    columns = np.arange(STEP_SIZE)
    lines = ["2.2a ideal low-pass on the step image (B = 200 square, side 256, on A = 50), P = 1024",
             f"profile row {PROFILE_ROW}; left edge at x = {LEFT_EDGE}; "
             f"predicted OS = 100 (Si(pi)/pi - 1/2) = {PREDICTED_OVERSHOOT:.4f} %", "",
             f"{'D0':>4s}{'OS %':>9s}{'pred %':>8s}{'d sample':>10s}{'d parabolic':>13s}"
             f"{'P/(2D0)':>9s}{'P/(2D0+1)':>11s}{'d right edge':>14s}{'max g':>9s}"]
    filtered = {}
    controls = []
    for D0 in RINGING_RADII:
        g = apply_filter(step, "ideal", D0)
        filtered[D0] = g
        profile = g[PROFILE_ROW]
        d_sample, d_fine = first_peak(profile, LEFT_EDGE)
        right_sample, _ = first_peak(profile[::-1], STEP_SIZE - 1 - RIGHT_EDGE)
        lines.append(f"{D0:4d}{overshoot(profile):9.3f}{PREDICTED_OVERSHOOT:8.2f}{d_sample:10.1f}"
                     f"{d_fine:13.2f}{P / (2 * D0):9.2f}{P / (2 * D0 + 1):11.2f}"
                     f"{right_sample:14.1f}{profile.max():9.3f}")
        row_1d = filter_row(step[PROFILE_ROW], make_filter("ideal", P, D0))
        controls.append((D0, overshoot(profile), overshoot(row_1d), first_peak(row_1d, LEFT_EDGE)[1],
                         *isolated_edge(D0, P)))

    lines += ["", "reconciliation: where the gap to 8.95 % comes from",
              "  2-d: the measured row above",
              "  1-d row: the same row filtered as if the image were constant along y "
              "(same P, same zero padding, both square edges and the padding jumps)",
              "  isolated: one edge, 16x longer grid with the same f_c = D0 / P, other edges far away",
              f"{'D0':>4s}{'OS 2-d %':>10s}{'OS 1-d row %':>14s}{'d 1-d row':>11s}"
              f"{'OS isolated %':>15s}{'d isolated':>12s}"]
    for D0, os_2d, os_row, d_row, os_iso, d_iso in controls:
        lines.append(f"{D0:4d}{os_2d:10.3f}{os_row:14.3f}{d_row:11.2f}{os_iso:15.3f}{d_iso:12.2f}")

    quadrant = (slice(128, 256), slice(128, 256))
    lines += ["", "near the top-left corner: maximum of g inside the quadrant rows/cols 128..255",
              f"{'D0':>4s}{'max g':>9s}{'OS %':>8s}{'(row, col)':>12s}{'dist from top':>15s}"
              f"{'dist from left':>16s}{'(1+OS_iso)^2-1 %':>18s}"]
    corner_rows = {}
    for D0, _, _, _, os_iso, _ in controls:
        block = filtered[D0][quadrant]
        r, c = np.unravel_index(np.argmax(block), block.shape)
        r, c = r + 128, c + 128
        corner_rows[D0] = r
        lines.append(f"{D0:4d}{block.max():9.3f}{overshoot(block.ravel()):8.3f}{str((int(r), int(c))):>12s}"
                     f"{r - LEFT_EDGE:15.1f}{c - LEFT_EDGE:16.1f}{100 * ((1 + os_iso / 100) ** 2 - 1):18.2f}")
    write_table(out / "ideal_ringing_table.txt", lines)

    fig, axes = plt.subplots(len(RINGING_RADII), 1, figsize=(8, 2.7 * len(RINGING_RADII)))
    window = slice(60, 200)
    for ax, D0 in zip(axes, RINGING_RADII):
        ax.plot(columns[window], filtered[D0][PROFILE_ROW, window], color="tab:blue",
                label=f"row {PROFILE_ROW}, measured")
        ax.plot(columns[window], predicted_row(columns[window], D0, P), "--", color="tab:red",
                label="Si prediction, two edges")
        ax.plot(columns[window], filtered[D0][corner_rows[D0], window], color="tab:green", lw=0.9,
                label=f"row {corner_rows[D0]}, through the corner maximum")
        ax.axhline(STEP_INSIDE, color="gray", lw=0.6)
        ax.axvline(LEFT_EDGE, color="gray", lw=0.6)
        ax.set_title(f"P2.2a ideal LPF D0 = {D0}: overshoot {overshoot(filtered[D0][PROFILE_ROW]):.2f} %")
        ax.set_xlabel("column x")
        ax.set_ylabel("g")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9, loc="upper left")
    save_figure(fig, out / "ideal_ringing_profiles.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    corner = filtered[20][96:224, 96:224]
    axes[0].imshow(corner, cmap="gray", vmin=30, vmax=220, extent=(95.5, 223.5, 223.5, 95.5))
    axes[0].axhline(PROFILE_ROW, color="tab:blue", lw=0.8)
    axes[0].axhline(corner_rows[20], color="tab:green", lw=0.8)
    axes[0].set_title("P2.2a ideal LPF D0 = 20, top-left corner of the square")
    excess = np.clip(filtered[20] - STEP_INSIDE, 0, None)[96:224, 96:224]
    shown = axes[1].imshow(excess, cmap="magma", extent=(95.5, 223.5, 223.5, 95.5))
    axes[1].set_title("overshoot above B = 200 (grey levels)")
    fig.colorbar(shown, ax=axes[1], fraction=0.046, pad=0.02)
    save_figure(fig, out / "ideal_ringing_corner.png")


def butterworth_orders(out):
    step = step_image()
    P = 2 * STEP_SIZE
    D0 = BUTTERWORTH_RADIUS
    cases = [(f"butterworth n={n}", "butterworth", D0, n) for n in BUTTERWORTH_ORDERS]
    cases += [("ideal", "ideal", D0, None),
              ("gaussian D0=40", "gaussian", D0, None),
              (f"gaussian D0'={matched_gaussian_d0(D0):.2f}", "gaussian", matched_gaussian_d0(D0), None)]
    lines = [f"2.2b step image, D0 = {D0}, P = {P}, profile row {PROFILE_ROW}",
             "h(x) = ift{H(u, 0)} is the 1-d kernel from the central slice of the P x P filter", "",
             f"{'filter':24s}{'OS %':>9s}{'d sample':>10s}{'min h':>13s}{'max h':>12s}{'|min h|/max h':>15s}{'sum h':>9s}"]
    kernels = {}
    profiles = {}
    for name, kind, radius, n in cases:
        g = apply_filter(step, kind, radius, n=n)
        profile = g[PROFILE_ROW]
        profiles[name] = profile
        h = slice_kernel(make_filter(kind, P, radius, n=n))
        kernels[name] = h
        d_sample, _ = first_peak(profile, LEFT_EDGE)
        lines.append(f"{name:24s}{overshoot(profile):9.4f}{d_sample:10.1f}{h.min():13.3e}"
                     f"{h.max():12.4e}{abs(min(h.min(), 0.0)) / h.max():15.4e}{h.sum():9.5f}")
    lines += ["", f"ideal-filter limit predicted for n -> infinity: {PREDICTED_OVERSHOOT:.4f} %"]
    write_table(out / "butterworth_order_table.txt", lines)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(-60, 61)
    for name in [f"butterworth n={n}" for n in BUTTERWORTH_ORDERS] + ["ideal"]:
        h = np.roll(kernels[name], 60)[:121]
        axes[0].plot(x, h, label=name, lw=1.1)
        axes[1].plot(np.arange(100, 200), profiles[name][100:200], label=name, lw=1.1)
    axes[0].axhline(0, color="gray", lw=0.6)
    axes[0].set_xlabel("x (pixels)")
    axes[0].set_ylabel("h(x)")
    axes[0].set_title(f"P2.2b 1-d kernels from H(u, 0), D0 = {D0}")
    axes[1].axhline(STEP_INSIDE, color="gray", lw=0.6)
    axes[1].set_xlabel("column x")
    axes[1].set_ylabel("g")
    axes[1].set_title("P2.2b step-edge profiles near the left edge")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    save_figure(fig, out / "butterworth_orders.png")


def filter_grid(out):
    image = load_gray(input_path("characters"))
    radii = power_radii(image)
    families = [("ideal", None, lambda r: r, "ILPF"),
                ("butterworth", GRID_ORDER, lambda r: r, f"BLPF n={GRID_ORDER}"),
                ("gaussian", None, matched_gaussian_d0, "GLPF (matched D0')")]
    # one figure per family, the four radii two to a row
    results = {}
    for index, (kind, n, radius_of, label) in enumerate(families):
        images, titles = [], []
        for target, r in zip(POWER_TARGETS, radii):
            g = apply_filter(image, kind, radius_of(r), n=n)
            results[(label, target)] = g
            images.append(g)
            titles.append(f"{label}, r={r:.1f} ({target}%)")
        fig, axes = image_grid(images, titles)
        if index == 0:
            for ax in axes:
                for _, rows, cols in INSETS:
                    ax.add_patch(Rectangle((cols.start, rows.start), cols.stop - cols.start,
                                           rows.stop - rows.start, fill=False, ec="tab:red", lw=1))
        save_figure(fig, out / f"filter_grid_{label.split()[0].lower()}.png")

    target = POWER_TARGETS[1]
    for name, rows, cols in INSETS:
        panels = [(image, "original")] + [(results[(label, target)], label)
                                           for _, _, _, label in families]
        fig, axes = panel_grid(len(panels), height=3.4)
        for ax, (data, label) in zip(axes, panels):
            ax.imshow(data[rows, cols], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(f"{name}: {label}", fontsize=10)
            ax.axis("off")
        fig.suptitle(f"P2.2c zoomed inset at r = {radii[1]:.1f} ({target}% power)", fontsize=11)
        save_figure(fig, out / f"filter_grid_insets_{name.replace(' ', '_')}.png")

    profile_row = 112
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for _, _, _, label in families:
        ax.plot(np.arange(480, 688), results[(label, target)][profile_row, 480:688], label=label)
    ax.plot(np.arange(480, 688), image[profile_row, 480:688], color="black", lw=0.8, label="original")
    ax.set_xlabel("column")
    ax.set_ylabel("grey level")
    ax.set_title(f"P2.2c row {profile_row} through the large square, r = {radii[1]:.1f}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    save_figure(fig, out / "filter_grid_profile.png")


def gaussian_equivalence(out):
    from scipy.ndimage import gaussian_filter  # comparison asked for by 2.2d
    image = load_gray(input_path("characters"))
    P = padded_size(image)
    lines = [f"2.2d frequency-domain GLPF vs scipy.ndimage.gaussian_filter, characters pattern, P = {P}",
             "sigma_s = P / (2 pi D0); scipy with mode='constant' (zero boundary, like the padding)", "",
             f"{'D0':>5s}{'sigma_s px':>12s}{'H at P/2':>11s}{'max err t=4':>13s}{'max err t=8':>13s}"
             f"{'mean err t=8':>14s}{'interior max t=8':>18s}"]
    for D0 in EQUIVALENCE_RADII:
        sigma = P / (2 * np.pi * D0)
        frequency = apply_filter(image, "gaussian", D0)
        spatial_4 = gaussian_filter(image, sigma, mode="constant", cval=0.0, truncate=4.0)
        spatial_8 = gaussian_filter(image, sigma, mode="constant", cval=0.0, truncate=8.0)
        error_8 = np.abs(frequency - spatial_8)
        margin = int(np.ceil(8 * sigma)) + 1
        interior = error_8[margin:-margin, margin:-margin]
        edge_value = np.exp(-(P / 2) ** 2 / (2 * D0 ** 2))
        lines.append(f"{D0:5d}{sigma:12.4f}{edge_value:11.2e}{np.abs(frequency - spatial_4).max():13.3e}"
                     f"{error_8.max():13.3e}{error_8.mean():14.3e}{interior.max():18.3e}")
    write_table(out / "gaussian_equivalence.txt", lines)


def psnr_sweep(out):
    clean = load_gray(input_path("cameraman"))
    noisy = clean + np.random.default_rng(NOISE_SEED).normal(0.0, NOISE_SIGMA, clean.shape)
    P = padded_size(clean)
    families = [("ideal", None, lambda r: r, "ILPF"),
                ("butterworth", SWEEP_ORDER, lambda r: r, f"BLPF n={SWEEP_ORDER}"),
                ("gaussian", None, matched_gaussian_d0, "GLPF (matched D0')")]
    # the noisy spectrum does not depend on the filter, so transform it once
    spectrum = centred_spectrum(pad_image(noisy, P, P))
    rows, cols = clean.shape

    def filtered(kind, radius, n):
        transfer = make_filter(kind, P, radius, n=n)
        return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * transfer)))[:rows, :cols]

    radii = np.concatenate([SWEEP_RADII, EXTENDED_RADII])
    curves = {}
    for kind, n, radius_of, label in families:
        curves[label] = np.array([psnr(clean, np.clip(filtered(kind, radius_of(r), n), 0, 255), 255)
                                  for r in radii])
    noisy_psnr = psnr(clean, np.clip(noisy, 0, 255), 255)
    in_range = radii <= SWEEP_RADII[-1]

    lines = [f"2.3a cameraman + gaussian noise sigma = {NOISE_SIGMA}, seed {NOISE_SEED}, not clipped before filtering",
             f"P = {P}, D0 swept {SWEEP_RADII[0]}..{SWEEP_RADII[-1]} step 1 as asked, then "
             f"{EXTENDED_RADII[0]}..{EXTENDED_RADII[-1]} step {EXTENDED_RADII[1] - EXTENDED_RADII[0]} "
             "to find the true peak",
             "D0 is the half-amplitude radius on the padded grid; gaussian uses D0' = D0 / sqrt(2 ln 2)",
             "outputs clipped to [0, 255] for psnr, data range 255",
             f"noisy input psnr = {noisy_psnr:.3f} dB", "",
             f"{'filter':22s}{'best in 5..150':>15s}{'psnr dB':>9s}{'best overall':>14s}{'psnr dB':>9s}"
             f"{'psnr D0=30':>12s}{'psnr D0=100':>13s}"]
    for label, values in curves.items():
        best_range = int(np.argmax(np.where(in_range, values, -np.inf)))
        best = int(np.argmax(values))
        lines.append(f"{label:22s}{radii[best_range]:15d}{values[best_range]:9.3f}{radii[best]:14d}"
                     f"{values[best]:9.3f}{values[radii == 30][0]:12.3f}{values[radii == 100][0]:13.3f}")
    write_table(out / "psnr_sweep.txt", lines)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for label, values in curves.items():
        best = int(np.argmax(values))
        line, = ax.plot(radii, values, label=label)
        ax.plot(radii[best], values[best], "o", color=line.get_color())
    ax.axvspan(SWEEP_RADII[0], SWEEP_RADII[-1], color="gray", alpha=0.08, label="range asked for (5..150)")
    ax.axhline(noisy_psnr, color="gray", ls="--", lw=0.8, label="noisy input")
    ax.set_xlabel("D0 (half-amplitude radius, frequency bins, P = 1024)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"P2.3a low-pass denoising of cameraman, sigma = {NOISE_SIGMA:g}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_figure(fig, out / "psnr_sweep.png")

    best_images = [noisy]
    titles = [f"noisy ({noisy_psnr:.2f} dB)"]
    for kind, n, radius_of, label in families:
        best = int(np.argmax(curves[label]))
        best_images.append(filtered(kind, radius_of(radii[best]), n))
        titles.append(f"{label}, D0={radii[best]} ({curves[label][best]:.2f} dB)")
    fig, _ = image_grid([np.clip(i, 0, 255) for i in best_images], titles)
    save_figure(fig, out / "psnr_sweep_best.png")


def dilate(mask, radius=1, outside=False):
    """binary dilation with a (2 radius + 1) square, by or-ing shifted copies."""
    size = 2 * radius + 1
    padded = np.pad(mask, radius, constant_values=outside)
    out = np.zeros_like(mask)
    for dr in range(size):
        for dc in range(size):
            out |= padded[dr:dr + mask.shape[0], dc:dc + mask.shape[1]]
    return out


def closing(mask, radius=1):
    return ~dilate(~dilate(mask, radius), radius, outside=True)


def text_regions(image):
    """text strokes, the 1-2 px gaps a 3x3 closing bridges, and background at least
    3 px from any stroke."""
    text = image < 127.5
    gaps = closing(text) & ~text
    background = ~dilate(text, 3)
    return text, gaps, background


def text_metrics(g, regions):
    text, gaps, background = regions
    darkness = (255.0 - g) / 255.0
    stroke = darkness[text].mean()
    gap = darkness[gaps].mean()
    back = darkness[background].mean()
    # threshold halfway between the filtered stroke and background levels
    mask = darkness > 0.5 * (stroke + back)
    count = label_components(mask)[1]
    return stroke, gap, gap / stroke, stroke - back, count


def text_repair(out):
    image = load_gray(input_path("text_gaps"))
    P = padded_size(image)
    regions = text_regions(image)
    lines = [f"2.3b gaussian low-pass on text_gaps ({image.shape[0]}x{image.shape[1]}), P = {P}",
             f"text pixels {int(regions[0].sum())}, gap pixels bridged by a 3x3 closing {int(regions[1].sum())}",
             "darkness = (255 - g) / 255; join = gap darkness / stroke darkness; "
             "contrast = stroke darkness - background darkness",
             "pieces = 8-connected components after thresholding halfway between stroke and background", "",
             f"{'D0':>5s}{'stroke':>8s}{'gap':>8s}{'join':>8s}{'contrast':>10s}{'pieces':>8s}"]
    stroke, gap, join, contrast, count = text_metrics(image, regions)
    lines.append(f"{'orig':>5s}{stroke:8.3f}{gap:8.3f}{join:8.3f}{contrast:10.3f}{count:8d}")
    table = {}
    for D0 in TEXT_RADII:
        table[D0] = text_metrics(apply_filter(image, "gaussian", D0), regions)
        stroke, gap, join, contrast, count = table[D0]
        lines.append(f"{D0:5d}{stroke:8.3f}{gap:8.3f}{join:8.3f}{contrast:10.3f}{count:8d}")

    # rule: gaps at least TEXT_JOIN as dark as strokes, and no more than TEXT_MERGE of the
    # pieces lost to letters merging (relative to the least-blurred setting)
    plateau = table[TEXT_RADII[-1]][4]
    chosen = max(D0 for D0, m in table.items()
                 if m[2] >= TEXT_JOIN and m[4] >= (1 - TEXT_MERGE) * plateau)
    radius = chosen * np.sqrt(2 * np.log(2))
    gaussian = apply_filter(image, "gaussian", chosen)
    ideal = apply_filter(image, "ideal", radius)
    _, _, join_i, contrast_i, count_i = text_metrics(ideal, regions)
    lines += ["", f"rule: join >= {TEXT_JOIN} and pieces >= {1 - TEXT_MERGE:.2f} * {plateau} (pieces at D0 = "
                  f"{TEXT_RADII[-1]})",
              f"chosen D0 = {chosen}; its half-amplitude radius is D0 sqrt(2 ln 2) = {radius:.2f}",
              f"ideal LPF at that radius: join {join_i:.3f}, contrast {contrast_i:.3f}, pieces {count_i}, "
              f"range [{ideal.min():.1f}, {ideal.max():.1f}] (gaussian range [{gaussian.min():.1f}, {gaussian.max():.1f}])"]
    write_table(out / "text_repair.txt", lines)

    # one row per image: the whole text, then the zoom
    crop = (slice(30, 140), slice(30, 260))
    panels = [image, image[crop], gaussian, gaussian[crop], ideal, ideal[crop]]
    titles = ["original", "original (zoom)", f"GLPF D0 = {chosen}", "GLPF (zoom)",
              f"ILPF D0 = {radius:.1f} (same H = 0.5 radius)", "ILPF (zoom)"]
    fig, _ = image_grid(panels, titles, aspect=0.9)
    save_figure(fig, out / "text_repair.png")


def ridge_cleanliness(binary):
    labels, count = label_components(binary)
    sizes = np.bincount(labels.ravel())[1:]
    return count, int(np.sum(sizes < 20))


def highpass(out):
    image = load_gray(input_path("thumb_print"))
    P = padded_size(image)
    radius = HIGHPASS_RADIUS
    families = [("IHPF", "ideal", radius, None),
                (f"BHPF n={HIGHPASS_ORDER}", "butterworth", radius, HIGHPASS_ORDER),
                ("GHPF", "gaussian", matched_gaussian_d0(radius), None)]
    lines = [f"2.3c high-pass on thumb_print ({image.shape[0]}x{image.shape[1]}), P = {P}, "
             f"common half-amplitude radius {radius}",
             f"input mean = {image.mean():.3f}", "",
             f"{'filter':12s}{'H_HP(P/2,P/2)':>15s}{'mean (crop)':>13s}{'mean (padded)':>15s}"
             f"{'std':>9s}{'components >0':>15s}{'specks <20px':>14s}"]
    outputs = {}
    for name, kind, d0, n in families:
        transfer = make_filter(kind, P, d0, n=n, highpass=True)
        padded = pad_image(image, P, P)
        full = np.real(np.fft.ifft2(np.fft.ifftshift(centred_spectrum(padded) * transfer)))
        g = full[:image.shape[0], :image.shape[1]]
        outputs[name] = g
        count, specks = ridge_cleanliness(g > 0)
        lines.append(f"{name:12s}{transfer[P // 2, P // 2]:15.3f}{g.mean():13.4f}{full.mean():15.2e}"
                     f"{g.std():9.3f}{count:15d}{specks:14d}")

    gaussian_hp = make_filter("gaussian", P, matched_gaussian_d0(radius), highpass=True)
    hfe = HFE_K1 + HFE_K2 * gaussian_hp
    emphasised = filter_image(image, hfe)
    lines += ["", f"high-frequency emphasis H = k1 + k2 H_GHPF, k1 = {HFE_K1}, k2 = {HFE_K2}",
              f"H_HFE(P/2, P/2) = {hfe[P // 2, P // 2]:.3f} -> expected mean about k1 * mean(f) = "
              f"{HFE_K1 * image.mean():.3f}",
              f"output mean = {emphasised.mean():.3f}, std = {emphasised.std():.3f} "
              f"(input std {image.std():.3f})"]
    write_table(out / "highpass.txt", lines)

    # two figures: the filter outputs, then the original next to the ridge maps
    limit = np.percentile(np.abs(outputs[f"BHPF n={HIGHPASS_ORDER}"]), 99.5)
    filtered_panels = [(outputs[name], f"{name} (mean {outputs[name].mean():.2f})", -limit, limit)
                       for name, _, _, _ in families]
    filtered_panels += [(emphasised, f"HFE k1={HFE_K1}, k2={HFE_K2}", 0, np.percentile(emphasised, 99.8))]
    ridge_panels = [(image, "original", 0, 255)]
    ridge_panels += [(outputs[name] > 0, f"{name} > 0 (ridge map)", 0, 1) for name, _, _, _ in families]
    for file_name, panels in [("highpass.png", filtered_panels), ("highpass_ridges.png", ridge_panels)]:
        fig, axes = panel_grid(len(panels))
        for ax, (data, title, vmin, vmax) in zip(axes, panels):
            ax.imshow(data, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"P2.3c {title}", fontsize=10)
            ax.axis("off")
        save_figure(fig, out / file_name)

    crop = (slice(380, 560), slice(80, 300))
    panels = [image[crop], outputs["IHPF"][crop] > 0, outputs[f"BHPF n={HIGHPASS_ORDER}"][crop] > 0,
              outputs["GHPF"][crop] > 0]
    fig, _ = image_grid(panels, ["original", "IHPF > 0", f"BHPF n={HIGHPASS_ORDER} > 0", "GHPF > 0"],
                        aspect=0.85, vmin=0, vmax=1)
    fig.axes[0].images[0].set_clim(0, 255)
    save_figure(fig, out / "ridge_maps_zoom.png")


EXPERIMENTS = {
    "transfer_functions": transfer_functions,
    "cutoff_semantics": cutoff_semantics,
    "power_retention": power_retention,
    "inverse_design": inverse_design,
    "ringing_ideal": ringing_ideal,
    "butterworth_orders": butterworth_orders,
    "filter_grid": filter_grid,
    "gaussian_equivalence": gaussian_equivalence,
    "psnr_sweep": psnr_sweep,
    "text_repair": text_repair,
    "highpass": highpass,
}


def run(names=None):
    out = results_dir(2)
    for name in names or EXPERIMENTS:
        print(f"\n== problem 2: {name}")
        start = time.perf_counter()
        EXPERIMENTS[name](out)
        print(f"  done in {time.perf_counter() - start:.2f} s")
