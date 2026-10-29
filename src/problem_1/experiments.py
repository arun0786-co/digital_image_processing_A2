import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np

from common.io_utils import load_gray, normalise_for_display
from common.paths import input_path, results_dir
from common.plotting import image_grid, save_figure, write_table
from problem_1.dft import (centred_kernel_grid, circular_convolve, circular_convolve_fft,
                           dft2, idft2, linear_convolve_fft, recover_sinusoids,
                           sinusoid_image)

# 1.1c ground truth: (u, v) in cycles per image, amplitude in grey levels, phase in rad
SINUSOIDS = [
    (6, 10, 50.0, np.pi / 4),
    (20, -14, 30.0, -np.pi / 3),
    (0, 45, 20.0, 2.0),
    (90, 5, 10.0, -2.5),
]
SINUSOID_SIZE = 256
SINUSOID_OFFSET = 128.0

SHIFTS = [(5, 0), (0, 17), (31, -12), (200, 300)]
SAMPLE_FREQUENCIES = [(1, 0), (0, 1), (3, 7), (50, -20), (-100, 140)]
NOISE_SEED = 2026
KERNEL_SEED = 7

LAPLACIAN_5 = np.array([[0, 0, 1, 0, 0],
                        [0, 1, 2, 1, 0],
                        [1, 2, -16, 2, 1],
                        [0, 1, 2, 1, 0],
                        [0, 0, 1, 0, 0]], dtype=float)


def median_time(fn, *args, repeats=5):
    fn(*args)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def wrap_phase(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def test_image():
    return load_gray(input_path("q1"))


def dft_validation(out):
    image = test_image()
    rng = np.random.default_rng(0)
    cases = [("q1.png 512x512", image),
             ("q1 crop 256x384", image[:256, :384]),
             ("normal noise 64x48", rng.normal(size=(64, 48)))]

    lines = ["1.1 custom separable dft F = W_M f W_N^T against numpy.fft.fft2",
             "complex128, unnormalised forward transform, times are medians",
             "",
             f"{'input':20s}{'max|F_custom-F_np|':>20s}{'max|F_np|':>12s}{'relative':>11s}"
             f"{'max|idft2(F)-f|':>17s}{'t_custom s':>12s}{'t_fft2 s':>11s}"]
    for name, image_case in cases:
        custom = dft2(image_case)
        reference = np.fft.fft2(image_case)
        error = np.abs(custom - reference).max()
        scale = np.abs(reference).max()
        round_trip = np.abs(np.real(idft2(custom)) - image_case).max()
        t_custom = median_time(dft2, image_case, repeats=3)
        t_numpy = median_time(np.fft.fft2, image_case, repeats=7)
        lines.append(f"{name:20s}{error:20.3e}{scale:12.3e}{error / scale:11.2e}"
                     f"{round_trip:17.3e}{t_custom:12.4f}{t_numpy:11.5f}")
    write_table(out / "dft_error_table.txt", lines)


def conjugate_symmetry(out):
    image = test_image()
    spectrum = dft2(image)
    rows, cols = spectrum.shape
    # F(-u, -v) with indices taken mod N
    mirrored = np.roll(spectrum[::-1, ::-1], 1, axis=(0, 1))
    error = np.abs(spectrum - np.conj(mirrored)).max()
    self_conjugate = [(0, 0), (0, cols // 2), (rows // 2, 0), (rows // 2, cols // 2)]
    max_imag = max(abs(spectrum[p].imag) for p in self_conjugate)

    lines = ["1.1a conjugate symmetry F(u, v) = conj F(-u, -v) for the real test image",
             f"max |F(u,v) - conj F(-u,-v)|            = {error:.3e}",
             f"max |F(u,v)|                            = {np.abs(spectrum).max():.3e}",
             f"max |Im F| at the 4 self-conjugate bins = {max_imag:.3e}",
             f"complex values in the full spectrum     = {rows * cols}",
             f"independent values (rfft2 half plane)   = {rows * (cols // 2 + 1)}",
             f"real degrees of freedom: 2*{rows * cols // 2 - 2} pair values + 4 real = "
             f"{2 * (rows * cols // 2 - 2) + 4} = M N"]
    write_table(out / "conjugate_symmetry.txt", lines)

    centred = np.fft.fftshift(np.log1p(np.abs(spectrum)))
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.imshow(centred, cmap="gray", extent=(-cols / 2, cols / 2, rows / 2, -rows / 2))
    ax.axhspan(0, rows / 2, color="tab:red", alpha=0.18)
    ax.set_xlabel("v (cycles/image)")
    ax.set_ylabel("u (cycles/image)")
    ax.set_title("P1.1a log(1+|F|), shaded half is the conjugate mirror of the other")
    save_figure(fig, out / "conjugate_symmetry.png")


def shift_theorem(out):
    image = test_image()
    spectrum = dft2(image)
    rows, cols = image.shape
    u = np.arange(rows)[:, None]
    v = np.arange(cols)[None, :]

    lines = ["1.1b shift theorem: f(x - x0, y - y0) <-> F(u, v) exp(-j 2 pi (u x0 + v y0) / N)",
             "circular shift with np.roll on q1.png, custom dft2", "",
             f"{'(x0, y0)':12s}{'max|F_s - F*ramp|':>20s}{'relative':>11s}{'max||F_s|-|F||':>17s}"]
    phase_rows = []
    for x0, y0 in SHIFTS:
        shifted = np.roll(image, (x0, y0), axis=(0, 1))
        measured = dft2(shifted)
        ramp = np.exp(-2j * np.pi * (u * x0 / rows + v * y0 / cols))
        predicted = spectrum * ramp
        error = np.abs(measured - predicted).max()
        magnitude_error = np.abs(np.abs(measured) - np.abs(spectrum)).max()
        lines.append(f"{str((x0, y0)):12s}{error:20.3e}{error / np.abs(spectrum).max():11.2e}"
                     f"{magnitude_error:17.3e}")
        for fu, fv in SAMPLE_FREQUENCIES:
            iu, iv = fu % rows, fv % cols
            measured_phase = np.angle(measured[iu, iv] / spectrum[iu, iv])
            theory_phase = wrap_phase(-2 * np.pi * (fu * x0 / rows + fv * y0 / cols))
            phase_rows.append(f"{str((x0, y0)):12s}{str((fu, fv)):12s}{theory_phase:12.6f}"
                              f"{measured_phase:12.6f}{abs(wrap_phase(measured_phase - theory_phase)):12.2e}")

    lines += ["", "phase change angle(F_s / F) at sample frequencies (rad)",
              f"{'(x0, y0)':12s}{'(u, v)':12s}{'predicted':>12s}{'measured':>12s}{'|diff|':>12s}"]
    lines += phase_rows
    write_table(out / "shift_theorem_table.txt", lines)


def convolution_theorem(out):
    image = test_image()
    rows, cols = image.shape
    rng = np.random.default_rng(KERNEL_SEED)
    kernels = [("box 3x3", np.ones((3, 3)) / 9),
               ("laplacian 5x5", LAPLACIAN_5),
               ("random 7x7", rng.normal(size=(7, 7)))]
    spectrum = dft2(image)

    lines = ["1.1b circular convolution theorem: f (*) h <-> F H",
             "spatial side: direct circular sum over kernel taps (np.roll)",
             "frequency side: real(idft2(dft2(f) * dft2(h))), custom transforms", "",
             f"{'kernel':16s}{'max|g_sp - g_fr|':>18s}{'max|g_sp|':>12s}{'relative':>11s}"
             f"{'max|Im g_fr|':>14s}{'max|G_sp - FH|':>16s}"]
    sample_rows = []
    for name, kernel in kernels:
        grid = centred_kernel_grid(kernel, rows, cols)
        spatial = circular_convolve(image, kernel)
        product = spectrum * dft2(grid)
        frequency = idft2(product)
        error = np.abs(spatial - np.real(frequency)).max()
        scale = np.abs(spatial).max()
        spectrum_error = np.abs(dft2(spatial) - product).max()
        lines.append(f"{name:16s}{error:18.3e}{scale:12.3e}{error / scale:11.2e}"
                     f"{np.abs(frequency.imag).max():14.3e}{spectrum_error:16.3e}")
        if name.startswith("random"):
            measured_spectrum = dft2(spatial)
            for fu, fv in [(0, 0), (5, 3), (100, -40), (-200, 17)]:
                iu, iv = fu % rows, fv % cols
                sample_rows.append(f"{str((fu, fv)):12s}{product[iu, iv]:>30.6e}"
                                   f"{measured_spectrum[iu, iv]:>30.6e}")

    lines += ["", "random 7x7 kernel, spectrum samples",
              f"{'(u, v)':12s}{'predicted F H':>30s}{'measured dft2(f (*) h)':>30s}"]
    lines += sample_rows
    write_table(out / "convolution_theorem_table.txt", lines)


def sinusoid_recovery(out):
    image = sinusoid_image(SINUSOID_SIZE, SINUSOIDS, offset=SINUSOID_OFFSET)
    spectrum = dft2(image)
    found, offset = recover_sinusoids(spectrum)
    found_map = {(u, v): (a, p) for u, v, a, p in found}

    truth_lines = [f"1.1c synthetic image ground truth, {SINUSOID_SIZE}x{SINUSOID_SIZE}",
                   "f(x, y) = offset + sum A cos(2 pi (u x + v y) / N + phi), x row, y column",
                   f"offset = {SINUSOID_OFFSET}",
                   f"{'u':>5s}{'v':>5s}{'A':>8s}{'phi (rad)':>12s}"]
    truth_lines += [f"{u:5d}{v:5d}{a:8.2f}{p:12.6f}" for u, v, a, p in SINUSOIDS]
    write_table(out / "sinusoid_ground_truth.txt", truth_lines)

    lines = ["1.1c parameters read from spectral peaks of the custom dft",
             f"peaks found above 1e-3 of the largest non-dc peak: {len(found)}",
             f"recovered offset = {offset:.10f} (truth {SINUSOID_OFFSET})", "",
             f"{'(u, v)':12s}{'A true':>8s}{'A found':>14s}{'|dA|':>10s}"
             f"{'phi true':>11s}{'phi found':>12s}{'|dphi|':>10s}"]
    for u, v, amplitude, phase in SINUSOIDS:
        got_a, got_p = found_map.get((u, v), (np.nan, np.nan))
        lines.append(f"{str((u, v)):12s}{amplitude:8.2f}{got_a:14.10f}{abs(got_a - amplitude):10.1e}"
                     f"{phase:11.6f}{got_p:12.6f}{abs(wrap_phase(got_p - phase)):10.1e}")
    lines += ["", f"grid resolution: delta f = 1/N = {1 / SINUSOID_SIZE:.6f} cycles/pixel "
                  "= 1 cycle/image"]
    lines += resolution_lines()
    write_table(out / "sinusoid_recovery.txt", lines)

    centred = np.fft.fftshift(np.log1p(np.abs(spectrum)))
    half = SINUSOID_SIZE / 2
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.2))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("P1.1c synthetic image (4 cosines + offset)")
    axes[0].axis("off")
    axes[1].imshow(centred, cmap="gray", extent=(-half, half, half, -half))
    for u, v, _, _ in found:
        for su, sv in [(u, v), (-u, -v)]:
            axes[1].plot(sv, su, "o", mfc="none", mec="tab:red", ms=10)
    axes[1].set_xlabel("v (cycles/image)")
    axes[1].set_ylabel("u (cycles/image)")
    axes[1].set_title("log(1+|F|), recovered peaks circled")
    save_figure(fig, out / "sinusoid_recovery.png")
    resolution_figure(out)


def row_signal(frequencies, amplitude=30.0, size=SINUSOID_SIZE):
    x = np.arange(size)
    return sum(amplitude * np.cos(2 * np.pi * f * x / size) for f in frequencies)


def resolution_lines():
    size = SINUSOID_SIZE
    single = np.fft.fft(row_signal([10.5], amplitude=50.0))
    estimate = 2 * np.abs(single[10:12]) / size
    predicted = 50.0 * abs(np.sin(np.pi * 0.5) / (size * np.sin(np.pi * 0.5 / size)))
    lines = ["", "off-grid frequency u = 10.5 cycles/image, A = 50, 1-d slice",
             f"amplitude read from bins 10 and 11: {estimate[0]:.4f}, {estimate[1]:.4f}",
             f"dirichlet prediction for a half-bin offset: {predicted:.4f}",
             f"bins with amplitude above 1 grey level: {int(np.sum(2 * np.abs(single[:size // 2]) / size > 1))}"]
    for pair in [(40.25, 40.75), (40.25, 41.75)]:
        dense = np.abs(np.fft.fft(row_signal(pair), 16 * size))[:16 * size // 2]
        freqs = np.arange(dense.size) / 16
        window = (freqs > 37) & (freqs < 45)
        local = dense[window]
        peaks = np.sum((local[1:-1] > local[:-2]) & (local[1:-1] > local[2:]) &
                       (local[1:-1] > 0.5 * local.max()))
        lines.append(f"pair {pair}: separation {pair[1] - pair[0]:.2f} cycles/image, "
                     f"main peaks in the 16x zero-padded spectrum = {int(peaks)}")
    return lines


def resolution_figure(out):
    size = SINUSOID_SIZE
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.6))
    for ax, pair in zip(axes, [(40.25, 40.75), (40.25, 41.75)]):
        signal = row_signal(pair)
        coarse = np.abs(np.fft.fft(signal))
        dense = np.abs(np.fft.fft(signal, 16 * size))
        dense_f = np.arange(16 * size) / 16
        ax.plot(dense_f, dense, color="tab:blue", lw=1.2, label="16x zero-padded (dtft)")
        ax.stem(np.arange(size), coarse, linefmt="k-", markerfmt="ko", basefmt=" ",
                label="N-point dft")
        for f in pair:
            ax.axvline(f, color="tab:red", ls="--", lw=0.8)
        ax.set_xlim(34, 48)
        ax.set_xlabel("u (cycles/image)")
        ax.set_ylabel("|F(u)|")
        ax.set_title(f"P1.1c two cosines {pair[1] - pair[0]:.2f} cycles/image apart")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
    save_figure(fig, out / "resolution_limit.png")


def gradient_magnitude(image):
    gx = np.roll(image, -1, axis=0) - np.roll(image, 1, axis=0)
    gy = np.roll(image, -1, axis=1) - np.roll(image, 1, axis=1)
    return np.hypot(gx, gy)


def correlation(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def axis_share(centred, exclude):
    """fraction of the energy outside `exclude` that lies on the central row or column."""
    rows, cols = centred.shape
    axes = np.zeros_like(exclude)
    axes[rows // 2, :] = True
    axes[:, cols // 2] = True
    outside = ~exclude
    return float(np.sum(centred[axes & outside] ** 2) / np.sum(centred[outside] ** 2))


def phase_magnitude(out):
    image = test_image()
    other = load_gray(input_path("cameraman"))
    spectrum = dft2(image)
    phase = np.exp(1j * np.angle(spectrum))

    noise = np.random.default_rng(NOISE_SEED).uniform(0.0, 1.0, size=image.shape)
    noise_magnitude = np.abs(dft2(noise))

    phase_only = idft2(phase)
    noise_magnitude_recon = idft2(noise_magnitude * phase)
    swapped = idft2(np.abs(dft2(other)) * phase)
    swapped_back = idft2(np.abs(spectrum) * np.exp(1j * np.angle(dft2(other))))
    zero_phase = idft2(np.abs(spectrum))

    # least-squares split of the noise-magnitude result into c p + residual, dc removed
    p = np.real(phase_only) - np.real(phase_only).mean()
    r = np.real(noise_magnitude_recon) - np.real(noise_magnitude_recon).mean()
    c = float(np.sum(r * p) / np.sum(p * p))
    residual = r - c * p
    edges = gradient_magnitude(image)

    eps = noise_magnitude.ravel()[1:] / noise_magnitude.ravel()[1:].mean() - 1
    zero_phase_real = np.real(zero_phase)
    flipped = np.roll(zero_phase_real[::-1, ::-1], 1, axis=(0, 1))
    peak = np.unravel_index(np.argmax(zero_phase_real), zero_phase_real.shape)
    # the dc term adds the constant mean(f) everywhere, so measure the shape without it
    centred_zero = np.fft.fftshift(zero_phase_real - zero_phase_real.mean())
    yy, xx = np.indices(image.shape) - np.array(image.shape)[:, None, None] // 2
    inner = np.hypot(yy, xx) <= 8

    lines = ["1.2 phase / magnitude reconstructions of q1.png (custom dft2 / idft2)",
             f"uniform noise U[0,1], seed {NOISE_SEED}", "",
             "noise magnitude statistics (dc excluded)",
             f"  var(eps) with |N| = mu (1 + eps)         = {eps.var():.4f}   (rayleigh: 4/pi - 1 = {4 / np.pi - 1:.4f})",
             "",
             "noise-magnitude reconstruction r split as r = c p + residual (dc removed)",
             f"  energy(residual) / energy(c p)          = {np.sum(residual ** 2) / np.sum((c * p) ** 2):.4f}   (predicted 0.273)",
             f"  corr(r, p)                              = {correlation(r, p):.4f}   (predicted 0.89)",
             "",
             "phase-only image p (flat magnitude, 1.2a)",
             f"  max |Im p| / max |Re p|                 = {np.abs(phase_only.imag).max() / np.abs(phase_only.real).max():.2e}",
             f"  corr(p, f)                              = {correlation(p, image):.4f}",
             f"  corr(|p|, |grad f|)                     = {correlation(np.abs(p), edges):.4f}",
             f"  corr(|r|, |grad f|)                     = {correlation(np.abs(r), edges):.4f}",
             f"  corr(swap: |F_cameraman| with q1 phase, f) = {correlation(np.real(swapped), image):.4f}",
             f"  corr(swap, cameraman)                   = {correlation(np.real(swapped), other):.4f}",
             f"  corr(|F_q1| with cameraman phase, cameraman) = {correlation(np.real(swapped_back), other):.4f}",
             "",
             "zero-phase image r0 = idft2(|F|) (1.2b)",
             f"  max |Im r0| / max |Re r0|               = {np.abs(zero_phase.imag).max() / np.abs(zero_phase.real).max():.2e}",
             f"  max |r0(x,y) - r0(-x,-y)|               = {np.abs(zero_phase_real - flipped).max():.3e}",
             f"  location of the maximum                 = {tuple(int(i) for i in peak)}",
             f"  r0(0,0) and sum|F|/N^2                  = {zero_phase_real[0, 0]:.4f}, {np.abs(spectrum).sum() / image.size:.4f}",
             f"  mean of r0 and mean of f                = {zero_phase_real.mean():.4f}, {image.mean():.4f}",
             f"  share of sum (r0 - mean)^2 within radius 8 of origin = {np.sum(centred_zero[inner] ** 2) / np.sum(centred_zero ** 2):.4f}",
             f"  share of energy outside radius 8 on the x = 0 or y = 0 lines = "
             f"{axis_share(centred_zero, inner):.4f} (those lines are {1022 / image.size:.4f} of the pixels)"]
    write_table(out / "phase_magnitude.txt", lines)

    # two to a row: the input and (a), the prediction case and (b), then the swapped pair
    display = [image / 255,
               normalise_for_display(np.real(phase_only), 0.5, 99.5),
               normalise_for_display(np.real(noise_magnitude_recon), 0.5, 99.5),
               normalise_for_display(np.log1p(np.abs(np.fft.fftshift(zero_phase_real))), 0.0, 100.0),
               normalise_for_display(np.real(swapped), 0.5, 99.5),
               normalise_for_display(np.real(swapped_back), 0.5, 99.5)]
    titles = ["q1.png", "(a) q1 phase, flat magnitude",
              "prediction case: q1 phase, noise magnitude",
              "(b) q1 magnitude, zero phase (fftshift, log)",
              "q1 phase, cameraman magnitude", "cameraman phase, q1 magnitude"]
    fig, _ = image_grid(display, titles, cell=3.8, vmin=0, vmax=1)
    save_figure(fig, out / "phase_magnitude.png")

    crop = (slice(60, 260), slice(140, 340))
    zoom = [image[crop] / 255,
            normalise_for_display(np.real(phase_only)[crop], 0.5, 99.5),
            normalise_for_display(np.real(noise_magnitude_recon)[crop], 0.5, 99.5)]
    fig, _ = image_grid(zoom, ["q1 crop", "phase only crop", "noise magnitude crop"],
                        cell=3.8, vmin=0, vmax=1)
    save_figure(fig, out / "phase_magnitude_zoom.png")


def wraparound(out):
    image = test_image()
    size = image.shape[0]
    kernel = LAPLACIAN_5
    from scipy.signal import convolve2d  # checking only
    reference = convolve2d(image, kernel, mode="same")

    circular = circular_convolve_fft(image, kernel)
    linear = linear_convolve_fft(image, kernel)
    difference = np.abs(circular - linear)
    tolerance = 1e-9 * np.abs(linear).max()

    lines = ["1.3a circular (unpadded) vs zero-padded linear convolution, 5x5 laplacian on q1.png",
             f"max |linear_fft - scipy convolve2d same| = {np.abs(linear - reference).max():.3e}",
             f"max |circular - linear|                  = {difference.max():.3f}",
             f"corrupted pixels (|diff| > 1e-9 max|g|)  = {int(np.sum(difference > tolerance))}"
             f"   (band of 2 px on every side: {size ** 2 - (size - 4) ** 2})",
             f"max |circular - linear| in the interior [2:-2, 2:-2] = {difference[2:-2, 2:-2].max():.3e}",
             "",
             "padded grid P (both axes) vs exactness of the 'same'-cropped result",
             f"{'P':>6s}{'max|g_P - g_linear|':>22s}{'corrupted pixels':>18s}"]
    for padded in [512, 513, 514, 515, 516, 520, 1024]:
        result = linear_convolve_fft(image, kernel, size=(padded, padded))
        error = np.abs(result - linear)
        lines.append(f"{padded:6d}{error.max():22.3e}{int(np.sum(error > tolerance)):18d}")

    # a 31x31 box blur makes the same effect wide enough to see at full size
    box = np.ones((31, 31)) / 31 ** 2
    box_circular = circular_convolve_fft(image, box)
    box_linear = linear_convolve_fft(image, box)
    box_difference = np.abs(box_circular - box_linear)
    lines += ["", "31x31 box blur, same comparison (for the figure)",
              f"max |circular - linear| = {box_difference.max():.3f}, corrupted pixels = "
              f"{int(np.sum(box_difference > 1e-9 * box_linear.max()))} "
              f"(band of 15 px: {size ** 2 - (size - 30) ** 2})"]
    write_table(out / "wraparound_table.txt", lines)

    # left column: the laplacian's top-left 24x24 corner, right column: the box blur, whole image
    zoom = slice(0, 24)
    limit = np.percentile(np.abs(linear), 99)
    fig, axes = plt.subplots(3, 2, figsize=(8, 10.6))
    left = [(circular[zoom, zoom], "laplacian corner, circular", "gray", -limit, limit),
            (linear[zoom, zoom], "laplacian corner, zero-padded linear", "gray", -limit, limit),
            (difference[zoom, zoom], "laplacian corner, |circular - linear|", "magma", 0, difference.max())]
    right = [(box_circular, "31x31 box, circular", "gray", 0, 255),
             (box_linear, "31x31 box, zero-padded linear", "gray", 0, 255),
             (box_difference, "31x31 box, |circular - linear|", "magma", 0, box_difference.max())]
    for ax, (data, title, cmap, vmin, vmax) in zip(axes[:, 0], left):
        shown = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.02)
    for ax, (data, title, cmap, vmin, vmax) in zip(axes[:, 1], right):
        shown = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(shown, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("P1.3a left: 5x5 laplacian, top-left 24x24 corner; right: 31x31 box blur, whole image",
                 fontsize=11)
    save_figure(fig, out / "wraparound.png")


def pipeline(image, kernel, padded):
    rows, cols = image.shape
    kh, kw = kernel.shape
    padded_image = np.zeros((padded, padded))
    padded_image[:rows, :cols] = image
    padded_kernel = np.zeros((padded, padded))
    padded_kernel[:kh, :kw] = kernel
    return np.real(np.fft.ifft2(np.fft.fft2(padded_image) * np.fft.fft2(padded_kernel)))


def peak_memory(fn, *args):
    tracemalloc.start()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def padding_cost(out):
    from scipy.fft import next_fast_len  # checking only, to show what scipy pads to
    image = test_image()
    kernel = LAPLACIAN_5
    size = image.shape[0]
    linear = size + kernel.shape[0] - 1
    sizes = [("N (circular)", size), ("N + M - 1", linear),
             ("next_fast_len real", next_fast_len(linear, True)),
             ("next_fast_len complex", next_fast_len(linear, False)),
             ("2N (problem 2 rule)", 2 * size)]

    # interleave the sizes round by round so slow drift in clock speed hits all of them
    rounds = 41
    times = {padded: [] for _, padded in sizes}
    for _, padded in sizes:
        pipeline(image, kernel, padded)
    for _ in range(rounds):
        for _, padded in sizes:
            start = time.perf_counter()
            pipeline(image, kernel, padded)
            times[padded].append(time.perf_counter() - start)
    median = {padded: float(np.median(t)) for padded, t in times.items()}
    memory = {padded: peak_memory(pipeline, image, kernel, padded) for _, padded in sizes}
    base_ops = size ** 2 * np.log2(size)

    lines = ["1.3b cost of zero padding, N = 512, M = 5, numpy.fft complex pipeline",
             f"time: median of {rounds} interleaved runs of fft2(image), fft2(kernel), product, ifft2",
             "memory: tracemalloc peak of the same pipeline", "",
             f"{'grid':24s}{'P':>6s}{'prime factors':>18s}{'(P/N)^2':>9s}"
             f"{'P^2logP ratio':>15s}{'time ms':>10s}{'time ratio':>12s}{'peak MiB':>10s}{'mem ratio':>11s}"]
    for name, padded in sizes:
        ops = padded ** 2 * np.log2(padded)
        lines.append(f"{name:24s}{padded:6d}{prime_factors(padded):>18s}{(padded / size) ** 2:9.4f}"
                     f"{ops / base_ops:15.4f}{1000 * median[padded]:10.3f}"
                     f"{median[padded] / median[size]:12.3f}"
                     f"{memory[padded] / 2**20:10.2f}{memory[padded] / memory[size]:11.3f}")

    from scipy.signal import fftconvolve  # checking only
    t_scipy = median_time(fftconvolve, image, kernel, "same", repeats=rounds)
    lines += ["", f"scipy.signal.fftconvolve(mode='same') on the same input: {1000 * t_scipy:.3f} ms",
              f"it pads each axis to next_fast_len({linear}, real=True) = {next_fast_len(linear, True)} "
              "and uses rfftn"]
    write_table(out / "padding_cost.txt", lines)


def prime_factors(n):
    """e.g. 540 -> '2^2*3^3*5'."""
    powers = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            powers[d] = powers.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        powers[n] = powers.get(n, 0) + 1
    return "*".join(f"{p}^{k}" if k > 1 else str(p) for p, k in powers.items())


EXPERIMENTS = {
    "dft_validation": dft_validation,
    "conjugate_symmetry": conjugate_symmetry,
    "shift_theorem": shift_theorem,
    "convolution_theorem": convolution_theorem,
    "sinusoid_recovery": sinusoid_recovery,
    "phase_magnitude": phase_magnitude,
    "wraparound": wraparound,
    "padding_cost": padding_cost,
}


def run(names=None):
    out = results_dir(1)
    for name in names or EXPERIMENTS:
        print(f"\n== problem 1: {name}")
        start = time.perf_counter()
        EXPERIMENTS[name](out)
        print(f"  done in {time.perf_counter() - start:.2f} s")
