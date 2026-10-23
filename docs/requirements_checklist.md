# requirements checklist

problems 1 to 6 only. problem 7 appears in the marks table but has no text in
the handout and is out of scope (see `docs/audit.md`).

global rules from page 1:

- g1. "from scratch" means numpy array arithmetic. no scipy.ndimage,
  scipy.signal, cv2 or skimage for anything a problem asks to implement; they
  may only check answers. numpy.fft is allowed everywhere.
- g2. every reported number comes from this machine. state cpu, ram, numpy
  version. -> `results/logs/environment_information.txt`, report setup table.

status: `todo`, `done`, `blocked (reason)`.

## problem 1: dft (15)

| id | requirement | code | output | report | status |
|---|---|---|---|---|---|
| 1.0 | synthetic image from >= 3 sinusoids, ground truth recorded | `problem_1/dft.py`, `experiments.py` | `sinusoid_ground_truth.txt` | 1.1c | done |
| 1.1 | 1-d dft matrix W_N, 2-d dft F = W f W^T, max error vs fft2 table | `problem_1/dft.py` | `dft_error_table.txt` | 1.1 | done |
| 1.1a | conjugate symmetry derivation, redundant region, why full spectrum shown | report + numeric check | `conjugate_symmetry.txt` | 1.1a | done |
| 1.1b | shift and circular convolution theorem tables on test image | `experiments.py` | `shift_theorem_table.txt`, `convolution_theorem_table.txt` | 1.1b | done |
| 1.1c | recover (u, v, A, phi) from peaks, derive delta f | `experiments.py` | `sinusoid_recovery.txt`, `.png` | 1.1c | done |
| 1.2p | dated prediction: test-image phase + white-noise magnitude | `docs/predictions/` | committed before run | 1.2 | done |
| 1.2a | phase + flat magnitude, compare with prediction, why flat beats swapping | `experiments.py` | `phase_magnitude.png` | 1.2a | done |
| 1.2b | magnitude with zero phase, explain via symmetry | `experiments.py` | same figure | 1.2b | done |
| 1.3a | circular vs zero-padded linear conv, wraparound evidence, minimal padding derivation | `experiments.py` | `wraparound.png`, table | 1.3a | done |
| 1.3b | memory factor, runtime increase, relate to fftconvolve padding | `experiments.py` | `padding_cost.txt` | 1.3b | done |

## problem 2: frequency-domain filtering (15)

pipeline: pad N x N to P x P with P = 2N (zero padding), centre, multiply real
zero-phase H, invert, real part, crop. non-square images are padded to a square
P = 2 max(H, W).

| id | requirement | code | output | report | status |
|---|---|---|---|---|---|
| 2.0 | synthetic 512x512 step image, B = 200 square of side 256 on A = 50 | `problem_2/experiments.py` | | 2 | done |
| 2.1 | `make_filter(kind, P, D0, n=None, highpass=False)`; radial plot of all six | `problem_2/filters.py` | `transfer_functions.png` | 2.1 | done |
| 2.1a | H at D = D0 analytic and numeric; why same D0 is unfair; matched gaussian D0' | report + check | `cutoff_semantics.txt` | 2.1a | done |
| 2.1b | alpha(r) for 90, 95, 99, 99.5 % on characters pattern; explain | `experiments.py` | `power_retention.txt` | 2.1b | done |
| 2.1c | inverse design (n, D0) from H(10) = 16/17, H(40) = 1/17; min n for H(2 D0) <= 1 % | report + check | `inverse_design.txt` | 2.1c | done |
| 2.2p | dated ringing prediction: Si closed form, overshoot %, first-peak distance vs D0, P; gaussian | `docs/predictions/` | committed before run | 2.2 | done |
| 2.2a | ILPF D0 in {10, 20, 40, 80}: measured vs predicted table; corner profile | `experiments.py` | `ideal_ringing_table.txt`, `.png` | 2.2a | done |
| 2.2b | BLPF D0 = 40, n in {1, 2, 3, 5, 10, 20}: overshoot; kernel ratio min h / max h; non-negative kernel proof | `experiments.py` | `butterworth_order_table.txt` | 2.2b | done |
| 2.2c | ILPF / BLPF(2) / GLPF(matched) grid on characters at 2.1b radii, ringing marked | `experiments.py` | `filter_grid.png` | 2.2c | done |
| 2.2d | sigma_s derivation; D0 in {10, 40, 160} vs scipy gaussian_filter error table | `experiments.py` | `gaussian_equivalence.txt` | 2.2d | done |
| 2.3a | cameraman + noise sigma 20 (seed), PSNR vs D0 in [5, 150], three curves, optimum table | `experiments.py` | `psnr_sweep.png`, `.txt` | 2.3a | done |
| 2.3b | GLPF repair of text gaps, chosen D0 justified, ideal at same radius | `experiments.py` | `text_repair.png` | 2.3b | done |
| 2.3c | IHPF, BHPF(4), GHPF on thumbprint at common radius; means vs H_HP(P/2, P/2); HFE; binary ridge map | `experiments.py` | `highpass.png`, `.txt` | 2.3c | done |

## problem 3: homomorphic filtering (10)

| id | requirement | code | output | report | status |
|---|---|---|---|---|---|
| 3.0 | unevenly lit photo | supplied `A2/Q3/tunnel_image.jpg` | | 3 | done |
| 3.1 | model I = L R, log makes it additive, why log before fft, what goes wrong in log-magnitude | report | | 3.1 | done |
| 3.2 | H = (gH - gL)(1 - exp(-c D^2 / D0^2)) + gL on log I, exp back | `problem_3/homomorphic.py` | | 3.2 | done |
| 3.2a | sweep gL, gH, D0 as image grid; good and halo settings identified | `experiments.py` | `sweep_grid_*.png` | 3.2 | done |
| 3.2b | global histogram equalisation (A1 implementation); local std in shadow region, bright-region blowout; table | `experiments.py` | `local_contrast.txt` | 3.2 | done |

## problem 4: edge-preserving smoothing (15)

| id | requirement | code | output | report | status |
|---|---|---|---|---|---|
| 4.0 | clean ground truth + gaussian noise, sigma reported | supplied `A2/Q1,4/q4.png` | | 4 | done |
| 4.1 | bilateral (sigma_s, sigma_r, normalised) and classical kuwahara from scratch | `problem_4/bilateral.py`, `kuwahara.py` | | 4.1 | done |
| 4.1a | definition of matched smoothing strength and the matching procedure | report + `experiments.py` | `matching.txt` | 4.1a | done |
| 4.1b | PSNR and SSIM vs clean; cropped edge comparisons | `experiments.py` | `edge_preservation_comparison.png` | 4.1b | done |
| 4.2a | why hard argmin gives blockiness, not blur or ringing | report | | 4.2a | done |
| 4.2b | modified kuwahara; before/after crops at same window; blockiness metric | `kuwahara.py` | `blockiness.png`, `.txt` | 4.2b | done |
| 4.2c | per-pixel add / mult / exp counts for kuwahara, modified, bilateral | report + timing | `complexity.txt` | 4.2c | done |
| 4.3a | synthetic noisy step generator across NCR; NCR* | `experiments.py` | | 4.3 | done |
| 4.3b | edge-preservation metric vs NCR with NCR* annotated | `experiments.py` | `ncr_breakdown.png` | 4.3 | done |
| 4.3c | range-kernel explanation of breakdown | report | | 4.3 | done |

## problem 5: wiener denoising (15)

| id | requirement | code | output | report | status |
|---|---|---|---|---|---|
| 5.0 | clean instructor RGB reference >= 1024 x 1024 | supplied `A2/Q5,6/reference_image.jpg` | | 5 | done |
| 5.1 | derive MMSE W = Sff / (Sff + Snn); implement on [0, 1] images | `problem_5/wiener.py` | | 5.1 | done |
| 5.1a | Snn from known variance or flat region; Sff estimation method stated | `wiener.py` | | 5.1 | done |
| 5.1b | sigma in {5, 10, 20, 40}, same noise per sigma, no clipping before filtering | `experiments.py` | | 5.1 | done |
| 5.1c | compare with P2 gaussian, butterworth and P4 bilateral; PSNR, SSIM, data range, SSIM impl, parameters, oracle labels | `experiments.py` | `denoising_table.txt` | 5.1 | done |
| 5.1d | shrinkage vs frequency and noise power; vs fixed cutoff | report + plot | `wiener_gain.png` | 5.1 | done |
| 5.2a | sigma 20 with 0.5 Snn and 2 Snn, side by side, PSNR / SSIM | `experiments.py` | `misspecification.png`, `.txt` | 5.2 | done |
| 5.2b | why correct wiener still blurs edges | report | | 5.2 | done |
| 5.2c | own noisy photo: noise estimate, crops, residuals, spectra, no PSNR | `experiments.py` | `real_noise_*.png` | 5.2 | blocked (photo needed) |

## problem 6: motion blur (20)

| id | requirement | code | output | report | status |
|---|---|---|---|---|---|
| 6.1a | derive h(x, y); discretisation and normalisation | `problem_6/motion.py` | `psf.png` | 6.1 | done |
| 6.1b | boundary convention used for blur and deconvolution; blur + noise at stated SNR (defined) | `experiments.py` | | 6.1 | done |
| 6.1c | derive OTF sinc(L(fx cos + fy sin)), zero lines m / L, spacing, orientation | report + check | `otf_zeros.txt` | 6.1 | done |
| 6.2 | naive inverse with stated floor; log spectra of G and F_hat, predicted zeros marked; noise amplification; one-sentence why | `experiments.py` | `inverse_spectrum.png` | 6.2 | done |
| 6.3 | wiener deconvolution derivation; scaling; K as NSR; log K sweep; low-K and high-K failures; optimum K at >= 3 noise levels | `experiments.py` | `k_sweep.png`, `.txt` | 6.3 | done |
| 6.4 | own blurred photo: theta, L from spectrum, stripes marked, PSF, wiener, report theta L K, artifact vs model | `experiments.py` | `real_blur_*.png` | 6.4 | blocked (photo needed) |

## cross-cutting

| id | requirement | status |
|---|---|---|
| x1 | unit tests, quick, separate from sweeps | done |
| x2 | real screenshots of runs in `results/problem_k/` | done |
| x3 | report.tex compiled to pdf, all numbers traceable to result files | done |
| x4 | small commits, private remote, push after audit | in progress |
