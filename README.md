# Digital Image Processing, Assignment 2

Problems 1 to 6 of `DIP_A2.pdf` (problem 7 appears in the marks table but has no
text in the handout, so it is out of scope). The report is `report/report.pdf`.

Every operation a problem asks to implement is written with NumPy array
arithmetic, including the PSNR and SSIM metrics and the connected-component
counts. SciPy is only called for the comparisons the handout asks for
(`convolve2d` and `fftconvolve` in 1.3, `gaussian_filter` in 2.2d) and for the
sine integral used to draw one predicted curve. scikit-image and
`scipy.ndimage.label` only appear in the unit tests, to check my SSIM and
component labelling.

## Setup

The supplied images and my own photos are not in the repository. Put the
extracted `A2/` folder from the handout zip in the repository root, so that for
example `A2/Q1,4/q1.png` exists, and the photos in `photos/`:

```text
photos/noisy.jpg              own noisy photo (5.2)
photos/blur1.jpg              own motion-blurred photo (6.4)
photos/reference_for_blur.jpg sharp photo of the same scene, a control for the blur estimate
```

```text
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The results were produced with Python 3.11.9 and the exact versions in
`requirements.txt`; `results/logs/environment_information.txt` records the
machine.

## Running

All commands run from the repository root.

```text
python src/run_experiments.py                     # problems 1 to 6
python src/run_experiments.py --problem 2         # one problem
python src/run_experiments.py --problem 1 --experiment shift_theorem
python src/run_experiments.py --list              # experiment names
python -m unittest discover -s tests -v           # 56 unit tests
```

Each run prints its tables, writes them with the figures to `results/problem_k/`,
and copies the terminal output to `results/logs/`. Problems 1, 2, 3, 4 and 6 take
5 to 60 seconds each and problem 5 about four minutes, because the
1024 x 1536 colour reference is filtered many times while the comparison filters
are tuned.

## Layout

```text
src/common/        image io, metrics, padding and fft helpers, plotting, machine info
src/problem_k/     algorithms for problem k and experiments.py, its experiment driver
src/run_experiments.py
tests/             unittest checks against analytic cases and library references
results/           tables (.txt), figures (.png), run screenshots, logs
docs/predictions/  the dated predictions for 1.2 and 2.2 (pdf)
report/            report.pdf
```

The `*_terminal.png` files are screenshots of Terminal windows running the
commands above, one per experiment and one per full problem run, and
`results/logs/` holds the matching terminal output.
