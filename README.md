# Digital Image Processing, Assignment 2

Problems 1 to 6 of `DIP_A2.pdf` (problem 7 appears in the marks table but has no
text in the handout, so it is out of scope). Every operation a problem asks to
implement is written with NumPy array arithmetic; SciPy and scikit-image are only
used to check answers, for the SSIM metric, and for connected-component counts.

## Setup

The supplied images are not in git. Put the extracted `A2/` folder from the
handout zip in the repository root, so that for example `A2/Q1,4/q1.png` exists.
The two photos for 5.2 and 6.4 go in `photos/`, named `noisy*` and `blur*`.

```text
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The measurements in the report were made with Python 3.11.9 and the exact
versions in `requirements.txt`; `results/logs/environment_information.txt`
records the machine.

## Running

All commands run from the repository root.

```text
python src/run_experiments.py                     # problems 1 to 6
python src/run_experiments.py --problem 2         # one problem
python src/run_experiments.py --problem 1 --experiment shift_theorem
python src/run_experiments.py --list              # experiment names
python -m unittest discover -s tests -v           # quick correctness tests
```

Each run prints its tables, writes them with the figures to `results/problem_k/`,
and copies the terminal output to `results/logs/`. On the machine in the report,
problems 1, 2, 3, 4 and 6 take 5 to 25 seconds each and problem 5 about four
minutes, because the 1024 x 1536 colour reference is filtered many times while the
comparison filters are tuned.

## Status

Problems 5.2 (real noisy photo) and 6.4 (real motion-blurred photo) need photos
taken by the student. Their code runs and reports "BLOCKED" until a `noisy*` and a
`blur*` file are placed in `photos/`; then
`python src/run_experiments.py --problem 5 --experiment real_noise` and
`python src/run_experiments.py --problem 6 --experiment real_blur` complete them.

## Layout

```text
src/common/       image io, metrics, padding and fft helpers, plotting, machine info
src/problem_k/    algorithms for problem k and experiments.py, its experiment driver
src/run_experiments.py
tests/            unittest checks against analytic cases and library references
results/          tables (.txt), figures (.png), run screenshots, logs
docs/             audit notes, requirements checklist, dated predictions
report/           report.tex and report.pdf
tools/            screenshot capture and screenshot check scripts (macOS)
```

## Screenshots

`*_terminal.png` files are real screenshots of Terminal windows running the
commands above, taken with `tools/capture_run.sh`. `tools/check_screenshots.py`
reads them back with the macOS Vision text recogniser and checks that each run
finished, shows no error, matches its saved log, and has nothing typed after the
final prompt.
