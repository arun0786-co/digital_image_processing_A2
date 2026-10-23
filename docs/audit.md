# initial audit

written 2026-10-23 16:20 IST, before any assignment code was written.

## repository state

- `DIP_A2/` is not a git repository and has no remote. it contains only the
  handout `DIP_A2.pdf` and the already extracted inputs under `A2/`.
- no `DIP_A2.zip` exists anywhere under the home folder (searched to depth 4),
  so there was no archive to preserve. the extracted files are left untouched.
- previous assignments, both next to this folder:
  - `../digital_image_processing_A0`: one commit on `main`,
    remote `git@github.com:arun0786-co/digital_image_processing_A0.git`.
  - `../DIP_Assignment_1`: one commit on `main`,
    remote `https://github.com/arun0786-co/DigitalImageProcessingAssignment1.git`.
- git identity (global): `Arun <arunsivan0786@gmail.com>`. no global
  `core.hooksPath`, only convenience aliases (`st`, `co`, `lg`, ...).
- python: shared venv `../.venv`, python 3.11.9, numpy 2.4.6, scipy 1.17.1,
  scikit-image 0.26.0, matplotlib 3.11.1, pillow 12.3.0. no opencv, no pytest.
- no local tex distribution (no pdflatex, latexmk or tectonic). pandoc is
  installed. both earlier reports were pdfTeX output from an online editor.
- machine: Apple M4 Pro, 12 cores (8 performance + 4 efficiency), 24 GiB RAM,
  macOS 15.7.7.

## inputs

| problem | file | notes |
|---|---|---|
| P1 | `A2/Q1,4/q1.png` | 512x512 uint8 grey, natural image |
| P2 | `A2/Q2_images/Fig0441(a)(characters_test_pattern).tif` | 688x688, 7 grey levels |
| P2 | `A2/Q2_images/Fig0419(a)(text_gaps_of_1_and_2_pixels).tif` | 444x508, binary |
| P2 | `A2/Q2_images/Fig0457(a)(thumb_print).tif` | 962x1026 |
| P2 | `A2/Q2_images/cameraman.tif` | 512x512 |
| P3 | `A2/Q3/tunnel_image.jpg` | 3644x3000 RGB phone photo, bright centre and dark walls |
| P4 | `A2/Q1,4/q4.png` | 512x512 clean piecewise-constant ground truth, 41 grey levels |
| P5, P6 | `A2/Q5,6/reference_image.jpg` | 2048x3072 RGB, clean reference |

## missing inputs

1. problem 7 (lens distortion, 10 marks) is listed in the marks table on page 1,
   but the pdf ends after problem 6 on page 10. there is no problem 7 text,
   formula or deliverable list anywhere in the handout. resolved 2026-10-23:
   the assignment covers problems 1 to 6 only, so problem 7 is out of scope.
2. a genuinely noisy photograph taken by the student (P5.2).
3. a genuinely motion-blurred photograph taken by the student (P6.4).

items 2 and 3 go in `photos/`. everything that does not depend on them is
built and run first.

## decisions

- remote: new private repo `arun0786-co/digital_image_processing_A2`, pushed
  only after the final audit.
- report: LaTeX compiled locally with tectonic (installed with homebrew).
- screenshots: real macOS window captures of the runs (`screencapture`).
- supplied images, the handout pdf and personal photos stay out of git.

## security and process audit

- `DIP_A2.pdf`: pdfTeX 1.40.29 with hyperref. 0 `/JavaScript`, `/JS`, `/Launch`,
  `/URI`, `/EmbeddedFile`, `/RichMedia`, `/SubmitForm`, `/GoToR`, `/XFA`,
  `/AcroForm`, 0 link annotations. one `/OpenAction`, which is hyperref's
  standard open-at-page-1 destination, not an action. no concern.
- image files are plain png, uncompressed tiff and baseline jpeg. the two jpegs
  carry camera exif (vivo 1920; Olympus C7000Z edited in Photoshop CS4, with a
  copyright tag for Vyacheslav Argenberg). neither has gps tags. the copyright
  tag is a reason not to republish the reference image in a public repo.
- the pdf and the tunnel image carry a macOS quarantine attribute from Safari,
  which only records the download source.
- no scripts, executables, hooks, ci files or credentials in `DIP_A2/`.
- A0 and A1 repos: no active git hooks (only `*.sample`), no submodules, no ci.
- `gh` is logged in as `arun0786-co` with repo scope (token not printed).
- claude processes at audit time: the Claude desktop app and its helpers, one
  unrelated `claude` cli session about 21 h old (pid 87120), and this session
  (pid 59824, started by `bridge start lily`). none were started by this task
  and none will be touched.

## conventions to follow

- A0: plain functions, lowercase snake_case names, short lowercase error
  messages (`raise ValueError("invalid method. ...")`), few comments.
- A1 (the closer model for numpy work): algorithms in their own module,
  experiment drivers that write `.txt` tables and `.png` figures to a per-problem
  results folder, `figsize=(9, 6)`, `dpi=180`, light grid, titles prefixed with
  the subproblem number, timing by median of repeats. comments are short and
  lowercase. the report is LaTeX article class with an experimental setup table
  first, then one section per problem numbered like the handout, booktabs tables
  with captions, and british spelling.

## planned structure

```
src/common/       io, metrics, fft helpers, plotting, environment info
src/problem_k/    algorithm module(s) plus experiments.py for each problem
src/run_experiments.py
tests/            unittest (stdlib) checks, quick, separate from sweeps
results/problem_k/, results/logs/
docs/predictions/ dated predictions, committed before the matching experiment
report/           report.tex and the compiled pdf
```
