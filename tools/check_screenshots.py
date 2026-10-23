"""check every run screenshot against the log it should show.

reads the text in each results/**/*_terminal.png with the macos vision framework
(tools/ocr.swift, compiled on first use) and checks that
  - no error text is visible,
  - the run reached its end ('finished in', or 'OK' for the unit tests),
  - the visible 'started' time or 'finished in' time matches the saved log,
  - the log path printed at the end is the expected one,
  - nothing was typed after the final prompt.

usage, from the repository root: python tools/check_screenshots.py
"""
import re
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "tools" / ".build"
PATH_MATCH = 0.85
ERRORS = ["Traceback", "Error", "command not found", "No such file", "FAILED"]


def ocr_binary():
    binary = BUILD / "ocr"
    source = ROOT / "tools" / "ocr.swift"
    if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
        BUILD.mkdir(parents=True, exist_ok=True)
        subprocess.run(["swiftc", "-O", str(ROOT / "tools" / "ocr.swift"), "-o", str(binary)], check=True)
    return binary


def read_text(paths):
    out = subprocess.run([str(ocr_binary())] + [str(p) for p in paths],
                         capture_output=True, text=True, check=True).stdout
    texts = {}
    for block in out.split("=== ")[1:]:
        path, _, body = block.partition("\n")
        texts[Path(path)] = body
    return texts


def expected_log(shot):
    if shot.parent.name == "logs":
        return None
    problem = shot.parent.name.split("_")[1]
    stem = shot.stem.removesuffix("_terminal")
    name = f"problem_{problem}.log" if stem == f"problem_{problem}_run" else f"problem_{problem}_{stem}.log"
    return ROOT / "results" / "logs" / name


def squash(text):
    # ocr often reads 'log' as '10g' or '1og'; compare letters and digits only
    return re.sub(r"[^a-z0-9]", "", re.sub(r"[1l][0o]g", "log", text.lower()))


def path_similarity(text, log_name):
    """best similarity between the expected log path and any 'log written to' line.
    the long path line is where ocr errs most ('1ogs', 'oroblem', 'lo0'), so an
    exact match is too strict; a different log name still scores well below 0.85."""
    expected = squash("results/logs/" + log_name)
    best = 0.0
    for line in text.splitlines():
        if "writ" not in line.lower():
            continue
        tail = squash(line.split("results", 1)[-1] if "results" in line else line)
        best = max(best, SequenceMatcher(None, expected, squash("results") + tail).ratio())
    return best


def check(shot, text):
    problems = []
    for word in ERRORS:
        if word in text:
            problems.append(f"shows '{word}'")
    log = expected_log(shot)
    if log is None:
        if not re.search(r"\bOK\b", text) or "Ran " not in text:
            problems.append("test summary ('Ran N tests' / 'OK') not visible")
    else:
        if "finished in" not in text:
            problems.append("no 'finished in' line, run may not have completed")
        if not log.exists():
            problems.append(f"log {log.name} missing")
        else:
            log_text = log.read_text()
            started = re.search(r"started (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
            finished = re.search(r"problem\s*\d,?\s*finished\s*in\s*(\d+\.\d)\s*s", text)
            # the fuzzy path match below cannot tell problem 4 from 5, this can
            number = re.search(r"problem\s*(\d),?\s*finished", text)
            if number and number.group(1) != log.name.split("_")[1].split(".")[0]:
                problems.append(f"shows problem {number.group(1)}, expected {log.name}")
            log_finished = re.search(r"problem \d finished in (\d+\.\d) s", log_text)
            if started:
                if started.group(1) not in log_text:
                    problems.append(f"start time {started.group(1)} is not the one in {log.name}")
            elif finished and log_finished:
                if finished.group(1) != log_finished.group(1):
                    problems.append(f"finish time {finished.group(1)} s differs from {log.name}")
            else:
                problems.append("neither the start time nor the final timing could be matched to the log")
            if path_similarity(text, log.name) < PATH_MATCH:
                problems.append(f"'{log.name}' not printed")
    # capture_run.sh hides the cursor, so the last prompt must end at '%'
    prompts = [line for line in text.splitlines() if "MacBook-Pro" in line]
    if not prompts:
        problems.append("final prompt not visible")
    elif not prompts[-1].rstrip().endswith("%"):
        problems.append(f"text after the final prompt: '{prompts[-1]}'")
    return problems


def main():
    shots = sorted((ROOT / "results").glob("**/*_terminal.png"))
    texts = read_text(shots)
    failed = 0
    for shot in shots:
        problems = check(shot, texts.get(shot, ""))
        status = "ok  " if not problems else "FAIL"
        failed += bool(problems)
        print(f"{status} {shot.relative_to(ROOT)}" + ("" if not problems else ": " + "; ".join(problems)))
    print(f"\n{len(shots)} screenshots, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
