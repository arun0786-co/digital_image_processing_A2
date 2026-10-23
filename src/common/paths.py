from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "A2"
PHOTOS_DIR = ROOT / "photos"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"

INPUTS = {
    "q1": DATA_DIR / "Q1,4" / "q1.png",
    "q4": DATA_DIR / "Q1,4" / "q4.png",
    "characters": DATA_DIR / "Q2_images" / "Fig0441(a)(characters_test_pattern).tif",
    "text_gaps": DATA_DIR / "Q2_images" / "Fig0419(a)(text_gaps_of_1_and_2_pixels).tif",
    "thumb_print": DATA_DIR / "Q2_images" / "Fig0457(a)(thumb_print).tif",
    "cameraman": DATA_DIR / "Q2_images" / "cameraman.tif",
    "tunnel": DATA_DIR / "Q3" / "tunnel_image.jpg",
    "reference": DATA_DIR / "Q5,6" / "reference_image.jpg",
}


def input_path(name):
    path = INPUTS[name]
    if not path.exists():
        raise FileNotFoundError(f"missing input '{name}', expected at {path}")
    return path


def results_dir(problem):
    path = RESULTS_DIR / f"problem_{problem}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_photo(stem):
    """first file in photos/ whose name starts with stem, or None."""
    if not PHOTOS_DIR.exists():
        return None
    matches = sorted(p for p in PHOTOS_DIR.iterdir()
                     if p.name.lower().startswith(stem) and not p.name.startswith("."))
    return matches[0] if matches else None
