"""list numbers in the report that do not appear in any result file.

every decimal number in report/sections/*.tex is looked up among the numbers in
results/**/*.txt, results/logs/*.log and docs/predictions/*.md. a report value
matches when some source value rounds to it at the precision it is printed with.
what is left over is printed for a manual check: mostly derived values such as
percentages, ratios and differences.

usage, from the repository root: python tools/check_report_numbers.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_TEX = re.compile(r"(-?\d+\.?\d*)\s*\\times\s*10\^\{(-?\d+)\}")
SCIENTIFIC = re.compile(r"-?\d+\.\d+e[-+]?\d+")
DECIMAL = re.compile(r"(?<![\w.])-?\d+\.\d+(?![\d.])")


def source_values():
    values = set()
    files = list((ROOT / "results").glob("**/*.txt")) + list((ROOT / "results" / "logs").glob("*.log"))
    files += list((ROOT / "docs" / "predictions").glob("*.md"))
    for path in files:
        text = path.read_text(errors="ignore")
        for token in SCIENTIFIC.findall(text):
            values.add(float(token))
        for token in DECIMAL.findall(text):
            values.add(float(token))
    return sorted(values)


def report_numbers(text):
    found = []
    for mantissa, exponent in SCIENTIFIC_TEX.findall(text):
        found.append((f"{mantissa}e{exponent}", float(f"{mantissa}e{exponent}"), mantissa))
    text = SCIENTIFIC_TEX.sub(" ", text)
    for token in DECIMAL.findall(text):
        found.append((token, float(token), token))
    return found


def matches(value, printed, sources):
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    if "e" not in printed.lower() and value != 0:
        return any(round(abs(s), decimals) == round(abs(value), decimals) for s in sources)
    # scientific: compare the mantissa at its printed precision
    for s in sources:
        if s == 0 or value == 0:
            if s == value:
                return True
            continue
        if abs(s / value - 1) < 0.5 * 10 ** (-decimals) / abs(float(printed)) + 1e-12:
            return True
    return False


def main():
    sources = source_values()
    unmatched = []
    for tex in sorted((ROOT / "report" / "sections").glob("*.tex")):
        for line_number, line in enumerate(tex.read_text().splitlines(), start=1):
            for token, value, printed in report_numbers(line):
                if not matches(value, printed, sources):
                    unmatched.append(f"{tex.name}:{line_number}: {token}")
    print("\n".join(unmatched))
    print(f"\n{len(unmatched)} numbers without a direct match (check these by hand)")


if __name__ == "__main__":
    main()
