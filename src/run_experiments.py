"""run the assignment 2 experiments.

examples, from the repository root:
    python src/run_experiments.py                     # every problem
    python src/run_experiments.py --problem 2         # one problem
    python src/run_experiments.py --problem 1 --experiment shift_theorem
    python src/run_experiments.py --list
"""
import argparse
import importlib
import sys
import time
from datetime import datetime

from common.environment import write_environment
from common.paths import LOGS_DIR

PROBLEMS = ["1", "2", "3", "4", "5", "6"]


class Tee:
    """copy everything printed to the terminal into a log file as well."""

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "w")

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def parse_args():
    parser = argparse.ArgumentParser(description="run assignment 2 experiments")
    parser.add_argument("--problem", choices=PROBLEMS, action="append",
                        help="problem to run, can be repeated (default: all)")
    parser.add_argument("--experiment", action="append",
                        help="experiment name inside the chosen problem, can be repeated")
    parser.add_argument("--list", action="store_true", help="list experiments and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    problems = args.problem or PROBLEMS
    modules = {p: importlib.import_module(f"problem_{p}.experiments") for p in problems}

    if args.list:
        for problem, module in modules.items():
            print(f"problem {problem}: {', '.join(module.EXPERIMENTS)}")
        return
    if args.experiment and len(problems) != 1:
        sys.exit("--experiment needs exactly one --problem")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    environment = write_environment(LOGS_DIR / "environment_information.txt")

    for problem, module in modules.items():
        suffix = "_" + "_".join(args.experiment) if args.experiment else ""
        log_path = LOGS_DIR / f"problem_{problem}{suffix}.log"
        tee = Tee(sys.stdout, log_path)
        sys.stdout = tee
        try:
            print(f"problem {problem}, started {datetime.now():%Y-%m-%d %H:%M:%S}")
            print("\n".join(environment))
            start = time.perf_counter()
            module.run(args.experiment)
            print(f"\nproblem {problem} finished in {time.perf_counter() - start:.1f} s")
        finally:
            sys.stdout = tee.stream
            tee.close()
        print(f"log written to {log_path}")


if __name__ == "__main__":
    main()
