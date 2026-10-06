#!/usr/bin/env python3
"""Convenience entry point: ``python3 run.py all``.

Exists so the repository is runnable from a clean checkout without setting
``PYTHONPATH`` or worrying about which directory you are in. It is a thin
wrapper -- all logic lives in :mod:`name_match.cli`.

    python3 run.py all        # generate dataset, train, evaluate, write report
    python3 run.py test       # run the full test suite
    python3 run.py all --test # both, in that order
"""

from __future__ import annotations

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Every default output path in the package ("data/model.json",
# "reports/results.md") is relative, and the CLI resolves them against the
# current directory. Importing the package is not enough to make the run
# self-contained -- without this, `python3 /path/to/run.py all` from a
# different directory succeeds and scatters a second copy of the generated
# dataset and reports into that directory. The brief's reviewer starts from
# "a clean checkout" and may well start anywhere.
os.chdir(REPO_ROOT)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        print("usage: python3 run.py {all|generate|train|evaluate|report|test}")
        return 0 if argv else 1

    command = argv[0]
    extra = argv[1:]

    if command == "test":
        return run_tests()

    if command == "all" and "--test" in extra:
        extra = [arg for arg in extra if arg != "--test"]
        status = run_cli(["all", *extra])
        if status:
            return status
        return run_tests()

    return run_cli([command, *extra])


def run_cli(args: list[str]) -> int:
    from name_match.cli import main as cli_main

    return cli_main(args)


def run_tests() -> int:
    print("Running test suite...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
        cwd=REPO_ROOT,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
