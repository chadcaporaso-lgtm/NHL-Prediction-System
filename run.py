"""One entry point for the project.

  python3 run.py check                  run the pipeline and leakage tests on synthetic
                                        data (no real data needed; what CI runs)
  python3 run.py restore <manifest>     rebuild data/ from the Drive transfer parts
                                        (src/inventory/restore_core_inputs.py)
  python3 run.py update [refresh date]  full rebuild on the real data: new odds,
                                        features, leakage tests, models, live grading,
                                        charts (src/live/weekly_update.py). The date
                                        defaults to the newest folder in data/live/
  python3 run.py test                   the leakage tests on the real data only

See CLAUDE.md ("Restoring the data in a new session") for getting the real data.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LEAKAGE_TESTS = ["tests/test_skater_feature_leakage.py", "tests/test_game_feature_leakage.py",
                 "tests/test_probable_starter_leakage.py", "tests/test_elo_leakage.py"]


def run(*args) -> int:
    print(">>", " ".join(args), flush=True)
    return subprocess.run([sys.executable, *args], cwd=ROOT).returncode


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "help"
    if cmd == "check":
        return run("tests/synthetic/run_synthetic.py")
    if cmd == "restore" and len(argv) == 2:
        return run("src/inventory/restore_core_inputs.py", argv[1])
    if cmd == "update":
        if len(argv) > 1:
            date = argv[1]
        else:
            dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "data/live/*")) if os.path.isdir(d))
            if not dirs:
                print("no data/live/<date>/ folder: restore the data first (python3 run.py restore <manifest>)")
                return 1
            date = os.path.basename(dirs[-1])
        return run("src/live/weekly_update.py", date)
    if cmd == "test":
        for t in LEAKAGE_TESTS:
            if run(t):
                return 1
        return 0
    print(__doc__)
    return 0 if cmd == "help" else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
