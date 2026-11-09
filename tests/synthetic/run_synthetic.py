"""Run the pipeline and its leakage tests on synthetic data (CI check).

Copies the repository's code, tests and docs to a temporary folder, generates
the synthetic data set there (tests/synthetic/make_data.py), and runs the
feature builders, every leakage test, the probable-starter model and the game
models. Nothing in the working copy, its data/ or its committed results is
touched. Odds-dependent steps (SportsGameOdds matching, market grading, live
tracking) need licensed data and are not run here.

Usage: python3 tests/synthetic/run_synthetic.py [--keep]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STEPS = [
    ["tests/test_shift_pairs.py"],
    ["src/features/build_skater_games.py"],
    ["src/features/build_skater_features.py"],
    ["tests/test_skater_feature_leakage.py"],
    ["src/features/build_game_features.py"],
    ["tests/test_game_feature_leakage.py"],
    ["src/features/build_probable_starter.py"],
    ["tests/test_probable_starter_leakage.py"],
    ["src/features/build_elo.py"],
    ["tests/test_elo_leakage.py"],
    ["src/models/game_model.py"],
    ["src/models/game_model.py", "probable"],
]


def main(keep: bool = False) -> int:
    work = tempfile.mkdtemp(prefix="nhl_synthetic_")
    for d in ["src", "tests", "docs"]:
        shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d), ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run([sys.executable, "tests/synthetic/make_data.py", work], cwd=work, check=True)
    for d in ["data/features", "data/models", "data/clean"]:
        os.makedirs(os.path.join(work, d), exist_ok=True)
    failed = []
    for step in STEPS:
        t0 = time.time()
        r = subprocess.run([sys.executable, *step], cwd=work, capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else "FAILED"
        print(f"{status:6s} {' '.join(step)}  ({time.time() - t0:.0f}s)", flush=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
            failed.append(step)
            break  # later steps depend on earlier outputs
    if keep:
        print("kept", work)
    else:
        shutil.rmtree(work, ignore_errors=True)
    print("all steps passed" if not failed else f"failed at {' '.join(failed[0])}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main("--keep" in sys.argv))
