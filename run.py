"""One entry point for the project.

  python3 run.py check                  run the pipeline and leakage tests on synthetic
                                        data (no real data needed; what CI runs)
  python3 run.py restore <manifest>     rebuild data/ from the Drive transfer parts
                                        (src/inventory/restore_core_inputs.py), then
                                        place owner-supplied odds (place_external.py)
  python3 run.py update [refresh date]  full rebuild on the real data: new odds,
                                        features, leakage tests, models, live grading,
                                        charts (src/live/weekly_update.py). The date
                                        defaults to the newest folder in data/live/
  python3 run.py test                   the leakage tests on the real data only
  python3 run.py extras                 optional player data and variants (decisions
                                        34-39): fetches NHL shift charts, NHL Edge and
                                        MoneyPuck game logs (about 3 hours), builds and
                                        leakage-tests the features, reruns the variant
                                        models and the earlier-season market tests

See CLAUDE.md ("Restoring the data in a new session") for getting the real data.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["shots_edge", "shots_mp", "points_lm", "points_edge", "points_lm_edge", "points_mp", "points_all"]
BLEND_NAMES = ["sog_edge", "sog_mp", "points_lm", "points_edge", "points_lm_edge", "points_mp", "points_all"]
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
        return run("src/inventory/restore_core_inputs.py", argv[1]) or run("src/inventory/place_external.py")
    if cmd == "extras":
        steps = [["src/ingest/fetch_shiftcharts.py"], ["src/ingest/fetch_edge.py"], ["src/ingest/fetch_moneypuck_skaters.py"],
                 ["src/features/build_linemates.py"], ["tests/test_linemate_leakage.py"],
                 ["src/features/build_edge_features.py"], ["tests/test_edge_leakage.py"],
                 ["src/features/build_mp_features.py"], ["tests/test_mp_leakage.py"]]
        steps += [["src/models/prop_model.py", v] for v in VARIANTS]
        steps += [["src/models/prop_market_blend.py", n] for n in BLEND_NAMES]
        steps += [["src/models/sog_older_seasons.py", "variants"]]
        if os.path.exists(os.path.join(ROOT, "data/external/closing_csv/nhl_2026_27_odds_2026-11-09.csv")):
            steps += [["src/ingest/build_external_crosswalk.py"]]
        if os.path.exists(os.path.join(ROOT, "data/external/odds_scrape_2026-10/games_nhl.parquet")):
            steps += [["src/models/game_market_history.py"]]
        for st in steps:
            if run(*st):
                return 1
        return 0
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
