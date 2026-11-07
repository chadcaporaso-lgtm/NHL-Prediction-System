"""Weekly live-tracking update, run here after the Colab refresh is downloaded.

Steps, in order, each stopping the run on failure:
  1. fetch new SportsGameOdds events (day after the newest raw file to yesterday)
  2. extract prop and game closing lines
  3. check the NHL web API feed against the audited tables (overlap games)
  4. rebuild the skater game table and features, the game features
  5. leakage tests
  6. match prop lines, rerun the shots model, the game model and market grading
  7. rebuild team Elo ratings (power ranking) and run its leakage tests
  8. grade the frozen rules (fingerprints checked first)
  9. redraw the charts in docs/figures/
Usage: python3 src/live/weekly_update.py <refresh date, e.g. 2026-11-14>
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFRESH = sys.argv[1]


def run(*args):
    print("\n>>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


newest = max(os.path.basename(p)[:10] for p in glob.glob(os.path.join(ROOT, "data/sgo/raw/*.json.gz")))
first = (date.fromisoformat(newest) + timedelta(days=1)).isoformat()
yesterday = (date.today() - timedelta(days=1)).isoformat()
if first <= yesterday:
    run("src/ingest/fetch_sgo_events.py", first, yesterday)
run("src/ingest/extract_sgo_props.py")
run("src/ingest/extract_sgo_game_odds.py")
run("src/audit/audit_live_feed.py", REFRESH)
run("src/features/build_skater_games.py")
run("src/features/build_skater_features.py")
run("src/features/build_game_features.py")
run("tests/test_skater_feature_leakage.py")
run("tests/test_game_feature_leakage.py")
run("src/ingest/match_sgo_props.py")
run("src/models/prop_model.py", "shots")
run("src/audit/profile_sgo_game_odds.py")
run("src/models/game_model.py")
run("src/models/game_market.py")
run("src/features/build_elo.py")
run("tests/test_elo_leakage.py")
run("src/live/track.py")
run("src/reporting/make_figures.py")
