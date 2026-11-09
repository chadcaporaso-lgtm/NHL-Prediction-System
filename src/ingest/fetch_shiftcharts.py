"""Fetch NHL shift charts in the Claude session (api.nhle.com, free, no key).

Same reduction as the Colab fallback (src/ingest/colab_fetch_shiftcharts.py):
per-skater seconds by strength and teammate pair seconds at 5-on-5 and on the
power play, for every game in the restored skater files. Raw shifts are
cached per season in data/shiftcharts/ (gitignored), so a rerun fetches only
new games.

Outputs
  data/shiftcharts/shift_toi.parquet        skater-game seconds by strength
  data/shiftcharts/shift_pairs.parquet      teammate pair seconds per game
  data/shiftcharts/shift_fetch_log.parquet  status and row count per game

Usage: python3 src/ingest/fetch_shiftcharts.py
"""

from __future__ import annotations

import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src/ingest"))
from colab_fetch_shiftcharts import fetch_all  # noqa: E402

OUT = os.path.join(ROOT, "data/shiftcharts")

if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(ROOT, "data/drive/**/skater_summary_*.parquet"), recursive=True))
    files += sorted(glob.glob(os.path.join(ROOT, "data/live/*/skater_summary_20262027.parquet")))[-1:]
    toi, pairs, games = fetch_all(files, os.path.join(OUT, "raw"))
    for name, d in {"shift_toi": toi, "shift_pairs": pairs, "shift_fetch_log": games}.items():
        d.to_parquet(os.path.join(OUT, f"{name}.parquet"), index=False)
