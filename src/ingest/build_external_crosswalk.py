"""Rebuild the external closing-odds crosswalk with the newest 2026-27 file (decisions 37, 40).

The external closing CSV (one row per game: opening and consensus closing
moneyline, puck line and total, de-vigged) was first used through v2's
crosswalk (built 2026-10-26: 2025-26 complete, 2026-27 to about 2026-10-25).
The owner supplied newer 2026-27 files on 2026-11-09: first 231 games to
2026-11-01 (SHA-256 f357e746...), then 282 games to 2026-11-08 (f324cc32...,
the one in use, data/external/closing_csv/nhl_2026_27_odds.csv). Games after
the newest NHL schedule refresh cannot be matched to official results yet;
they are reported and left out until the next refresh. This script keeps v2's 2025-26 rows (the 2025-26 CSV is
byte-identical to v2's source, SHA-256 459588b3...) and maps the new
2026-27 rows to official NHL games by start time and teams, with the game
date and teams as a fallback (one game's listed start differs by an hour).

Checks (printed and written to docs/audit/external_crosswalk_<last game date>.json):
  * every new row maps to exactly one official game and the scores agree;
  * on the 184 games in v2's crosswalk, the official game ids are identical;
  * revisions: how many of those 184 games' prices changed between files, and
    against the previous owner-supplied file when it is present.

Inputs (gitignored): data/external/closing_csv/nhl_2026_27_odds.csv,
  v2's crosswalk, data/live/<newest>/games_schedule_live.parquet
Output: data/external/external_official_crosswalk.parquet
"""

from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OLD = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/data/evaluation/external_closing_comparison/20261026T210505Z_bed9e020/external_official_crosswalk.parquet")
NEW_CSV = os.path.join(ROOT, "data/external/closing_csv/nhl_2026_27_odds.csv")
PREV_CSV = os.path.join(ROOT, "data/external/closing_csv/nhl_2026_27_odds_2026-11-09.csv")  # 231 games, optional
OUT = os.path.join(ROOT, "data/external/external_official_crosswalk.parquet")
PRICE_COLS = ["open_home_win_prob", "close_home_win_prob", "close_spread_home", "close_home_cover_prob",
              "open_total", "close_total", "close_over_prob"]


def main():
    old = pd.read_parquet(OLD)
    new = pd.read_csv(NEW_CSV)
    sched = pd.read_parquet(sorted(glob.glob(os.path.join(ROOT, "data/live/*/games_schedule_live.parquet")))[-1])
    sched_end = str(sched.game_date.max())[:10]
    later = new.date_et > sched_end
    pending = new[later]
    new = new[~later].copy()
    team_id = dict(zip(old.home_team, old.official_home_id))
    new["official_home_id"], new["official_away_id"] = new.home_team.map(team_id), new.away_team.map(team_id)
    assert new.official_home_id.notna().all() and new.official_away_id.notna().all(), "unmapped team name"
    s = sched.rename(columns={"game_id": "official_game_id", "home_team_id": "official_home_id",
                              "away_team_id": "official_away_id"})
    s["date_et"] = pd.to_datetime(s.scheduled_utc).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    k = ["official_home_id", "official_away_id"]
    a = new.merge(s[["official_game_id", "scheduled_utc"] + k], left_on=["start_time_utc"] + k, right_on=["scheduled_utc"] + k, how="left")
    miss = a.official_game_id.isna()
    b = new[miss.values].merge(s[["official_game_id", "date_et"] + k], on=["date_et"] + k, how="left")
    a.loc[miss, "official_game_id"] = b.official_game_id.values
    assert a.official_game_id.notna().all() and a.official_game_id.is_unique, "unmatched or duplicate game"
    a = a.merge(s[["official_game_id", "home_final_score", "away_final_score"]], on="official_game_id")
    checks = {"file_sha256": hashlib.sha256(open(NEW_CSV, "rb").read()).hexdigest(),
              "file_games": int(len(a) + len(pending)), "schedule_ends": sched_end,
              "pending_after_schedule": int(len(pending)),
              "new_rows": int(len(a)), "matched_by_start_time": int((~miss).sum()), "matched_by_date": int(miss.sum()),
              "scores_agree": int(((a.home_score == a.home_final_score) & (a.away_score == a.away_final_score)).sum())}
    # Same official ids as v2 on the games both files cover.
    o26 = old[old.external_season == "2026-27"]
    j = a.merge(o26, left_on="game_id", right_on="external_event_id", suffixes=("", "_old"))
    checks["overlap_with_v2_crosswalk"] = int(len(j))
    checks["official_id_identical"] = int((j.official_game_id == j.official_game_id_old).sum())
    rev = {}
    for c in PRICE_COLS + ["bookmakers"]:
        x, y = j[c].astype(float), j[f"{c}_old"].astype(float)
        changed = ~(np.isclose(x, y) | (x.isna() & y.isna()))
        rev[c] = {"changed": int(changed.sum()), "max_abs_change": round(float((x - y).abs().max()), 4)}
    checks["revisions_vs_v2_copy"] = rev
    if os.path.exists(PREV_CSV):
        prev = pd.read_csv(PREV_CSV)
        k2 = pd.read_csv(NEW_CSV).merge(prev, on="game_id", suffixes=("", "_old"))
        rev2 = {}
        for c in PRICE_COLS + ["bookmakers", "home_score", "away_score"]:
            x, y = k2[c].astype(float), k2[f"{c}_old"].astype(float)
            changed = ~(np.isclose(x, y) | (x.isna() & y.isna()))
            rev2[c] = {"changed": int(changed.sum()), "max_abs_change": round(float((x - y).abs().max()), 4)}
        checks["revisions_vs_previous_owner_file"] = {"shared_games": int(len(k2)), "columns": rev2}
    checks["new_games"] = int(len(a) - len(j))
    # Assemble: v2's 2025-26 rows unchanged, new 2026-27 rows in the same schema.
    a = a.rename(columns={"game_id": "external_event_id", "season": "external_season",
                          "home_team_id": "external_home_team_id", "away_team_id": "external_away_team_id"})
    a["season"], a["game_type"] = 20262027, 2
    a["official_game_id"] = a.official_game_id.astype(int)
    a["source_csv_sha256"] = hashlib.sha256(open(NEW_CSV, "rb").read()).hexdigest()
    keep = [c for c in old.columns if c in a.columns]
    out = pd.concat([old[old.external_season == "2025-26"], a[keep]], ignore_index=True)
    out.to_parquet(OUT, index=False)
    checks["output_rows"] = {"2025-26": int((out.external_season == "2025-26").sum()), "2026-27": int((out.external_season == "2026-27").sum())}
    last = str(pd.read_csv(NEW_CSV).date_et.max())
    json.dump(checks, open(os.path.join(ROOT, f"docs/audit/external_crosswalk_{last}.json"), "w"), indent=2)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
