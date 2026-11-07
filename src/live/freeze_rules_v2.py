"""Freeze version 2 of the live-tracking rules (run once, 2026-11-07).

Changes from version 1 (docs/live/frozen_rules.json, decisions 20, 23, 25),
recorded in decision 29:
  Puck line  the game model now uses the probable-starter variant
             (`game_model.py probable`): the opposing goalie's save % and
             experience are weighted by the probable-starter model instead of
             taken from the goalie who actually started, so every input is
             known before the starter is announced. Same calibrated GLM,
             2-point edge and SportsGameOdds consensus close.
             Window: games after 2026-11-07 (no game had been graded under v1).
             Fingerprint: the 2026-27 games through 2026-11-05, the last date
             with complete team and goalie data when the rule was frozen
             (the v1 fingerprint counted games through 2026-11-06, a date the
             team feed did not yet cover, so it would have failed next week).
  Shots on goal  unchanged: copied from version 1 with its window (games after
             2026-11-06), blend coefficients and fingerprint.

Output: docs/live/frozen_rules_v2.json (committed). Version 1 stays in the
repository as the record of the original rule.
"""

from __future__ import annotations

import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PL_WINDOW_AFTER = "2026-11-07"
PL_FINGERPRINT_THROUGH = "2026-11-05"

if __name__ == "__main__":
    v1 = json.load(open(os.path.join(ROOT, "docs/live/frozen_rules.json")))
    games = pd.read_parquet(os.path.join(ROOT, "data/models/game_predictions_probable.parquet"))
    fp = games[(games.season == 20262027) & (games.game_date <= PL_FINGERPRINT_THROUGH)]
    ps = pd.read_parquet(os.path.join(ROOT, "data/features/probable_starter_team.parquet"))
    ps_fp = ps[(ps.season == 20262027) & (ps.game_date <= PL_FINGERPRINT_THROUGH)]
    rules = {
        "version": 2, "frozen_on": "2026-11-07", "supersedes": "docs/live/frozen_rules.json",
        "sog": {**v1["sog"], "window_starts_after": v1["window_starts_after"], "unchanged_from_version": 1},
        "puck_line": {
            "model": "glm calibrated, probable-starter variant (game_model.py probable)",
            "threshold": 0.02, "price": "SportsGameOdds consensus close",
            "window_starts_after": PL_WINDOW_AFTER,
            "fingerprint": {"through": PL_FINGERPRINT_THROUGH, "games": int(len(fp)),
                            "mean_lh_glm": round(float(fp.lh_glm.mean()), 8),
                            "theta_glm": round(float(fp.theta_glm.iloc[0]), 8),
                            "q_ot_home": round(float(fp.q_ot_home.iloc[0]), 8),
                            "starter_team_games": int(len(ps_fp)),
                            "mean_exp_sv_l40": round(float(ps_fp.exp_sv_l40.mean()), 8)}},
    }
    json.dump(rules, open(os.path.join(ROOT, "docs/live/frozen_rules_v2.json"), "w"), indent=2)
    print(json.dumps(rules, indent=2))
