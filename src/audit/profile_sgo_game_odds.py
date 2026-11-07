"""Match SportsGameOdds game lines to NHL results and profile the closing market.

Input:  data/sgo/game_odds_close.parquet (extract_sgo_game_odds.py), v2 games_schedule
Output: data/sgo/game_odds_matched.parquet, docs/audit/sgo_game_odds_summary.json

Bets are graded on the official final score (a shootout win counts as one
goal), which is how sportsbooks settle moneyline, puck line and totals.
Probabilities are de-vigged proportionally. Pushes (whole-number totals) are
dropped from log loss.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEAM = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL"}
SCHED = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/data/normalized/official_incremental/20261103T160656Z_4414a3fb/games_schedule.parquet")


def implied(a):
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def devig(a, b):
    pa, pb = implied(a), implied(b)
    return pa / (pa + pb), pa + pb - 1


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


g = pd.read_parquet(os.path.join(ROOT, "data/sgo/game_odds_close.parquet"))
g["home_nhl"] = g.home.map(lambda t: TEAM.get(t, t))
g["away_nhl"] = g.away.map(lambda t: TEAM.get(t, t))
g["ts"] = pd.to_datetime(g.starts_at, utc=True)
s = pd.read_parquet(SCHED)
s["ts_nhl"] = pd.to_datetime(s.scheduled_utc, utc=True)
m = g.merge(s, left_on=["home_nhl", "away_nhl"], right_on=["home_abbrev", "away_abbrev"])
m = m[(m.ts - m.ts_nhl).abs() < pd.Timedelta(hours=12)].drop_duplicates("event_id")
m = m[m.home_final_score.notna()].copy()
m.to_parquet(os.path.join(ROOT, "data/sgo/game_odds_matched.parquet"), index=False)

out = {"events": int(len(g)), "matched_completed_games": int(len(m)),
       "date_range": [str(m.game_date.min()), str(m.game_date.max())],
       "by_season": {str(k): int(v) for k, v in m.season.value_counts().sort_index().items()},
       "by_game_type": {str(k): int(v) for k, v in m.game_type.value_counts().items()}}
margin = (m.home_final_score - m.away_final_score)
total = m.home_final_score + m.away_final_score

ml = m[m.ml_home_close.notna() & m.ml_away_close.notna()]
p, vig = devig(ml.ml_home_close, ml.ml_away_close)
y = (ml.home_final_score > ml.away_final_score).astype(int).values
po, _ = devig(ml.ml_home_open, ml.ml_away_open)
ok = ~np.isnan(po)
out["moneyline"] = {"games": int(len(ml)), "home_win_rate": round(float(y.mean()), 4),
                    "close_log_loss": round(logloss(p, y), 4), "open_log_loss": round(logloss(po[ok], y[ok]), 4),
                    "constant_log_loss": round(logloss(np.full(len(y), y.mean()), y), 4),
                    "median_margin": round(float(np.median(vig)), 4),
                    "abs_close_odds_over_1000": int((ml[["ml_home_close", "ml_away_close"]].abs() > 1000).any(axis=1).sum())}

pl = m[m.pl_home_close.notna() & m.pl_away_close.notna()].copy()
pl["consistent"] = pl.pl_home_close_line == -pl.pl_away_close_line
plc = pl[pl.consistent]
p, vig = devig(plc.pl_home_close, plc.pl_away_close)
y = ((margin[plc.index] + plc.pl_home_close_line) > 0).astype(int).values
out["puck_line"] = {"games": int(len(pl)), "line_consistent": int(len(plc)),
                    "home_line_counts": {str(k): int(v) for k, v in plc.pl_home_close_line.value_counts().items()},
                    "home_cover_rate": round(float(y.mean()), 4), "close_log_loss": round(logloss(p, y), 4),
                    "constant_log_loss": round(logloss(np.full(len(y), y.mean()), y), 4),
                    "median_margin": round(float(np.median(vig)), 4)}

tt = m[m.tot_over_close.notna() & m.tot_under_close.notna()].copy()
tt = tt[tt.tot_over_close_line == tt.tot_under_close_line]
push = total[tt.index] == tt.tot_over_close_line
t2 = tt[~push]
p, vig = devig(t2.tot_over_close, t2.tot_under_close)
y = (total[t2.index] > t2.tot_over_close_line).astype(int).values
out["totals"] = {"games": int(len(tt)), "pushes": int(push.sum()),
                 "line_counts": {str(k): int(v) for k, v in tt.tot_over_close_line.value_counts().items()},
                 "over_rate": round(float(y.mean()), 4), "close_log_loss": round(logloss(p, y), 4),
                 "constant_log_loss": round(logloss(np.full(len(y), y.mean()), y), 4),
                 "median_margin": round(float(np.median(vig)), 4)}

json.dump(out, open(os.path.join(ROOT, "docs/audit/sgo_game_odds_summary.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
