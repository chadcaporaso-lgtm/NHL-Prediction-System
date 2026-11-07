"""Profile the PuckAPI prop history sample and check it against NHL box scores.

Inputs
  data/puckapi/prop_history/*.json            (fetch_prop_history.py)
  v1 data/raw/nhl_api/skater_summary_20252026.parquet   (complete 2025-26 skater game logs)

Outputs
  data/puckapi/prop_history_all.parquet        one row per player-game-market
  docs/audit/prop_history_summary.json

Checks
  P1  Recorded result agrees with the NHL box score (shots, points).
  P2  Line and price sanity: two-sided margin, no impossible prices.
  P3  How well the de-vigged closing price predicts over/under (log loss,
      calibration by price bucket). This is the market a prop model must beat.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUMMARY = os.path.join(ROOT, "docs/audit/prop_history_summary.json")


def american_to_prob(a: pd.Series) -> pd.Series:
    a = a.astype(float)
    return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "data/puckapi/prop_history/*.json"))):
    d = json.load(open(path))
    r = d["response"]
    for g in r["games"]:
        rows.append({"player_id": r["player_id"], "player_name": r["player_name"], "market": r["market"], **g})
df = pd.DataFrame(rows)
df["game_id"] = df.game_id.astype(int)
df = df.drop_duplicates(["player_id", "market", "game_id"])
df.to_parquet(os.path.join(ROOT, "data/puckapi/prop_history_all.parquet"), index=False)

out: dict = {
    "rows": len(df),
    "players": int(df.player_id.nunique()),
    "by_market": df.market.value_counts().astype(int).to_dict(),
    "date_range": [df.game_date.min(), df.game_date.max()],
    "books_per_line_median": float(df.books.median()),
    "lines": {m: d.line.value_counts().sort_index().astype(int).to_dict() for m, d in df.groupby("market")},
}

# P1: agreement with NHL box scores (2025-26 regular season)
box = pd.read_parquet(os.path.join(ROOT, "data/drive/NHL_Prediction_System/data/raw/nhl_api/skater_summary_20252026.parquet"))[["gameId", "playerId", "shots", "points"]]
m = df.merge(box, left_on=["game_id", "player_id"], right_on=["gameId", "playerId"], how="inner")
m["nhl_value"] = np.where(m.market == "player_shots_on_goal", m.shots, m.points)
graded = m[m.actual.notna()]
out["P1_rows_matched_to_nhl_box_score"] = len(graded)
out["P1_actual_disagrees_with_nhl"] = int((graded.actual != graded.nhl_value).sum())

# P2: price sanity
df["p_over_raw"] = american_to_prob(df.over_odds)
df["p_under_raw"] = american_to_prob(df.under_odds)
df["overround"] = df.p_over_raw + df.p_under_raw
out["P2_overround_quantiles"] = df.overround.quantile([0, 0.05, 0.5, 0.95, 1]).round(4).tolist()
out["P2_impossible_prices"] = int(((df.over_odds.abs() < 100) | (df.under_odds.abs() < 100)).sum())

# P3: market accuracy on graded, non-push games
g = df[df.result.isin(["over", "under"])].copy()
g["p_over"] = g.p_over_raw / g.overround
g["y"] = (g.result == "over").astype(int)
p = g.p_over.clip(1e-6, 1 - 1e-6)
g["ll"] = -(g.y * np.log(p) + (1 - g.y) * np.log(1 - p))
out["P3_market"] = {}
for mk, d in g.groupby("market"):
    base = d.y.mean()
    out["P3_market"][mk] = {
        "graded_games": len(d),
        "over_rate": round(float(base), 4),
        "mean_market_p_over": round(float(d.p_over.mean()), 4),
        "market_log_loss": round(float(d.ll.mean()), 4),
        "constant_log_loss": round(float(-(base * np.log(base) + (1 - base) * np.log(1 - base))), 4),
        "calibration_by_price": {
            str(k): {"n": int(len(x)), "mean_p": round(float(x.p_over.mean()), 3), "over_rate": round(float(x.y.mean()), 3)}
            for k, x in d.groupby(pd.cut(d.p_over, [0, 0.4, 0.5, 0.6, 0.7, 1.0]), observed=True)
        },
    }
json.dump(out, open(SUMMARY, "w"), indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
