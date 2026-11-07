"""Evaluate Elo: on its own as a moneyline model, and as a game-model feature.

Elo alone: the standard (margin-adjusted) pre-game home win probability, raw
and Platt-calibrated on earlier test seasons only (as in game_model.py).
Game model: the default GLM versus the same GLM with elo_diff added.
Both are compared with the closing line on the same games.

Inputs: data/features/elo_games.parquet, data/models/game_predictions[_elo].parquet,
        data/models/game_market_graded[_elo].parquet
Output: docs/models/elo_eval.json
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST = [20212022, 20222023, 20232024, 20242025, 20252026, 20262027]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4)


e = pd.read_parquet(os.path.join(ROOT, "data/features/elo_games.parquet"))
base = pd.read_parquet(os.path.join(ROOT, "data/models/game_predictions.parquet"))
withe = pd.read_parquet(os.path.join(ROOT, "data/models/game_predictions_elo.parquet"))
d = base[["season", "final_home", "final_away", "ml_glm"]].join(withe[["ml_glm"]], rsuffix="_elo")
d = d.join(e.set_index("game_id")[["std_ph", "flat_ph"]])
d["y"] = (d.final_home > d.final_away).astype(int)
# Platt calibration of Elo using earlier test seasons only
d["elo_cal"] = d.std_ph
for s in TEST[1:]:
    past, now = d.season < s, d.season == s
    lr = LogisticRegression(C=1e6).fit(logit(d.loc[past, "std_ph"]).values.reshape(-1, 1), d.loc[past, "y"])
    d.loc[now, "elo_cal"] = lr.predict_proba(logit(d.loc[now, "std_ph"]).values.reshape(-1, 1))[:, 1]

out = {"all_games_moneyline": {}, "vs_market": {}}
for s, x in d.groupby("season"):
    past = d[d.season < s]
    const = past.y.mean() if len(past) else x.y.mean()
    out["all_games_moneyline"][str(s)] = {
        "games": len(x), "constant": ll(np.full(len(x), const), x.y),
        "elo_flat": ll(x.flat_ph, x.y), "elo_std": ll(x.std_ph, x.y), "elo_std_cal": ll(x.elo_cal, x.y),
        "game_model": ll(x.ml_glm, x.y), "game_model_with_elo": ll(x.ml_glm_elo, x.y)}

gm = pd.read_parquet(os.path.join(ROOT, "data/models/game_market_graded.parquet"))
ge = pd.read_parquet(os.path.join(ROOT, "data/models/game_market_graded_elo.parquet"))
for (src, mk, s), x in gm.groupby(["source", "market", "season"]):
    xe = ge[(ge.source == src) & (ge.market == mk) & (ge.season == s)].set_index("game_id").reindex(x.game_id)
    row = {"games": len(x), "market": ll(x.p_mkt, x.y), "game_model": ll(x.p_cal, x.y),
           "game_model_with_elo": ll(xe.p_cal.values, x.y)}
    if mk == "ml":
        el = d.reindex(x.game_id)
        row["elo_alone"] = ll(el.elo_cal.values, x.y)
        row["elo_raw"] = ll(el.std_ph.values, x.y)
    out["vs_market"][f"{src}|{mk}|{s}"] = row
json.dump(out, open(os.path.join(ROOT, "docs/models/elo_eval.json"), "w"), indent=2)
for k, v in out["all_games_moneyline"].items():
    print(k, v)
for k, v in out["vs_market"].items():
    print(k, v)
