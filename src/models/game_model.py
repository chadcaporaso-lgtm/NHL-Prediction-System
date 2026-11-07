"""Game models for moneyline, puck line and total goals.

Each team's regulation goals are modeled as Poisson with mean lambda from
team form, the opponent's form, the opposing starter and schedule features.
The two teams are independent in regulation except for ties: independent
Poisson scores predict about 16% of games tied after regulation where 21-25%
are, so tied scorelines are scaled up by a factor (1 + theta) and the
distribution renormalized, with theta set on the training seasons so the
predicted tie rate matches the actual one. Then:
  moneyline    P(home win) = P(H > A) + P(H = A) * q, q = home share of
               overtime/shootout wins in the training seasons
  puck line    home -1.5 covers if home wins by 2+, home +1.5 covers unless
               away wins by 2+ (overtime games end one goal apart)
  totals       official total = H + A, plus 1 if tied after regulation (the
               overtime goal or the shootout credit)
Independence and the Poisson assumption miss things like empty-net goals,
so every market probability is calibrated (Platt scaling: logistic
regression on the logit) using out-of-sample predictions from earlier test
seasons only. 2021-22, the first test season, has no earlier predictions
and is reported uncalibrated.

Walk-forward: test seasons 2021-22 to 2026-27, each trained on every earlier
season from 2019-20.

Variant: `python3 src/models/game_model.py elo` adds the team's pre-game
Elo rating minus the opponent's (src/features/build_elo.py) as a feature and
writes *_elo outputs; the default model and its files are unchanged.

Outputs
  data/models/game_predictions[_elo].parquet  one row per test game
  docs/models/game_results[_elo].json         metrics on all games
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_SEASONS = [20212022, 20222023, 20232024, 20242025, 20252026, 20262027]
MAXG = 15
OWN = ["ewm20_xgf", "ewm60_xgf", "ewm20_gf", "ewm60_gf", "ewm20_sf", "ewm20_cf_pct",
       "ewm20_xgf60_5v5", "ewm20_pp_xgf60", "ewm20_pp_toi"]
OPP = ["ewm20_xga", "ewm60_xga", "ewm20_ga", "ewm60_ga", "ewm20_sa",
       "ewm20_xga60_5v5", "ewm20_pk_xga60", "ewm20_pk_toi", "starter_sv_l40", "starter_prior_apps"]
FEATURES = OWN + [f"opp_{c}" for c in OPP] + ["is_home", "is_playoff", "rest", "opp_rest", "back_to_back", "opp_back_to_back",
                                              "league_reg_gpg_l300"]
MARKETS = ["ml", "home_by2", "away_by2", "over55", "over65"]
VARIANT = sys.argv[1] if len(sys.argv) > 1 else ""
SUFFIX = f"_{VARIANT}" if VARIANT else ""
if VARIANT == "elo":
    FEATURES = FEATURES + ["elo_diff"]


def load() -> pd.DataFrame:
    f = pd.read_parquet(os.path.join(ROOT, "data/features/game_team_features.parquet"))
    for c in [c for c in f.columns if c.startswith("ewm")]:
        f[c] = f[c].fillna(f[c].mean())
    f["starter_sv_l40"] = f.starter_sv_l40.fillna(0.905)
    f["starter_prior_apps"] = f.starter_prior_apps.fillna(0).clip(upper=200)
    f["rest"] = f.days_rest.fillna(7).clip(upper=7)
    f["is_playoff"] = (f.game_type == 3).astype(int)
    f["league_reg_gpg_l300"] = f.league_reg_gpg_l300.fillna(f.league_reg_gpg_l300.mean())
    opp = f[["game_id", "team"] + OPP + ["rest", "back_to_back"]].rename(
        columns={**{c: f"opp_{c}" for c in OPP}, "team": "opponent", "rest": "opp_rest", "back_to_back": "opp_back_to_back"})
    f = f.merge(opp, on=["game_id", "opponent"], validate="one_to_one")
    if VARIANT == "elo":
        e = pd.read_parquet(os.path.join(ROOT, "data/features/elo_games.parquet"))[["game_id", "std_rh", "std_ra"]]
        f = f.merge(e, on="game_id", how="left", validate="many_to_one")
        f["elo_diff"] = np.where(f.is_home == 1, f.std_rh - f.std_ra, f.std_ra - f.std_rh)
        f["elo_diff"] = f.elo_diff.fillna(0.0)
    return f


def joint_scores(lh, la, theta=0.0):
    """[game, home goals, away goals] probabilities, ties inflated by theta."""
    k = np.arange(MAXG + 1)
    ph = stats.poisson.pmf(k[None, :], lh[:, None])
    pa = stats.poisson.pmf(k[None, :], la[:, None])
    joint = ph[:, :, None] * pa[:, None, :]
    joint = joint * (1 + theta * np.eye(MAXG + 1))[None]
    return joint / joint.sum((1, 2), keepdims=True)


def fit_theta(lh, la, actual_tie_rate):
    from scipy.optimize import brentq
    return brentq(lambda t: joint_scores(lh, la, t).diagonal(axis1=1, axis2=2).sum(1).mean() - actual_tie_rate, -0.5, 3.0)


def game_probs(lh, la, q, theta=0.0):
    """Market probabilities from regulation score probabilities."""
    k = np.arange(MAXG + 1)
    joint = joint_scores(lh, la, theta)
    diff = k[:, None] - k[None, :]
    tot = k[:, None] + k[None, :]
    p_home_reg = (joint * (diff > 0)).sum((1, 2))
    p_tie = (joint * (diff == 0)).sum((1, 2))
    official_total = tot + (diff == 0)  # tied games get one more goal
    out = {"ml": p_home_reg + q * p_tie, "p_tie": p_tie,
           "home_by2": (joint * (diff >= 2)).sum((1, 2)), "away_by2": (joint * (diff <= -2)).sum((1, 2))}
    for line in [5.5, 6.5]:
        out[f"over{int(line * 10)}"] = (joint * (official_total > line)).sum((1, 2))
    out["_joint"], out["_official_total"] = joint, official_total
    return out


def p_total_over(joint, official_total, line):
    over = (joint * (official_total > line)).sum((1, 2))
    push = (joint * (official_total == line)).sum((1, 2))
    return over / (1 - push)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def outcomes(g: pd.DataFrame) -> pd.DataFrame:
    margin = g.final_home - g.final_away
    total = g.final_home + g.final_away
    return pd.DataFrame({"ml": (margin > 0).astype(int), "home_by2": (margin >= 2).astype(int),
                         "away_by2": (margin <= -2).astype(int), "over55": (total > 5.5).astype(int),
                         "over65": (total > 6.5).astype(int)}, index=g.index)


def walk_forward(f: pd.DataFrame):
    games, joints = [], {}
    for s in TEST_SEASONS:
        tr, te = f[f.season < s], f[f.season == s]
        glm = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=2000)).fit(tr[FEATURES], tr.reg_goals_for)
        gbm = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.03, max_leaf_nodes=15,
                                            min_samples_leaf=100, l2_regularization=1.0, random_state=0).fit(tr[FEATURES], tr.reg_goals_for)
        trh = tr[tr.is_home == 1]
        tied = trh[trh.decision_type != "REG"]
        q = float(tied.won_final.astype(bool).mean())
        tie_rate = float((trh.decision_type != "REG").mean())
        trh_a = tr[tr.is_home == 0].set_index("game_id").loc[trh.game_id]
        thetas = {m: fit_theta(est.predict(trh[FEATURES]), est.predict(trh_a[FEATURES]), tie_rate)
                  for m, est in [("glm", glm), ("gbm", gbm)]}
        te = te.assign(mu_glm=glm.predict(te[FEATURES]), mu_gbm=gbm.predict(te[FEATURES]))
        h, a = te[te.is_home == 1].set_index("game_id"), te[te.is_home == 0].set_index("game_id")
        a = a.loc[h.index]
        g = pd.DataFrame({"season": s, "game_date": h.game_date, "home": h.team, "away": a.team,
                          "final_home": h.final_score_for, "final_away": a.final_score_for,
                          "reg_home": h.reg_goals_for, "reg_away": a.reg_goals_for, "decision_type": h.decision_type,
                          "q_ot_home": q, "theta_glm": thetas["glm"], "theta_gbm": thetas["gbm"]}, index=h.index)
        for m in ["glm", "gbm"]:
            g[f"lh_{m}"], g[f"la_{m}"] = h[f"mu_{m}"], a[f"mu_{m}"]
            pr = game_probs(h[f"mu_{m}"].values, a[f"mu_{m}"].values, q, thetas[m])
            for k in MARKETS + ["p_tie"]:
                g[f"{k}_{m}_raw"] = pr[k]
            joints[(s, m)] = (g.index.values, pr["_joint"], pr["_official_total"])
        games.append(g)
        print(s, "train team-games", len(tr), "test games", len(g), "q_ot_home", round(q, 3),
              "train tie rate", round(tie_rate, 3), "theta", {k: round(v, 3) for k, v in thetas.items()})
    return pd.concat(games), joints


def calibrate(g: pd.DataFrame):
    """Platt scaling per market, fitted on earlier test seasons only."""
    y = outcomes(g)
    cal = {}
    for m in ["glm", "gbm"]:
        for k in MARKETS:
            g[f"{k}_{m}"] = g[f"{k}_{m}_raw"]
            for s in TEST_SEASONS[1:]:
                past = g.season < s
                lr = LogisticRegression(C=1e6).fit(logit(g.loc[past, f"{k}_{m}_raw"]).values.reshape(-1, 1), y.loc[past, k])
                now = g.season == s
                g.loc[now, f"{k}_{m}"] = lr.predict_proba(logit(g.loc[now, f"{k}_{m}_raw"]).values.reshape(-1, 1))[:, 1]
                cal[(m, k, s)] = (float(lr.intercept_[0]), float(lr.coef_[0, 0]))
    return g, cal


def score_all_games(g: pd.DataFrame):
    y = outcomes(g)
    out = {}
    for s, d in g.groupby("season"):
        row = {"games": len(d), "home_win_rate": round(float(y.loc[d.index, "ml"].mean()), 4),
               "reg_tie_rate": round(float((d.decision_type != "REG").mean()), 4)}
        past = g.season < s
        for k in MARKETS:
            base = y.loc[past, k].mean() if past.any() else y.loc[d.index, k].mean()
            row[k] = {"constant": round(logloss(np.full(len(d), base), y.loc[d.index, k]), 4)}
            for m in ["glm", "gbm"]:
                row[k][f"{m}_raw"] = round(logloss(d[f"{k}_{m}_raw"], y.loc[d.index, k]), 4)
                row[k][m] = round(logloss(d[f"{k}_{m}"], y.loc[d.index, k]), 4)
        row["pred_tie_rate_glm"] = round(float(d.p_tie_glm_raw.mean()), 4)
        out[str(s)] = row
    return out


if __name__ == "__main__":
    f = load()
    g, joints = walk_forward(f)
    g, cal = calibrate(g)
    g.to_parquet(os.path.join(ROOT, f"data/models/game_predictions{SUFFIX}.parquet"))
    res = {"features": FEATURES, "all_games": score_all_games(g),
           "calibration": {f"{m}|{k}|{s}": v for (m, k, s), v in cal.items()}}
    json.dump(res, open(os.path.join(ROOT, f"docs/models/game_results{SUFFIX}.json"), "w"), indent=2)
    for s, r in res["all_games"].items():
        print(s, r["games"], "tie", r["reg_tie_rate"], r["pred_tie_rate_glm"], {k: r[k] for k in MARKETS})
