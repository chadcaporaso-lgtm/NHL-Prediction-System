"""Shots-on-goal model against PuckAPI closing prices, 2023-24 to 2026-01-15.

SportsGameOdds closing prices start on 2026-01-16, so the shots model's
market tests so far cover one period. PuckAPI's archive goes back to 2023-24;
a sample was bought on 2026-11-08 (src/ingest/fetch_puckapi_sog_history.py):
random 25-game blocks for randomly drawn regular players, plus Connor
McDavid's full window. These lines were never used to build or tune anything.

Market: PuckAPI's median closing line across books with median over and under
prices, de-vigged proportionally (an approximation: the two medians need not
come from the same book). Grading: official NHL box-score shots
(data/models/sog_predictions.parquet), not the provider's result field
(decision 14). Pushes on whole-number lines are dropped.

Model: gradient-boosted negative binomial from prop_model.py, walk-forward
(each season predicted by a model trained on earlier seasons), so every
probability is out of sample.

Tests
  log loss       market vs model vs blend
  blend          y ~ logit(p_market) + logit(p_model) fitted on these lines;
                 the model coefficient measures information beyond the price
                 (standard errors clustered by player, the coarser grouping)
  frozen blend   the live rule's blend (fitted on 2026 SportsGameOdds lines,
                 docs/live/frozen_rules_v2.json) applied unchanged: out of
                 sample in data, though fitted on a later period
  ROI            4-point rules (model alone; frozen blend) at the median
                 closing price, clustered by player

Output: docs/models/sog_older_seasons_results.json (no purchased rows)
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def implied(a):
    a = np.asarray(a, dtype=float)
    return np.where(a < 0, -a / (100 - np.minimum(a, 0)), 100 / (np.maximum(a, 0) + 100))


def payout(a):
    a = np.asarray(a, dtype=float)
    return np.where(a < 0, 100 / -np.minimum(a, -1e-9), np.maximum(a, 0) / 100)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def p_over(line, mu, alpha):
    r = 1.0 / alpha
    pr = r / (r + mu)
    k = np.floor(line)
    over = stats.nbinom.sf(k, r, pr)
    push = np.where(line == k, stats.nbinom.pmf(k, r, pr), 0.0)
    return over / (1 - push)


def clustered_mean(x, g):
    x, g = np.asarray(x, float), np.asarray(g)
    m = x.mean()
    s = pd.Series(x - m).groupby(g).sum()
    return float(m), float(np.sqrt((s ** 2).sum()) / len(x))


def load() -> tuple[pd.DataFrame, dict]:
    rows = []
    for path in glob.glob(os.path.join(ROOT, "data/puckapi/sog_history/*.json")):
        d = json.load(open(path))
        for g in d.get("games", []):
            rows.append({"player_id": int(d["player_id"]), "game_id": int(g["game_id"]), "line": g["line"],
                         "open_line": g["open_line"], "over_odds": g["over_odds"], "under_odds": g["under_odds"],
                         "books": g["books"], "actual_provider": g["actual"]})
    raw = pd.DataFrame(rows)
    info = {"rows_fetched": len(raw), "duplicates": int(raw.duplicated(["player_id", "game_id"]).sum())}
    raw = raw.drop_duplicates(["player_id", "game_id"])
    pred = pd.read_parquet(os.path.join(ROOT, "data/models/sog_predictions.parquet"))
    d = raw.merge(pred, on=["player_id", "game_id"], how="left", validate="one_to_one")
    info["not_in_box_scores"] = int(d.shots.isna().sum())
    d = d.dropna(subset=["shots", "over_odds", "under_odds", "line"]).copy()
    info["provider_vs_official_shots_mismatch"] = int((d.actual_provider.notna() & (d.actual_provider != d.shots)).sum())
    alpha = json.load(open(os.path.join(ROOT, "docs/models/sog_results.json")))["alpha"]
    d["season"] = d.season.astype(int)
    d["alpha"] = d.season.map(lambda s: alpha[str(s)]["gbm"])
    po, pu = implied(d.over_odds), implied(d.under_odds)
    d["p_market"], d["margin"] = po / (po + pu), po + pu - 1
    d = d[d.shots != d.line].copy()  # pushes on whole-number lines
    d["y"] = (d.shots > d.line).astype(int)
    d["p_model"] = p_over(d.line.values, d.mu_gbm.values, d.alpha.values)
    info["graded_lines"] = len(d)
    return d, info


def rule(d: pd.DataFrame, p: pd.Series, thr=0.04) -> dict:
    edge = p - d.p_market
    b = d[edge.abs() > thr].copy()
    if b.empty:
        return {"bets": 0}
    over = edge[b.index] > 0
    price = np.where(over, b.over_odds, b.under_odds)
    win = np.where(over, b.y == 1, b.y == 0)
    profit = np.where(win, payout(price), -1.0)
    roi, se = clustered_mean(profit, b.player_id)
    _, se_g = clustered_mean(profit, b.game_id)
    return {"bets": int(len(b)), "players": int(b.player_id.nunique()), "under_share": round(float((~over).mean()), 3),
            "win_rate": round(float(win.mean()), 4), "roi": round(roi, 4), "roi_se_player_clustered": round(se, 4),
            "roi_se_game_clustered": round(se_g, 4)}


def summarize(d: pd.DataFrame, frozen) -> dict:
    out = {"lines": int(len(d)), "players": int(d.player_id.nunique()), "games": int(d.game_id.nunique()),
           "median_margin": round(float(d.margin.median()), 4), "over_rate": round(float(d.y.mean()), 4),
           "mean_p_market": round(float(d.p_market.mean()), 4),
           "log_loss": {"constant": round(logloss(np.full(len(d), d.y.mean()), d.y), 4),
                        "market": round(logloss(d.p_market, d.y), 4), "model": round(logloss(d.p_model, d.y), 4)}}
    c0, c1, c2 = frozen
    pf = pd.Series(1 / (1 + np.exp(-(c0 + c1 * logit(d.p_market) + c2 * logit(d.p_model)))), index=d.index)
    out["log_loss"]["frozen_blend"] = round(logloss(pf, d.y), 4)
    out["roi_4pt_model_alone"] = rule(d, d.p_model)
    out["roi_4pt_frozen_blend"] = rule(d, pf)
    return out


if __name__ == "__main__":
    d, info = load()
    frozen = json.load(open(os.path.join(ROOT, "docs/live/frozen_rules_v2.json")))["sog"]["blend_coef_const_market_model"]
    res = {"source": "PuckAPI get_prop_history, bought 2026-11-08", "window": ["2023-10-01", "2026-01-15"], **info}
    res["all"] = summarize(d, frozen)
    for s, x in d.groupby("season"):
        res[str(s)] = summarize(x, frozen)
    X = sm.add_constant(np.column_stack([logit(d.p_market), logit(d.p_model)]))
    for grp in ["player_id", "game_id"]:
        fit = sm.GLM(d.y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d[grp]})
        res[f"blend_fit_clustered_by_{grp}"] = {"coef_const_market_model": [round(float(v), 4) for v in fit.params],
                                               "se": [round(float(v), 4) for v in fit.bse]}
    # Reference: the same quantities on the 2026 SportsGameOdds lines.
    ref = pd.read_parquet(os.path.join(ROOT, "data/models/sog_market_graded.parquet"))
    res["reference_sgo_2026"] = {"lines": int(len(ref)), "log_loss_market": round(logloss(ref.p_market, ref.y), 4),
                                 "log_loss_model": round(logloss(ref.p_gbm, ref.y), 4),
                                 "frozen_blend_model_weight": round(frozen[2], 4)}
    json.dump(res, open(os.path.join(ROOT, "docs/models/sog_older_seasons_results.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))
