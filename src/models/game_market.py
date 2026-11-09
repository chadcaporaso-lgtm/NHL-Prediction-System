"""Compare the game model with closing lines: moneyline, puck line, totals.

Market sources (both pregame consensus closes, de-vigged proportionally)
  external  external closing CSV, 2025-26 and early 2026-27 (benchmark only;
            its methodology is undocumented, so no ROI is computed from it)
  sgo       SportsGameOdds consensus close from 2026-01-16 (prices used for ROI)

Model probabilities at the market's line
  moneyline  P(home win), calibrated
  puck line  home -1.5: P(home wins by 2+); home +1.5: 1 - P(away wins by 2+); calibrated
  totals     P(official total > L) with pushes removed; calibrated for 5.5 and
             6.5 lines, raw for others (whole-number lines are few)
Blend test: logistic regression y ~ logit(market) + logit(model) fitted on
2025-26 external closes, tested on 2026-27 (external and sgo).

Usage:  python3 src/models/game_market.py [elo]
Input:  data/models/game_predictions[_elo].parquet (game_model.py)
Output: docs/models/game_market_results.json, data/models/game_market_graded.parquet
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src/models"))
from game_model import joint_scores, logit, logloss  # noqa: E402

# External closes: the crosswalk rebuilt with the newest 2026-27 file (src/ingest/build_external_crosswalk.py,
# decisions 37, 40) when present, else v2's crosswalk of 2026-10-26. Earlier results are kept:
# game_market_results*_xw2026-10-26.json (v2's crosswalk) and *_xw2026-11-01.json (231-game file).
XW_V2 = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/data/evaluation/external_closing_comparison/20261026T210505Z_bed9e020/external_official_crosswalk.parquet")
XW_NEW = os.path.join(ROOT, "data/external/external_official_crosswalk.parquet")
XW = XW_NEW if os.path.exists(XW_NEW) else XW_V2
M = "glm"
SUFFIX = f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""


def american_prob(a):
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))


def payout(a):
    a = np.asarray(a, dtype=float)
    return np.where(a < 0, 100 / -a, a / 100)


def model_total(g, line):
    """Raw P(official total > line | not a push), and the calibrated value where available."""
    k = np.arange(16)
    joint = joint_scores(g[f"lh_{M}"].values, g[f"la_{M}"].values, g[f"theta_{M}"].values[0]) if g[f"theta_{M}"].nunique() == 1 else None
    out = np.empty(len(g))
    for th in g[f"theta_{M}"].unique():
        idx = np.where(g[f"theta_{M}"].values == th)[0]
        joint = joint_scores(g[f"lh_{M}"].values[idx], g[f"la_{M}"].values[idx], th)
        diff = k[:, None] - k[None, :]
        tot = k[:, None] + k[None, :] + (diff == 0)
        L = line[idx][:, None, None]
        over = (joint * (tot[None] > L)).sum((1, 2))
        push = (joint * (tot[None] == L)).sum((1, 2))
        out[idx] = over / (1 - push)
    cal = np.where(line == 5.5, g[f"over55_{M}"], np.where(line == 6.5, g[f"over65_{M}"], out))
    return out, cal


def market_frame() -> pd.DataFrame:
    g = pd.read_parquet(os.path.join(ROOT, f"data/models/game_predictions{SUFFIX}.parquet"))
    g.index.name = "game_id"
    g = g.reset_index()
    rows = []
    x = pd.read_parquet(XW)
    x = x[x.official_game_id.notna()].drop(columns=["season", "game_type"]).merge(g, left_on="official_game_id", right_on="game_id")
    for mk in ["ml", "pl", "tot"]:
        d = x.copy()
        d["source"] = "external"
        d["market"] = mk
        if mk == "ml":
            d["p_mkt"], d["line"] = d.close_home_win_prob, np.nan
        elif mk == "pl":
            d["p_mkt"], d["line"] = d.close_home_cover_prob, d.close_spread_home
        else:
            d["p_mkt"], d["line"] = d.close_over_prob, d.close_total
        rows.append(d)
    s = pd.read_parquet(os.path.join(ROOT, "data/sgo/game_odds_matched.parquet"))
    s = s[["game_id"] + [c for c in s.columns if c.startswith(("ml_", "pl_", "tot_"))]].merge(g, on="game_id")
    for mk, a, b, line in [("ml", "ml_home", "ml_away", None), ("pl", "pl_home", "pl_away", "pl_home_close_line"),
                           ("tot", "tot_over", "tot_under", "tot_over_close_line")]:
        d = s.dropna(subset=[f"{a}_close", f"{b}_close"]).copy()
        pa, pb = american_prob(d[f"{a}_close"]), american_prob(d[f"{b}_close"])
        d["p_mkt"], d["price_a"], d["price_b"] = pa / (pa + pb), d[f"{a}_close"], d[f"{b}_close"]
        d["line"] = np.nan if line is None else d[line]
        d["source"], d["market"] = "sgo", mk
        rows.append(d)
    keep = ["game_id", "season", "game_date", "source", "market", "line", "p_mkt", "price_a", "price_b",
            "final_home", "final_away", f"lh_{M}", f"la_{M}", f"theta_{M}", f"ml_{M}", f"home_by2_{M}", f"away_by2_{M}",
            f"over55_{M}", f"over65_{M}", f"ml_{M}_raw", f"home_by2_{M}_raw", f"away_by2_{M}_raw"]
    d = pd.concat([r.reindex(columns=keep) for r in rows], ignore_index=True)
    d = d[d.p_mkt.between(0.01, 0.99)]
    margin = d.final_home - d.final_away
    total = d.final_home + d.final_away
    # outcome and model probability for side A (home, or over)
    d["push"] = (d.market == "tot") & (total == d.line)
    d["y"] = np.select([d.market == "ml", d.market == "pl"], [margin > 0, margin + d.line > 0], total > d.line).astype(int)
    d["p_raw"] = np.select([d.market == "ml", (d.market == "pl") & (d.line < 0)],
                           [d[f"ml_{M}_raw"], d[f"home_by2_{M}_raw"]], 1 - d[f"away_by2_{M}_raw"])
    d["p_cal"] = np.select([d.market == "ml", (d.market == "pl") & (d.line < 0)],
                           [d[f"ml_{M}"], d[f"home_by2_{M}"]], 1 - d[f"away_by2_{M}"])
    t = d.market == "tot"
    raw, cal = model_total(d[t], d.loc[t, "line"].values)
    d.loc[t, "p_raw"], d.loc[t, "p_cal"] = raw, cal
    d = d[~d.push & ~(d.market == "pl") | (d.market == "pl") & d.line.abs().eq(1.5)]
    return d[~d.push]


def roi(d, p, thr):
    edge = p - d.p_mkt
    a, b = edge > thr, edge < -thr
    bet = a | b
    win = np.where(a, d.y == 1, d.y == 0)
    prof = np.where(win, np.where(a, payout(d.price_a), payout(d.price_b)), -1.0)[bet]
    g = pd.Series(prof).groupby(d.game_id.values[bet]).sum()
    n = int(bet.sum())
    return {"bets": n, "roi": round(float(prof.mean()), 4) if n else None,
            "se": round(float(g.std(ddof=1) * np.sqrt(len(g)) / n), 4) if n > 2 else None}


if __name__ == "__main__":
    d = market_frame()
    d.to_parquet(os.path.join(ROOT, f"data/models/game_market_graded{SUFFIX}.parquet"), index=False)
    res = {"log_loss": {}, "blend": {}, "roi_model_alone_sgo": {}}
    for (src, mk, s), x in d.groupby(["source", "market", "season"]):
        res["log_loss"][f"{src}|{mk}|{s}"] = {"games": len(x), "market": round(logloss(x.p_mkt, x.y), 4),
                                              "model_raw": round(logloss(x.p_raw, x.y), 4),
                                              "model_cal": round(logloss(x.p_cal, x.y), 4),
                                              "constant": round(logloss(np.full(len(x), x.y.mean()), x.y), 4)}
    for mk in ["ml", "pl", "tot"]:
        tr = d[(d.source == "external") & (d.market == mk) & (d.season == 20252026)]
        X = lambda x: sm.add_constant(np.column_stack([logit(x.p_mkt), logit(x.p_cal)]), has_constant="add")
        fit = sm.GLM(tr.y, X(tr), family=sm.families.Binomial()).fit()
        r = {"train_games": len(tr), "coef_const_market_model": [round(float(v), 4) for v in fit.params],
             "se": [round(float(v), 4) for v in fit.bse]}
        for src in ["external", "sgo"]:
            te = d[(d.source == src) & (d.market == mk) & (d.season == 20262027)]
            if len(te) == 0:
                continue
            p = fit.predict(X(te))
            r[f"test_{src}"] = {"games": len(te), "market": round(logloss(te.p_mkt, te.y), 4),
                                "model": round(logloss(te.p_cal, te.y), 4), "blend": round(logloss(p, te.y), 4)}
            if src == "sgo":
                r["test_sgo"]["roi_blend"] = {str(t): roi(te, p, t) for t in [0.02, 0.04]}
        res["blend"][mk] = r
        sg = d[(d.source == "sgo") & (d.market == mk)]
        res["roi_model_alone_sgo"][mk] = {str(t): roi(sg, sg.p_cal, t) for t in [0.02, 0.04, 0.06]}
    json.dump(res, open(os.path.join(ROOT, f"docs/models/game_market_results{SUFFIX}.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))
