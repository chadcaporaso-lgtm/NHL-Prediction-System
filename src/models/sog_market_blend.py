"""Does the shots-on-goal model add information beyond the closing price?

Fits logistic regression y ~ logit(p_market) + logit(p_model) on 2025-26
prop lines and tests it on 2026-27 lines (walk-forward: the blend never sees
2026-27). If the model coefficient is clearly above zero and the blend beats
the market on 2026-27, the model carries information the market has not
priced. Also reports a reliability table for the model's over probability.

Input:  data/models/sog_market_graded.parquet (sog_model.py)
Output: docs/models/sog_blend_results.json
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def payout(a):
    a = np.asarray(a, dtype=float)
    return np.where(a < 0, 100 / -a, a / 100)


d = pd.read_parquet(os.path.join(ROOT, "data/models/sog_market_graded.parquet"))
tr, te = d[d.season == 20252026], d[d.season == 20262027]
out = {"train_lines_2025_26": len(tr), "test_lines_2026_27": len(te), "models": {}}
for m in ["glm", "gbm"]:
    X = lambda x: sm.add_constant(np.column_stack([logit(x.p_market), logit(x[f"p_{m}"])]))
    fit = sm.GLM(tr.y, X(tr), family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": tr.game_id})
    p = fit.predict(X(te))
    edge = p - te.p_market
    res = {"coef_const_market_model": [round(float(c), 4) for c in fit.params],
           "se_game_clustered": [round(float(c), 4) for c in fit.bse],
           "test_log_loss_market": round(logloss(te.p_market, te.y), 4),
           "test_log_loss_model": round(logloss(te[f"p_{m}"], te.y), 4),
           "test_log_loss_blend": round(logloss(p, te.y), 4), "bets": {}}
    for thr in [0.02, 0.04, 0.06]:
        over, under = edge > thr, edge < -thr
        bet = over | under
        win = np.where(over, te.y == 1, te.y == 0)
        prof = np.where(win, np.where(over, payout(te.close_odds_over), payout(te.close_odds_under)), -1.0)[bet]
        g = pd.Series(prof).groupby(te.game_id.values[bet]).sum()
        res["bets"][str(thr)] = {"n": int(bet.sum()), "roi": round(float(prof.mean()), 4) if bet.any() else None,
                                 "se": round(float(g.std(ddof=1) * np.sqrt(len(g)) / bet.sum()), 4) if bet.sum() > 1 else None}
    out["models"][m] = res

# Reliability of the gbm over probability, all lines.
d["bin"] = pd.cut(d.p_gbm, [0, .3, .4, .45, .5, .55, .6, .7, 1])
out["gbm_reliability"] = [{"bin": str(k), "n": int(len(g)), "mean_p": round(float(g.p_gbm.mean()), 3),
                           "actual": round(float(g.y.mean()), 3), "market_p": round(float(g.p_market.mean()), 3)}
                          for k, g in d.groupby("bin", observed=True)]
json.dump(out, open(os.path.join(ROOT, "docs/models/sog_blend_results.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
