"""Closing-line value of the shots-on-goal model's picks at the opening price.

If a model finds real edges, the market should move toward its picks between
the opening and the closing price, and bets taken at the open should beat the
close. This does not depend on game results, so it is a sharper test than
ROI on a few hundred bets.

Data: SportsGameOdds consensus opening and closing prices for every graded
shots-on-goal line from 2026-01-16 (data/models/sog_market_graded.parquet,
built by prop_model.py; decision 15). Prices are de-vigged proportionally.

Bets at the open
  model     gradient-boosted negative binomial (trained on earlier seasons
            only, so every line is out of sample); P(over the opening line)
            with that season's dispersion; whole-number lines with pushes
            removed, as in prop_model.py
  rules     A  model alone: |p_model - p_open| > threshold, both seasons
            B  blend: y ~ logit(p_open) + logit(p_model) fitted on 2025-26
               opening prices, bets on 2026-27 only (walk-forward)
  side      the one the rule favors, at the opening price

Closing-line value of a bet
  same line (opening line = closing line):
     clv = p_close_fair(side) - 1 / decimal(open price of side)
           (expected profit per unit staked if the close is the true price,
           divided by the decimal price; > 0 means the bet beat the close)
     move = p_close_fair(side) - p_open_fair(side)  (did the market move our way)
  line moved: whether it moved our way (over bet and line went up, under bet
     and line went down: the opening bet is then strictly better than the close)

Information test (no betting rule): on every same-line line, regress the
market move (p_close_fair - p_open_fair, over side) on the model's opening
edge (p_model - p_open_fair). A positive slope means the market moves toward
the model.

Controls: a model-free predictor (the player's exponentially weighted shots,
`mu_baseline` in prop_model.py) through the same rules and information test,
and "bet every under" / "bet every over" at the open. Results are also split
into same-line and moved-line bets, because a line move hands the opening
bet a better line than the close whatever the model says.

Standard errors are clustered by game. Results: docs/models/sog_clv_results.json.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THRESHOLDS = [0.02, 0.04, 0.06]


def implied(a):
    a = np.asarray(a, dtype=float)
    return np.where(a < 0, -a / (100 - np.minimum(a, 0)), 100 / (np.maximum(a, 0) + 100))


def payout(a):
    a = np.asarray(a, dtype=float)
    return np.where(a < 0, 100 / -a, a / 100)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def p_over(line, mu, alpha):
    r = 1.0 / alpha
    p = r / (r + mu)
    k = np.floor(line)
    over = stats.nbinom.sf(k, r, p)
    push = np.where(line == k, stats.nbinom.pmf(k, r, p), 0.0)
    return over / (1 - push)


def clustered_mean(x, g):
    """Mean of x and its standard error clustered by g."""
    x, g = np.asarray(x, float), np.asarray(g)
    n = len(x)
    if n < 2:
        return float(x.mean()) if n else None, None
    m = x.mean()
    s = pd.Series(x - m).groupby(g).sum()
    return float(m), float(np.sqrt((s ** 2).sum()) / n)


def load() -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(ROOT, "data/models/sog_market_graded.parquet"))
    alpha = json.load(open(os.path.join(ROOT, "docs/models/sog_results.json")))["alpha"]
    d["alpha"] = d.season.map(lambda s: alpha[str(s)]["gbm"])
    d["open_line"] = d.open_line.astype(float)
    po, pu = implied(d.open_odds_over), implied(d.open_odds_under)
    d["p_open"] = po / (po + pu)
    d["open_margin"] = po + pu - 1
    co, cu = implied(d.close_odds_over), implied(d.close_odds_under)
    d["p_close"] = co / (co + cu)  # fair P(over the closing line); = p_market
    d["p_model_open"] = p_over(d.open_line.values, d.mu_gbm.values, d.alpha.values)
    a_base = d.season.map(lambda s: alpha[str(s)]["baseline"])
    d["p_baseline_open"] = p_over(d.open_line.values, d.mu_baseline.values, a_base.values)
    d["same_line"] = d.open_line == d.line
    # Grade at the opening line; pushes there are void.
    d["push_open"] = d.actual == d.open_line
    d["y_open"] = (d.actual > d.open_line).astype(int)
    return d


def bets(d: pd.DataFrame, p_rule: pd.Series, thr: float) -> pd.DataFrame:
    edge = p_rule - d.p_open
    b = d[edge.abs() > thr].copy()
    b["edge"] = edge[b.index]
    b["side"] = np.where(b.edge > 0, "over", "under")
    over = b.side == "over"
    b["open_price"] = np.where(over, b.open_odds_over, b.open_odds_under)
    b["close_price"] = np.where(over, b.close_odds_over, b.close_odds_under)
    b["p_open_side"] = np.where(over, b.p_open, 1 - b.p_open)
    b["p_close_side"] = np.where(over, b.p_close, 1 - b.p_close)
    b["breakeven_open"] = implied(b.open_price)
    b["clv"] = b.p_close_side - b.breakeven_open
    b["move"] = b.p_close_side - b.p_open_side
    b["line_moved_our_way"] = np.where(over, b.line > b.open_line, b.line < b.open_line)
    b["line_moved_against"] = np.where(over, b.line < b.open_line, b.line > b.open_line)
    win = np.where(over, b.y_open == 1, b.y_open == 0)
    b["profit"] = np.where(b.push_open, 0.0, np.where(win, payout(b.open_price), -1.0))
    return b


def summarize(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"bets": 0}
    s = b[b.same_line]
    clv, clv_se = clustered_mean(s.clv, s.game_id)
    mv, mv_se = clustered_mean(s.move, s.game_id)
    beat = np.where(b.same_line, b.clv > 0, b.line_moved_our_way)
    br, br_se = clustered_mean(beat.astype(float), b.game_id)
    graded = b[~b.push_open]
    roi, roi_se = clustered_mean(graded.profit, graded.game_id)
    moved = b[~b.same_line]
    return {
        "bets": int(len(b)), "games": int(b.game_id.nunique()), "over_share": round(float((b.side == "over").mean()), 3),
        "mean_abs_edge": round(float(b.edge.abs().mean()), 4),
        "same_line_bets": int(len(s)),
        "clv_mean": round(clv, 4), "clv_se": round(clv_se, 4),
        "market_move_toward_bet_mean": round(mv, 4), "market_move_se": round(mv_se, 4),
        "line_moved_bets": int(len(moved)),
        "line_moved_our_way": int(moved.line_moved_our_way.sum()), "line_moved_against": int(moved.line_moved_against.sum()),
        "beat_close_rate": round(br, 4), "beat_close_se": round(br_se, 4),
        "roi_at_open": round(roi, 4), "roi_at_open_se": round(roi_se, 4), "graded_bets": int(len(graded)),
    }


def information_test(d: pd.DataFrame, cols=("p_model_open",)) -> dict:
    s = d[d.same_line]
    X = sm.add_constant(np.column_stack([(s[c] - s.p_open).values for c in cols]))
    fit = sm.OLS((s.p_close - s.p_open).values, X).fit(cov_type="cluster", cov_kwds={"groups": s.game_id})
    out = {"lines": int(len(s)), "r2": round(float(fit.rsquared), 4), "intercept": round(float(fit.params[0]), 5)}
    for i, c in enumerate(cols):
        out[f"slope_{c}"], out[f"se_{c}"] = round(float(fit.params[i + 1]), 4), round(float(fit.bse[i + 1]), 4)
    return out


def split(b: pd.DataFrame) -> dict:
    """ROI at the open and at the close, same-line and moved-line bets apart."""
    over = b.side == "over"
    win_close = np.where(over, b.y == 1, b.y == 0)
    b = b.assign(profit_close=np.where(win_close, payout(b.close_price), -1.0))
    out = {}
    for name, x in [("same_line", b[b.same_line]), ("line_moved", b[~b.same_line])]:
        g = x[~x.push_open]
        win = np.where(g.side == "over", g.y_open == 1, g.y_open == 0)
        r, se = clustered_mean(g.profit, g.game_id)
        rc, sec = clustered_mean(x.profit_close, x.game_id)
        out[name] = {"bets": int(len(x)), "win_rate_at_open_line": round(float(win.mean()), 4),
                     "p_close_fair_side": round(float(x.p_close_side.mean()), 4),
                     "breakeven_open_price": round(float(x.breakeven_open.mean()), 4),
                     "roi_at_open": round(r, 4), "roi_at_open_se": round(se, 4),
                     "roi_at_close": round(rc, 4), "roi_at_close_se": round(sec, 4)}
    return out


if __name__ == "__main__":
    d = load()
    out = {"lines": int(len(d)), "games": int(d.game_id.nunique()),
           "dates": [str(d.game_date.min())[:10], str(d.game_date.max())[:10]],
           "same_line_share": round(float(d.same_line.mean()), 4),
           "median_open_margin": round(float(d.open_margin.median()), 4),
           "median_close_margin": round(float(d.margin.median()), 4)}

    # Control: every line, both sides. Without skill, CLV is about minus half the margin.
    ctrl = pd.concat([bets(d, d.p_open + 1.0, 0.0), bets(d, d.p_open - 1.0, 0.0)])
    out["control_every_line_both_sides"] = summarize(ctrl)

    out["control_every_under"] = split(bets(d, d.p_open - 1.0, 0.0))
    out["control_every_over"] = split(bets(d, d.p_open + 1.0, 0.0))

    out["information_test"] = {"model": information_test(d), "baseline": information_test(d, ("p_baseline_open",)),
                               "model_and_baseline": information_test(d, ("p_model_open", "p_baseline_open"))}
    for s, x in d.groupby("season"):
        out["information_test"][f"model_{s}"] = information_test(x)

    out["model_alone"], out["baseline_alone"] = {}, {}
    for thr in THRESHOLDS:
        b = bets(d, d.p_model_open, thr)
        out["model_alone"][str(thr)] = ({"all": summarize(b), "split": split(b)}
                                        | {str(s): summarize(x) for s, x in b.groupby("season")})
        b = bets(d, d.p_baseline_open, thr)
        out["baseline_alone"][str(thr)] = {"all": summarize(b), "split": split(b)}

    # Walk-forward blend on opening prices: fit on 2025-26, bet on 2026-27.
    tr, te = d[d.season == 20252026], d[d.season == 20262027]
    X = lambda x: sm.add_constant(np.column_stack([logit(x.p_open), logit(x.p_model_open)]))
    fit = sm.GLM(tr.y_open[~tr.push_open], X(tr[~tr.push_open]), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": tr.game_id[~tr.push_open]})
    p_blend = pd.Series(fit.predict(X(te)), index=te.index)
    out["blend_walk_forward"] = {"fit_lines_2025_26": int((~tr.push_open).sum()),
                                 "coef_const_market_model": [round(float(c), 4) for c in fit.params],
                                 "se_game_clustered": [round(float(c), 4) for c in fit.bse]}
    for thr in THRESHOLDS:
        b = bets(te, p_blend, thr)
        out["blend_walk_forward"][str(thr)] = {"all": summarize(b), "split": split(b)}

    os.makedirs(os.path.join(ROOT, "docs/models"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "docs/models/sog_clv_results.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))
