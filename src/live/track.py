"""Grade the frozen live-tracking rules (docs/live/frozen_rules.json) on
games after the cutoff.

Run after the weekly rebuild (see docs/live/README.md). Checks the model
fingerprints first and stops if a retrained model differs from the frozen one.

Outputs
  docs/live/tracking.json   cumulative and weekly results (committed)
  data/live/ledger_sog.parquet, data/live/ledger_puck_line.parquet  every bet
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RULES = json.load(open(os.path.join(ROOT, "docs/live/frozen_rules.json")))
CUTOFF = RULES["window_starts_after"]


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def payout(a):
    a = np.asarray(a, dtype=float)
    return np.where(a < 0, 100 / -a, a / 100)


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else None


def check_fingerprints():
    sog = pd.read_parquet(os.path.join(ROOT, "data/models/sog_predictions.parquet"))
    sog["game_date"] = pd.to_datetime(sog.game_date).dt.strftime("%Y-%m-%d")
    fp = sog[(sog.season == 20262027) & (sog.game_date <= CUTOFF)]
    alpha = json.load(open(os.path.join(ROOT, "docs/models/sog_results.json")))["alpha"]["20262027"]["gbm"]
    f = RULES["sog"]["fingerprint"]
    assert len(fp) == f["rows_to_cutoff"], ("sog rows changed", len(fp))
    assert np.isclose(fp.mu_gbm.mean(), f["mean_mu_gbm_to_cutoff"], atol=1e-7), ("sog model changed", fp.mu_gbm.mean())
    assert np.isclose(alpha, f["alpha_20262027_gbm"]), ("sog alpha changed", alpha)
    g = pd.read_parquet(os.path.join(ROOT, "data/models/game_predictions.parquet"))
    gp = g[(g.season == 20262027) & (g.game_date <= CUTOFF)]
    f = RULES["puck_line"]["fingerprint"]
    assert len(gp) == f["games_to_cutoff"], ("game rows changed", len(gp))
    assert np.isclose(gp.lh_glm.mean(), f["mean_lh_glm_to_cutoff"], atol=1e-7), ("game model changed", gp.lh_glm.mean())


def summarize(b: pd.DataFrame) -> dict:
    if b.empty:
        return {"bets": 0}
    by_game = b.groupby("game_id").profit.sum()
    n = len(b)
    return {"bets": n, "games": int(b.game_id.nunique()), "wins": int(b.win.sum()),
            "units": round(float(b.profit.sum()), 2), "roi": round(float(b.profit.mean()), 4),
            "roi_se_game_clustered": round(float(by_game.std(ddof=1) * np.sqrt(len(by_game)) / n), 4) if len(by_game) > 1 else None,
            "mean_edge": round(float(b.edge.abs().mean()), 4)}


def sog_bets():
    r = RULES["sog"]
    d = pd.read_parquet(os.path.join(ROOT, "data/models/sog_market_graded.parquet"))
    d["game_date"] = pd.to_datetime(d.game_date).dt.strftime("%Y-%m-%d")
    d = d[d.game_date > CUTOFF].copy()
    c0, c1, c2 = r["blend_coef_const_market_model"]
    d["p_blend"] = 1 / (1 + np.exp(-(c0 + c1 * logit(d.p_market) + c2 * logit(d.p_gbm))))
    d["edge"] = d.p_blend - d.p_market
    over, under = d.edge > r["threshold"], d.edge < -r["threshold"]
    lines = {"lines": int(len(d)), "games": int(d.game_id.nunique()),
             "log_loss_market": logloss(d.p_market, d.y), "log_loss_blend": logloss(d.p_blend, d.y)}
    b = d[over | under].copy()
    b["side"] = np.where(b.edge > 0, "over", "under")
    b["price"] = np.where(b.side == "over", b.close_odds_over, b.close_odds_under)
    b["win"] = np.where(b.side == "over", b.y == 1, b.y == 0)
    b["profit"] = np.where(b.win, payout(b.price), -1.0)
    return b[["game_id", "game_date", "player_id", "player_name", "line", "side", "price", "p_market", "p_blend", "edge",
              "actual", "win", "profit"]], lines


def puck_line_bets():
    r = RULES["puck_line"]
    d = pd.read_parquet(os.path.join(ROOT, "data/models/game_market_graded.parquet"))
    d["game_date"] = pd.to_datetime(d.game_date).dt.strftime("%Y-%m-%d")
    d = d[(d.source == "sgo") & (d.market == "pl") & (d.game_date > CUTOFF)].copy()
    d["edge"] = d.p_cal - d.p_mkt
    lines = {"games": int(len(d)), "log_loss_market": logloss(d.p_mkt, d.y), "log_loss_model": logloss(d.p_cal, d.y)}
    a, b_ = d.edge > r["threshold"], d.edge < -r["threshold"]
    b = d[a | b_].copy()
    b["side"] = np.where(b.edge > 0, "home", "away")
    b["price"] = np.where(b.side == "home", b.price_a, b.price_b)
    b["win"] = np.where(b.side == "home", b.y == 1, b.y == 0)
    b["profit"] = np.where(b.win, payout(b.price), -1.0)
    return b[["game_id", "game_date", "line", "side", "price", "p_mkt", "p_cal", "edge", "final_home", "final_away",
              "win", "profit"]], lines


if __name__ == "__main__":
    check_fingerprints()
    out = {"window_starts_after": CUTOFF, "rules": "docs/live/frozen_rules.json"}
    for name, fn in [("sog", sog_bets), ("puck_line", puck_line_bets)]:
        b, lines = fn()
        b.to_parquet(os.path.join(ROOT, f"data/live/ledger_{name}.parquet"), index=False)
        week = pd.to_datetime(b.game_date).dt.to_period("W-SUN").astype(str) if len(b) else pd.Series(dtype=str)
        out[name] = {"all_lines": lines, "cumulative": summarize(b),
                     "by_week": {w: summarize(x) for w, x in b.groupby(week)} if len(b) else {}}
        out[name]["last_graded_game_date"] = b.game_date.max() if len(b) else None
    json.dump(out, open(os.path.join(ROOT, "docs/live/tracking.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
