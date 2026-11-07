"""Recompute the backtest results reported by the earlier projects.

v1: reports/stage8_verified_pregame_ledger.parquet holds every verified
    pregame closing quote (9 books, moneyline and totals, 728 games from
    2026-01-17 to 2026-10-13) with the v1 model probability and EV.
v2: reports/stage27_walk_forward/<run>/probability_metrics.csv holds the
    monthly-refit walk-forward scores for 2025-26 and 2026-27.

Prints model-vs-market log loss and betting ROI under several selection rules.
Usage: python src/audit/review_backtests.py
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L = pd.read_parquet(os.path.join(ROOT, "data/drive/NHL_Prediction_System/reports/stage8_verified_pregame_ledger.parquet"))
v = L[L.is_verified_pregame_pair].copy()
assert np.isclose(v.pregame_dec_odds, v.close_decimal_odds).all(), "ledger prices are not the closing prices"
v["pnl"] = np.where(v.official_outcome == "WIN", v.pregame_dec_odds - 1, np.where(v.official_outcome == "PUSH", 0.0, -1.0))


def report(name, b):
    g = b.groupby("game_id").pnl.sum()
    se = g.std(ddof=1) * np.sqrt(len(g)) / len(b) * 100
    print(f"{name:62} bets {len(b):6}  ROI {b.pnl.mean() * 100:6.2f}%  game-clustered SE {se:5.2f}")


h = v[(v.market == "moneyline") & (v.side_id == "home") & (v.book == "pinnacle")]
y = (h.official_outcome == "WIN").astype(int)
ll = lambda p: float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())  # noqa: E731
print(f"v1 moneyline, {len(h)} games: model log loss {ll(h.pW_stats.clip(1e-6, 1 - 1e-6)):.4f}, Pinnacle close {ll(h.pin_close_no_vig.clip(1e-6, 1 - 1e-6)):.4f}")

for thr in [0, 0.02, 0.05]:
    report(f"v1 every book, ev_stats_only > {thr}", v[v.ev_stats_only > thr])
    report(f"v1 every book, ev_odds_informed > {thr}", v[v.ev_odds_informed > thr])
best = v.sort_values("pregame_dec_odds").groupby(["game_id", "market", "side_id", "pregame_line"]).tail(1)
best = best.assign(ev_best=best.pW_stats * best.pregame_dec_odds - 1)
for thr in [0, 0.02, 0.05]:
    b = best[best.ev_best > thr].sort_values("ev_best").groupby(["game_id", "market"]).tail(1)
    report(f"v1 best price, one bet per game and market, EV > {thr}", b)
    report(f"   of which moneyline", b[b.market == "moneyline"])
    report(f"   of which totals", b[b.market == "total_goals"])

wf = pd.read_csv(glob.glob(os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/reports/stage27_walk_forward/*/probability_metrics.csv"))[0])
k = wf[wf.cohort == "all_eligible_regular"]
for s, d in k.groupby("season"):
    models = d[~d.model.str.startswith(("external", "constant", "monthly"))]
    best_m = models.sort_values("log_loss").iloc[0]
    flag = models[models.primary_research_challenger]
    print(f"v2 walk-forward {s}: {len(models)} variants, games {int(d.games.iloc[0])}")
    print(f"   best variant {best_m.model}: {best_m.log_loss:.4f}; pre-declared challenger: {flag.log_loss.iloc[0]:.4f}")
    for m in ["external_csv_close", "external_csv_open", "constant_parent_fit_prior"]:
        print(f"   {m}: {d[d.model == m].log_loss.iloc[0]:.4f}")
