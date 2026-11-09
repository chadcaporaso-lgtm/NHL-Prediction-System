"""Game model against second-source market prices for earlier seasons (decisions 38-39).

Until 2026-11-09 the game models had market prices for 2025-26 and 2026-27
only. Two owner-supplied sources reach further back (both gitignored, see
docs/audit/external-odds.md):

A. Average closing three-way prices (home win / tie / away win after 60
   minutes), scraped in October 2026 from public odds-comparison sites by the
   dataset's author (data/external/odds_scrape_2026-10/games_nhl.parquet).
   Matched to official games by teams and date (the source dates games in
   European time, so the date may be one day later). De-vigged
   proportionally. Average prices cannot be bet, so only log loss is
   reported, never ROI.
B. Favorite-only moneyline and total line, 2021-22 and 2022-23, of unknown
   timing (data/external/sbr/). Only the favorite's price is given, so the
   margin is removed approximately (4.5 points, a typical two-way NHL
   margin) and only log loss is reported.

Model: the default game model's walk-forward predictions
(data/models/game_predictions.parquet), so every season is out of sample.
Three-way model probabilities come from its regulation score distribution
(Poisson means and the fitted tie adjustment theta).

Tie-rate check (step 2 of the 2026-11-09 plan): the model's tie
probability is compared with the market's and the actual rate, and one
walk-forward alternative is scored: theta refitted each season so the
model's predictions for the PREVIOUS test season match that season's
actual tie rate (instead of the pooled rate of all earlier seasons). A
diagnostic (not a model, it uses the market) replaces the model's tie
probability with the market's to measure how much of the gap is ties.

Output: docs/models/game_market_history_results.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src/models"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import official  # noqa: E402
from game_model import game_probs, joint_scores  # noqa: E402

M = "glm"
XW = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/data/evaluation/external_closing_comparison/20261026T210505Z_bed9e020/external_official_crosswalk.parquet")
SBR_FIX = {"WAS": "WSH", "TB": "TBL", "LA": "LAK", "NJ": "NJD", "SJ": "SJS"}


def ll3(ph, pd_, pa, yh, yd, ya):
    p = np.clip(np.column_stack([ph, pd_, pa]), 1e-6, 1)
    return float(-np.mean(yh * np.log(p[:, 0]) + yd * np.log(p[:, 1]) + ya * np.log(p[:, 2])))


def ll2(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def three_way_model(g: pd.DataFrame, theta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(home reg win), P(tie), P(away reg win) for each game, per-season theta."""
    ph, pt = np.empty(len(g)), np.empty(len(g))
    for th in np.unique(theta):
        i = np.where(theta == th)[0]
        pr = game_probs(g[f"lh_{M}"].values[i], g[f"la_{M}"].values[i], 0.0, th)
        ph[i], pt[i] = pr["ml"], pr["p_tie"]  # q = 0: "ml" is the regulation home win
    return ph, pt, 1 - ph - pt


def match_scrape(gp: pd.DataFrame) -> pd.DataFrame:
    n = pd.read_parquet(os.path.join(ROOT, "data/external/odds_scrape_2026-10/games_nhl.parquet"))
    x = pd.read_parquet(XW)
    ab = official.team_abbrev()
    name = {nm: ab[i] for nm, i in zip(x.home_team, x.official_home_id)}
    name.setdefault("Arizona Coyotes", "ARI")
    n["home"], n["away"], n["d"] = n.home_team.map(name), n.away_team.map(name), pd.to_datetime(n.date)
    out = []
    for lag in (0, 1):  # European-time dates are often one day later
        k = gp.assign(d=gp.d0 + pd.Timedelta(days=lag)).merge(n, on=["d", "home", "away"], suffixes=("", "_src"))
        out.append(k.assign(lag=lag))
    m = pd.concat(out).sort_values("lag").drop_duplicates("game_id")
    return m


def main():
    gp = pd.read_parquet(os.path.join(ROOT, "data/models/game_predictions.parquet"))
    gp.index.name = "game_id"
    gp = gp.reset_index()
    gp["d0"] = pd.to_datetime(gp.game_date.astype(str).str[:10])
    res = {"source_a_three_way": {}, "source_b_favorite_moneyline": {}}

    # ---- A. three-way regulation result ----
    m = match_scrape(gp)
    res["source_a_matching"] = {
        "matched_share_by_season": {str(k): round(float(v), 4) for k, v in (m.groupby("season").size() / gp.groupby("season").size()).items()},
        "date_one_day_later": int((m.lag == 1).sum()), "same_date": int((m.lag == 0).sum()),
        "final_scores_agree": round(float(((m.final_home == m.home_score) & (m.final_away == m.away_score)).mean()), 5)}
    m = m[m[["avg_home_odds", "avg_draw_odds", "avg_away_odds"]].notna().all(1)].copy()
    inv = 1 / m[["avg_home_odds", "avg_draw_odds", "avg_away_odds"]].to_numpy()
    over = inv.sum(1)
    m["mh"], m["md"], m["ma"], m["margin"] = inv[:, 0] / over, inv[:, 1] / over, inv[:, 2] / over, over - 1
    m["yh"] = (m.reg_home > m.reg_away).astype(int)
    m["yd"] = (m.reg_home == m.reg_away).astype(int)
    m["ya"] = 1 - m.yh - m.yd
    m["ph"], m["pt"], m["pa"] = three_way_model(m, m[f"theta_{M}"].values)

    # Walk-forward alternative: theta matched to the previous test season's actual tie rate.
    m = m.sort_values(["season", "game_date"])
    theta_prev = {}
    for s in sorted(gp.season.unique()):
        prev = gp[gp.season == s - 10001]
        if len(prev):
            rate = float((prev.reg_home == prev.reg_away).mean())
            theta_prev[s] = brentq(lambda t: joint_scores(prev[f"lh_{M}"].values, prev[f"la_{M}"].values, t)
                                   .diagonal(axis1=1, axis2=2).sum(1).mean() - rate, -0.5, 3.0)
    m["theta_prev"] = m.season.map(theta_prev).fillna(m[f"theta_{M}"])
    m["ph2"], m["pt2"], m["pa2"] = three_way_model(m, m.theta_prev.values)
    # Diagnostic: model's home/away split with the market's tie probability.
    r = m.ph / (m.ph + m.pa)
    m["ph3"], m["pt3"], m["pa3"] = (1 - m.md) * r, m.md, (1 - m.md) * (1 - r)

    for s, d in m.groupby("season"):
        y = (d.yh, d.yd, d.ya)
        c = d[["yh", "yd", "ya"]].mean().to_numpy()
        res["source_a_three_way"][str(s)] = {
            "games": int(len(d)), "median_margin": round(float(d.margin.median()), 4),
            "log_loss": {"constant": round(ll3(*[np.full(len(d), v) for v in c], *y), 4),
                         "market": round(ll3(d.mh, d.md, d.ma, *y), 4),
                         "model": round(ll3(d.ph, d.pt, d.pa, *y), 4),
                         "model_theta_prev_season": round(ll3(d.ph2, d.pt2, d.pa2, *y), 4),
                         "diagnostic_model_with_market_tie": round(ll3(d.ph3, d.pt3, d.pa3, *y), 4)},
            "tie_rate": {"actual": round(float(d.yd.mean()), 4), "market": round(float(d.md.mean()), 4),
                         "model": round(float(d.pt.mean()), 4), "model_theta_prev_season": round(float(d.pt2.mean()), 4)},
            "theta": {"pooled_earlier_seasons": round(float(d[f"theta_{M}"].iloc[0]), 4),
                      "previous_season": round(float(d.theta_prev.iloc[0]), 4)}}
    full = m[m.season < 20262027]
    y = (full.yh, full.yd, full.ya)
    res["source_a_three_way"]["2021-22_to_2025-26"] = {
        "games": int(len(full)),
        "log_loss": {k: round(ll3(*[full[c] for c in cols], *y), 4) for k, cols in
                     [("market", ["mh", "md", "ma"]), ("model", ["ph", "pt", "pa"]),
                      ("model_theta_prev_season", ["ph2", "pt2", "pa2"]),
                      ("diagnostic_model_with_market_tie", ["ph3", "pt3", "pa3"])]},
        "tie_rate": {"actual": round(float(full.yd.mean()), 4), "market": round(float(full.md.mean()), 4),
                     "model": round(float(full.pt.mean()), 4), "model_theta_prev_season": round(float(full.pt2.mean()), 4)}}
    # Paired standard error of the model-minus-market log-loss gap, by game.
    lm = -(full.yh * np.log(full.mh) + full.yd * np.log(full.md) + full.ya * np.log(full.ma))
    lo = -(full.yh * np.log(full.ph) + full.yd * np.log(full.pt) + full.ya * np.log(full.pa))
    gap = lo - lm
    res["source_a_three_way"]["2021-22_to_2025-26"]["model_minus_market"] = {
        "mean": round(float(gap.mean()), 4), "se": round(float(gap.std(ddof=1) / np.sqrt(len(gap))), 4)}

    # ---- B. favorite-only moneyline, 2021-22 and 2022-23 ----
    gp["date"] = gp.d0.dt.strftime("%Y-%m-%d")
    for f, season in [("sportsbook-nhl-2021-2022.csv", 20212022), ("sportsbook-nhl-2022-2023.csv", 20222023)]:
        d = pd.read_csv(os.path.join(ROOT, "data/external/sbr", f))
        d["home"], d["away"] = d.h__team.replace(SBR_FIX), d.a__team.replace(SBR_FIX)
        k = d.merge(gp, on=["date", "home", "away"])
        a = k.moneyline.abs()
        p_imp = np.where(k.moneyline == 0, 0.5, a / (a + 100))
        y = (k.final_home > k.final_away).astype(int)
        out = {"rows": int(len(d)), "matched": int(len(k)),
               "scores_agree": round(float(((k.h__goals_total == k.final_home) & (k.a__goals_total == k.final_away)).mean()), 4),
               "even_lines": int((k.moneyline == 0).sum()), "log_loss": {"constant": round(ll2(np.full(len(k), y.mean()), y), 4)}}
        for lab, mgn in [("line_no_margin_removed", 0.0), ("line_margin_4.5pts_removed", 0.045)]:
            pf = np.where(k.moneyline == 0, 0.5, p_imp - mgn / 2)
            out["log_loss"][lab] = round(ll2(np.where(k.fav == "Away", 1 - pf, pf), y), 4)
        out["log_loss"]["model_raw"] = round(ll2(k[f"ml_{M}_raw"], y), 4)
        if f"ml_{M}" in k:
            out["log_loss"]["model_calibrated"] = round(ll2(k[f"ml_{M}"], y), 4)
        res["source_b_favorite_moneyline"][str(season)] = out
    json.dump(res, open(os.path.join(ROOT, "docs/models/game_market_history_results.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
