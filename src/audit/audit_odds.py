"""Audit the odds data from both earlier projects.

Sources
  v1  data/normalized/odds.parquet                (SportsGameOdds, one row per book/market/side)
  v2  odds_qualified/<run>/quote_observations     (SportsGameOdds, every quote with timing flags)
  v2  odds_qualified/<run>/closing_benchmarks     (SportsGameOdds dedicated closing pairs)
  v2  external_closing_comparison crosswalk        (user-supplied consensus closing CSV)

Checks
  O1  When v1 prices were captured relative to puck drop.
  O2  Whether v1 "current" prices behave like pregame prices. A pregame
      moneyline price cannot predict the winner much better than the closing
      line (log loss about 0.67 to 0.68 in the NHL). A much lower log loss
      means the price was captured during or after the game.
  O3  Which price v1 used to grade bets (eval_decimal_odds).
  O4  Whether v1 graded every bet correctly and computed profit correctly.
  O5  v2 timing flags on quote observations.
  O6  v2 closing benchmarks: coverage, log loss, margins, duplicates,
      impossible prices.
  O7  External closing CSV: match rate to official games, log loss.

Output: docs/audit/odds_summary.json, printed summary.
Usage: python src/audit/audit_odds.py
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = os.path.join(ROOT, "data/drive/NHL_Prediction_System")
V2 = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT")
ODDS_RUN = "20261019T011635Z_112d14f7"
LABEL_RUN = "20261103T160656Z_4414a3fb"
SUMMARY = os.path.join(ROOT, "docs/audit/odds_summary.json")
out: dict = {}


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# ---------------- v1 ----------------
o = pd.read_parquet(os.path.join(V1, "data/normalized/odds.parquet"))
o["start"] = pd.to_datetime(o.starts_at_utc, utc=True)
o["updated"] = pd.to_datetime(o.last_updated_utc, utc=True, errors="coerce")
o["minutes_after_start"] = (o.updated - o.start).dt.total_seconds() / 60
out["v1_rows"] = len(o)
out["v1_games_by_season"] = o.groupby("season").game_id.nunique().astype(int).to_dict()
out["v1_books"] = sorted(o.book.unique().tolist())
out["v1_share_with_close_price"] = round(float(o.close_decimal_odds.notna().mean()), 3)

ml = o[o.market == "moneyline"].copy()
ml["won"] = (ml.official_outcome == "WIN").astype(int)
out["O1_v1_moneyline_updated_after_start_share"] = round(float((ml.minutes_after_start > 0).mean()), 3)
out["O1_v1_minutes_after_start_median"] = round(float(ml.minutes_after_start.median()), 1)

groups = {
    "current_updated_after_start": ml[ml.updated > ml.start],
    "current_updated_before_start": ml[ml.updated <= ml.start],
    "current_no_timestamp": ml[ml.updated.isna()],
}
out["O2_v1_current_price_log_loss"] = {k: {"rows": len(d), "log_loss": round(log_loss(d.no_vig_prob_proportional, d.won), 4)} for k, d in groups.items()}
c = ml[ml.close_decimal_odds.notna()].copy()
c["p"] = 1 / c.close_decimal_odds
c["p"] = c.p / c.groupby(["game_id", "book"]).p.transform("sum")
out["O2_v1_close_price_log_loss"] = {"rows": len(c), "log_loss": round(log_loss(c.p, c.won), 4)}

out["O3_v1_eval_price_source_share"] = {
    f: round(float(np.isclose(o.eval_decimal_odds, o[f]).mean()), 3)
    for f in ["close_decimal_odds", "current_decimal_odds", "open_decimal_odds"]
}

side = np.where(o.side_id == "home", o.home_final_score, o.away_final_score)
opp = np.where(o.side_id == "home", o.away_final_score, o.home_final_score)
expected = pd.Series("", index=o.index)
m = o.market == "moneyline"
expected[m] = np.where(side[m] > opp[m], "WIN", "LOSS")
p = o.market == "puck_line"
v = side[p] - opp[p] + o.eval_line[p]
expected[p] = np.select([v > 0, v == 0], ["WIN", "PUSH"], "LOSS")
t = o.market == "total_goals"
total = o.home_final_score[t] + o.away_final_score[t]
d = np.where(o.side_id[t] == "over", total - o.eval_line[t], o.eval_line[t] - total)
expected[t] = np.select([d > 0, d == 0], ["WIN", "PUSH"], "LOSS")
pnl = np.select([o.official_outcome == "WIN", o.official_outcome == "LOSS"], [o.eval_decimal_odds - 1, -1.0], 0.0)
out["O4_v1_grading_errors"] = int((expected != o.official_outcome).sum())
out["O4_v1_profit_errors"] = int((~np.isclose(pnl, o.unit_pnl)).sum())

# ---------------- v2 quotes ----------------
qdir = os.path.join(V2, f"data/normalized/odds_qualified/{ODDS_RUN}")
q = pd.read_parquet(os.path.join(qdir, "quote_observations.parquet"),
                    columns=["quote_kind", "event_started_at_snapshot", "historical_asof_eligible", "roi_eligible", "event_join_status"])
out["O5_v2_quotes"] = len(q)
out["O5_v2_quotes_snapshot_after_start_share"] = round(float(q.event_started_at_snapshot.mean()), 4)
out["O5_v2_quotes_asof_eligible"] = int(q.historical_asof_eligible.sum())
out["O5_v2_quotes_roi_eligible"] = int(q.roi_eligible.sum())
out["O5_v2_join_status"] = q.event_join_status.value_counts().astype(int).to_dict()

# ---------------- v2 closing benchmarks ----------------
cb = pd.read_parquet(os.path.join(qdir, "closing_benchmarks.parquet"))
sched = pd.read_parquet(glob.glob(os.path.join(V2, f"data/normalized/official_incremental/{LABEL_RUN}/games_schedule.parquet"))[0])[["game_id", "game_date"]]
labels = pd.read_parquet(glob.glob(os.path.join(V2, f"data/features/*/{LABEL_RUN}/game_labels.parquet"))[0])[["game_id", "label_home_final_win"]]
cb = cb.merge(sched, on="game_id", how="left")
out["O6_v2_closing_rows"] = len(cb)
out["O6_v2_closing_games"] = int(cb.game_id.nunique())
out["O6_v2_closing_books"] = int(cb.bookmaker.nunique())
out["O6_v2_closing_date_range"] = [cb.game_date.min(), cb.game_date.max()]
out["O6_v2_closing_games_by_season"] = cb.groupby("season").game_id.nunique().astype(int).to_dict()
out["O6_v2_closing_duplicates"] = int(cb.duplicated(["game_id", "bookmaker", "market", "canonical_pair_line"]).sum())
out["O6_v2_closing_impossible_prices"] = int(((cb.first_decimal_odds <= 1) | (cb.second_decimal_odds <= 1)).sum())
out["O6_v2_closing_margin_quantiles"] = cb.implied_probability_sum.quantile([0, 0.01, 0.5, 0.99, 1]).round(4).tolist()
out["O6_v2_closing_margin_below_1"] = cb[cb.implied_probability_sum < 1].bookmaker.value_counts().astype(int).to_dict()
out["O6_v2_closing_margin_above_1_15"] = cb[cb.implied_probability_sum > 1.15].bookmaker.value_counts().astype(int).to_dict()
mlc = cb[cb.market == "moneyline"].merge(labels, on="game_id")
home_p = np.where(mlc.first_side == "home", mlc.first_side_proportional_devig_probability, 1 - mlc.first_side_proportional_devig_probability)
out["O6_v2_closing_moneyline_log_loss"] = {"rows": len(mlc), "games": int(mlc.game_id.nunique()), "log_loss": round(log_loss(home_p, mlc.label_home_final_win.astype(int)), 4)}

# ---------------- external closing CSV ----------------
xw = pd.read_parquet(glob.glob(os.path.join(V2, "data/evaluation/external_closing_comparison/*/external_official_crosswalk.parquet"))[0])
ok = xw[xw.official_record_checks_passed == True]  # noqa: E712
out["O7_external_rows"] = len(xw)
out["O7_external_matched"] = len(ok)
out["O7_external_games_by_season_type"] = {f"{s}-{g}": int(n) for (s, g), n in ok.groupby(["season", "game_type"]).size().items()}
out["O7_external_close_log_loss"] = round(log_loss(ok.close_home_win_prob, ok.official_home_win.astype(int)), 4)
out["O7_external_open_log_loss"] = round(log_loss(ok.open_home_win_prob, ok.official_home_win.astype(int)), 4)
out["O7_external_methodology_verified"] = bool(xw.source_methodology_verified.any())

os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
json.dump(out, open(SUMMARY, "w"), indent=2, default=str)
print(json.dumps(out, indent=2, default=str))
