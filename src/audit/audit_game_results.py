"""Audit game results and labels from both earlier projects.

Reference: the raw NHL schedule responses saved by v1
(data/raw/nhl_api/schedule_<season>.json). Each completed game there has an
official final score, where the shootout winner is credited one goal, and the
period the game ended in (REG, OT or SO).

Checks
  A. Reference sanity: one row per game id; completed games only.
  B. Coverage: every completed reference game appears exactly once in each
     label table, and no label exists for a game the reference lacks.
  C. Agreement with the reference: teams, final scores, decision type.
  D. Internal consistency of each label table:
       - final = regulation + overtime (+ 1 for the shootout winner)
       - margin and total derived correctly from the final score
       - OT and SO games tied after regulation; REG games not tied
       - OT games decided by exactly one goal; SO winner by one credited goal
       - shootout flag matches decision type
  E. v1 and v2 labels agree with each other.

Output: data/audit/game_results/ (CSV of every failing row per check) and a
JSON summary printed to stdout and written to docs/audit/game_results_summary.json.

Usage: python src/audit/audit_game_results.py
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
V2_RUN = "20261103T160656Z_4414a3fb"
OUT = os.path.join(ROOT, "data/audit/game_results")
SUMMARY = os.path.join(ROOT, "docs/audit/game_results_summary.json")
GAME_TYPES = {2: "REGULAR", 3: "PLAYOFF"}

summary: dict = {}
os.makedirs(OUT, exist_ok=True)


def record(check: str, failing: pd.DataFrame, total: int, note: str = "") -> None:
    n = len(failing)
    summary[check] = {"checked": int(total), "failing": int(n), **({"note": note} if note else {})}
    if n:
        failing.to_csv(os.path.join(OUT, f"{check}.csv"), index=False)
    status = "PASS" if n == 0 else "FAIL"
    print(f"{status:4}  {check:55} {n:6} of {total:6}  {note}")


def load_reference() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(V1, "data/raw/nhl_api/schedule_*.json"))):
        payload = json.load(open(path))
        for g in payload["games"]:
            outcome = (g.get("gameOutcome") or {}).get("lastPeriodType")
            rows.append(
                {
                    "game_id": g["id"],
                    "season_int": g["season"],
                    "game_type": g["gameType"],
                    "game_state": g["gameState"],
                    "scheduled_utc": g["startTimeUTC"],
                    "home_abbrev": g["homeTeam"]["abbrev"],
                    "away_abbrev": g["awayTeam"]["abbrev"],
                    "home_team_id": g["homeTeam"]["id"],
                    "away_team_id": g["awayTeam"]["id"],
                    "home_score": g["homeTeam"].get("score"),
                    "away_score": g["awayTeam"].get("score"),
                    "decision": outcome,
                }
            )
    ref = pd.DataFrame(rows)
    ref = ref[ref.game_type.isin(GAME_TYPES)]
    return ref


ref_all = load_reference()
record("A1_reference_duplicate_game_ids", ref_all[ref_all.game_id.duplicated(keep=False)], len(ref_all))
ref = ref_all[ref_all.game_state.isin(["OFF", "FINAL"]) & ref_all.decision.notna()].copy()
summary["reference_completed_games_by_season"] = ref.groupby("season_int").size().astype(int).to_dict()
summary["reference_decisions"] = ref.decision.value_counts().astype(int).to_dict()
print("reference completed games:", len(ref), summary["reference_completed_games_by_season"])

# ---------- v1 labels ----------
v1 = pd.read_parquet(os.path.join(V1, "data/labels/game_labels.parquet"))
v1g = pd.read_parquet(os.path.join(V1, "data/normalized/games.parquet"))

record("B1_v1_labels_duplicate_game_ids", v1[v1.game_id.duplicated(keep=False)], len(v1))
m = ref.merge(v1, on="game_id", how="left", indicator=True, suffixes=("_ref", ""))
record("B2_v1_labels_missing_completed_reference_games", m[m._merge == "left_only"][["game_id", "season_int", "game_type", "scheduled_utc_ref", "home_abbrev_ref", "away_abbrev_ref"]], len(ref))
extra = v1[~v1.game_id.isin(ref_all.game_id)]
record("B3_v1_labels_for_games_not_in_reference", extra, len(v1))
not_done = v1.merge(ref_all[["game_id", "game_state"]], on="game_id")
not_done = not_done[~not_done.game_state.isin(["OFF", "FINAL"])]
record("B4_v1_labels_for_games_not_completed", not_done, len(v1))

both = m[m._merge == "both"].copy()
record("C1_v1_teams_differ", both[(both.home_abbrev != both.home_abbrev_ref) | (both.away_abbrev != both.away_abbrev_ref)][["game_id", "home_abbrev_ref", "home_abbrev", "away_abbrev_ref", "away_abbrev"]], len(both),
       "abbreviation changes such as ARI to UTA appear here")
diff = both[(both.home_final_score != both.home_score) | (both.away_final_score != both.away_score)]
record("C2_v1_final_score_differs", diff[["game_id", "decision", "home_score", "away_score", "home_final_score", "away_final_score", "decision_type"]], len(both))
record("C3_v1_decision_differs", both[both.decision != both.decision_type][["game_id", "decision", "decision_type"]], len(both))

so_home = ((v1.decision_type == "SO") & (v1.home_final_score > v1.away_final_score)).astype(int)
so_away = ((v1.decision_type == "SO") & (v1.away_final_score > v1.home_final_score)).astype(int)
rebuilt_home = v1.home_reg_score + v1.home_ot_goals + so_home
rebuilt_away = v1.away_reg_score + v1.away_ot_goals + so_away
record("D1_v1_final_not_reg_plus_ot_plus_so_credit", v1[(rebuilt_home != v1.home_final_score) | (rebuilt_away != v1.away_final_score)], len(v1))
record("D2_v1_margin_wrong", v1[v1.home_margin != v1.home_final_score - v1.away_final_score], len(v1))
record("D3_v1_total_wrong", v1[v1.total_final_goals != v1.home_final_score + v1.away_final_score], len(v1),
       "total_final_goals is the official total, including the shootout credit")
tied = v1.home_reg_score == v1.away_reg_score
record("D4_v1_reg_decision_but_tied_after_regulation", v1[(v1.decision_type == "REG") & tied], len(v1))
record("D5_v1_ot_or_so_but_not_tied_after_regulation", v1[v1.decision_type.isin(["OT", "SO"]) & ~tied], len(v1))
record("D6_v1_ot_margin_not_one", v1[(v1.decision_type == "OT") & (v1.home_margin.abs() != 1)], len(v1))
record("D7_v1_so_margin_not_one", v1[(v1.decision_type == "SO") & (v1.home_margin.abs() != 1)], len(v1))
record("D8_v1_shootout_flag_mismatch", v1[v1.shootout_flag != (v1.decision_type == "SO")], len(v1))
record("D9_v1_ties", v1[v1.home_final_score == v1.away_final_score], len(v1))
pl = v1[v1.competition_phase == "PLAYOFF"]
record("D10_v1_playoff_games_marked_shootout", pl[pl.decision_type == "SO"], len(pl))
summary["v1_label_rows_by_season"] = v1.groupby("season").size().astype(int).to_dict()
summary["v1_label_phase_counts"] = v1.competition_phase.value_counts().astype(int).to_dict()

# v1 labels agree with v1 normalized games table
j = v1.merge(v1g, on="game_id", suffixes=("", "_g"))
cols = ["home_final_score", "away_final_score", "home_reg_score", "away_reg_score", "decision_type"]
bad = j[np.logical_or.reduce([j[c] != j[c + "_g"] for c in cols])]
record("D11_v1_labels_disagree_with_v1_games_table", bad[["game_id"] + cols + [c + "_g" for c in cols]], len(j))
gm = v1g[~v1g.game_id.isin(v1.game_id)]
summary["v1_games_table_rows_without_label"] = {"rows": int(len(gm)), "status_counts": gm.status.value_counts().astype(int).to_dict()}

# ---------- v2 labels ----------
v2 = pd.read_parquet(glob.glob(os.path.join(V2, f"data/features/*/{V2_RUN}/game_labels.parquet"))[0])
record("B5_v2_labels_duplicate_game_ids", v2[v2.game_id.duplicated(keep=False)], len(v2))
m2 = ref.merge(v2, on="game_id", how="left", indicator=True, suffixes=("", "_v2"))
record("B6_v2_labels_missing_completed_reference_games", m2[m2._merge == "left_only"][["game_id", "season_int", "game_type", "scheduled_utc"]], len(ref))
record("B7_v2_labels_for_games_not_in_reference", v2[~v2.game_id.isin(ref_all.game_id)][["game_id", "season", "game_type"]], len(v2),
       "games outside the v1 schedule files, e.g. seasons before 2019-20")
b2 = m2[m2._merge == "both"].copy()
record("C4_v2_final_score_differs", b2[(b2.label_home_final_score != b2.home_score) | (b2.label_away_final_score != b2.away_score)][["game_id", "decision", "home_score", "away_score", "label_home_final_score", "label_away_final_score"]], len(b2))
record("C5_v2_decision_differs", b2[b2.decision != b2.label_decision_type][["game_id", "decision", "label_decision_type"]], len(b2))
record("D12_v2_final_not_real_plus_credit", v2[(v2.label_home_real_goals + v2.label_home_shootout_credit != v2.label_home_final_score) | (v2.label_away_real_goals + v2.label_away_shootout_credit != v2.label_away_final_score)], len(v2))
record("D13_v2_shootout_credit_outside_so_games", v2[(v2.label_decision_type != "SO") & ((v2.label_home_shootout_credit + v2.label_away_shootout_credit) != 0)], len(v2))
record("D14_v2_so_games_without_exactly_one_credit", v2[(v2.label_decision_type == "SO") & ((v2.label_home_shootout_credit + v2.label_away_shootout_credit) != 1)], len(v2))
record("D15_v2_win_label_wrong", v2[v2.label_home_final_win != (v2.label_home_final_score > v2.label_away_final_score).astype(v2.label_home_final_win.dtype)], len(v2))
record("D16_v2_totals_wrong", v2[(v2.label_total_real_goals != v2.label_home_real_goals + v2.label_away_real_goals) | (v2.label_total_official_score != v2.label_home_final_score + v2.label_away_final_score)], len(v2))
summary["v2_label_rows_by_season"] = v2.groupby("season").size().astype(int).to_dict()

# ---------- v1 vs v2 ----------
x = v1.merge(v2, on="game_id")
bad = x[(x.home_final_score != x.label_home_final_score) | (x.away_final_score != x.label_away_final_score) | (x.decision_type != x.label_decision_type)]
record("E1_v1_v2_labels_disagree", bad[["game_id", "home_final_score", "away_final_score", "decision_type", "label_home_final_score", "label_away_final_score", "label_decision_type"]], len(x))
summary["v1_v2_overlap_games"] = int(len(x))

# ---------- F. independent source: PuckAPI get_games ----------
# data/puckapi/games_all.parquet is built from PuckAPI get_games responses
# (regular season and playoffs, every season from 2019-20). PuckAPI shows the
# Arizona franchise as UTA in every season, so teams are not compared here.
pk_path = os.path.join(ROOT, "data/puckapi/games_all.parquet")
if os.path.exists(pk_path):
    pk = pd.read_parquet(pk_path)
    pk["game_id"] = pk["id"].astype(int)
    pk = pk[pk.gameState.isin(["OFF", "FINAL"])]
    summary["puckapi_games_by_season"] = pk.groupby("season").size().astype(int).to_dict()

    f1 = v2.merge(pk, on="game_id", how="outer", indicator=True)
    record("F1_v2_labels_missing_from_puckapi", f1[f1._merge == "left_only"][["game_id", "season_x"]], len(v2))
    record("F2_puckapi_games_missing_from_v2_labels", f1[f1._merge == "right_only"][["game_id", "season_y", "gameDate"]], len(pk),
           "games finished after the v2 run date are expected here")
    fb = f1[f1._merge == "both"]
    record("F3_v2_final_score_differs_from_puckapi", fb[(fb.label_home_final_score != fb.homeScore) | (fb.label_away_final_score != fb.awayScore)][["game_id", "label_home_final_score", "label_away_final_score", "homeScore", "awayScore", "label_decision_type"]], len(fb))
    f4 = v1.merge(pk, on="game_id")
    record("F4_v1_final_score_differs_from_puckapi", f4[(f4.home_final_score != f4.homeScore) | (f4.away_final_score != f4.awayScore)][["game_id", "home_final_score", "away_final_score", "homeScore", "awayScore", "decision_type"]], len(f4))

os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
json.dump(summary, open(SUMMARY, "w"), indent=2, default=str)
print("summary written to", os.path.relpath(SUMMARY, ROOT))
