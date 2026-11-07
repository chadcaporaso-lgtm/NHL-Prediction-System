"""Audit the v2 goalie game table against the official schedule and team scores.

Input:  v2 goalie_history_candidates.parquet (run 20261103T160656Z_4414a3fb)
        v2 games_schedule.parquet and official_team_games.parquet (same run)
Output: docs/audit/goalie_games_summary.json

Checks
  K1  coverage: completed games per season with goalie rows for both teams
  K2  duplicates on (game_id, player_id)
  K3  exactly one starter per team-game
  K4  saves + goals against = shots against
  K5  goals against summed over a team's goalies <= opponent real goals
      (the difference should be empty-net goals, so 0 to about 3)
  K6  shots against summed over a team's goalies vs official opponent shots
"""

from __future__ import annotations

import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "data/drive/NHL_Prediction_System_ChatGPT/data/normalized/official_incremental/20261103T160656Z_4414a3fb")
g = pd.read_parquet(f"{RUN}/goalie_history_candidates.parquet")
s = pd.read_parquet(f"{RUN}/games_schedule.parquet")
tg = pd.read_parquet(f"{RUN}/official_team_games.parquet")
done = s[s.home_final_score.notna() & s.game_type.isin([2, 3])]

out = {"rows": len(g), "seasons": {}}
for season, d in done.groupby("season"):
    gs = g[g.season == season]
    teams_per_game = gs.groupby("game_id").team_id.nunique()
    out["seasons"][str(season)] = {
        "completed_games": int(len(d)),
        "games_with_goalie_rows": int(gs.game_id.nunique()),
        "games_with_both_teams": int((teams_per_game == 2).sum()),
    }
out["K2_duplicate_game_player"] = int(g.duplicated(["game_id", "player_id"]).sum())

starters = g.groupby(["game_id", "team_id"]).starter.sum()
out["K3_team_games"] = int(len(starters))
out["K3_team_games_not_one_starter"] = int((starters != 1).sum())
out["K3_sample"] = [list(map(int, k)) + [int(v)] for k, v in starters[starters != 1].head(10).items()]

out["K4_saves_ga_mismatch"] = int((g.saves + g.goals_against != g.shots_against).sum())

team = g.groupby(["game_id", "team_id"])[["goals_against", "shots_against"]].sum().reset_index()
team = team.merge(tg[["game_id", "team_id", "real_goals_against", "sog_against", "game_type"]], on=["game_id", "team_id"], how="left")
diff = team.real_goals_against - team.goals_against
out["K5_team_games_matched"] = int(team.real_goals_against.notna().sum())
out["K5_real_minus_goalie_ga_counts"] = {str(int(k)): int(v) for k, v in diff.value_counts().sort_index().items()}
sdiff = team.sog_against - team.shots_against
out["K6_official_minus_goalie_sa_counts"] = {str(int(k)): int(v) for k, v in sdiff.value_counts().sort_index().items()}
out["K6_sample"] = team.loc[sdiff.abs() > 0, ["game_id", "team_id", "sog_against", "shots_against"]].head(10).astype(int).values.tolist()

out["starter_share_of_minutes"] = round(float(g[g.starter].toi_total_sec.sum() / g.toi_total_sec.sum()), 4)
out["flags"] = {c: int(g[c].sum()) for c in ["historical_starter_flag_only", "target_pregame_starter_eligible", "historical_availability_verified"]}
json.dump(out, open(os.path.join(ROOT, "docs/audit/goalie_games_summary.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
