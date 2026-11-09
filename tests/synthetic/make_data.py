"""Generate a small synthetic NHL data set with the same files and columns as
the real inputs, so the pipeline and its leakage tests can run where the
real (licensed) data cannot be shared, such as GitHub Actions.

Every number is random. The league has 8 teams, 2018-19 (MoneyPuck warm-up
only) to 2026-27, with the features the code has to handle: Arizona's history
continuing into Utah from 2024-25, back-to-backs, overtime and shootout
games, goalie rotations with a third goalie called up, pulled goalies,
playoff games, and power-play-free games (zero 5-on-4 ice time).

Usage: python3 tests/synthetic/make_data.py <root>
Writes <root>/data/... at the paths src/common/official.py and the feature
builders read. Refuses to write into a folder that already has data/.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

RUN = "data/drive/NHL_Prediction_System_ChatGPT/data/normalized/official_incremental/20261103T160656Z_4414a3fb"
REFETCH = "data/drive/NHL_Prediction_System_Claude/01_raw/nhl_api_refetch"
V1 = "data/drive/NHL_Prediction_System/data/raw/nhl_api"
MONEYPUCK = "data/drive/NHL_Prediction_System_Claude/01_raw/moneypuck/moneypuck_all_teams_2018_plus_2026-11-07.parquet"
TEAMS = {"NJD": 1, "NYR": 3, "BOS": 6, "MTL": 8, "OTT": 9, "TOR": 10, "CHI": 16, "ARI": 53, "UTA": 59}
SEASONS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]  # start years
GAME_DAYS = {2026: 14}  # the season in progress is short
DEFAULT_GAME_DAYS = 40

rng = np.random.default_rng(2026)


def teams_for(year: int) -> list[str]:
    t = ["NJD", "NYR", "BOS", "MTL", "OTT", "TOR", "CHI"]
    return t + (["UTA"] if year >= 2024 else ["ARI"])


def franchise(team: str) -> str:
    return "UTA" if team == "ARI" else team  # Utah keeps Arizona's players


def make_rosters():
    """Stable player ids per franchise with some turnover each season."""
    next_id = [8470000]

    def new_player(pos):
        next_id[0] += 1
        return {"id": next_id[0], "pos": pos, "rate": float(rng.gamma(4, 0.5 if pos != "D" else 0.3))}

    skaters, goalies = {}, {}
    for fr in {franchise(t) for t in TEAMS}:
        roster = [new_player(p) for p in ["C"] * 5 + ["L"] * 5 + ["R"] * 4 + ["D"] * 8]
        gs = [{"id": 0, "skill": 0.0} for _ in range(3)]
        for g in gs:
            next_id[0] += 1
            g["id"], g["skill"] = next_id[0], float(rng.normal(0, 0.01))
        for y in SEASONS:
            if y > SEASONS[0]:
                for i in rng.choice(len(roster), 3, replace=False):
                    roster[i] = new_player(roster[i]["pos"])
            skaters[(fr, y)] = [dict(p) for p in roster]
            goalies[(fr, y)] = gs
    return skaters, goalies


def schedule(year: int) -> list[dict]:
    games, day, n = [], pd.Timestamp(f"{year}-10-05"), 0
    teams = teams_for(year)
    for gd in range(GAME_DAYS.get(year, DEFAULT_GAME_DAYS)):
        day += pd.Timedelta(days=int(rng.choice([1, 1, 2, 2, 3])))
        order = rng.permutation(teams)
        for i in range(0, len(order), 2):
            n += 1
            home, away = (order[i], order[i + 1]) if rng.random() < 0.5 else (order[i + 1], order[i])
            hour = "23:00:00" if rng.random() < 0.8 else "23:30:00"
            games.append({"game_id": year * 1_000_000 + 20_000 + n, "game_type": 2, "date": day,
                          "utc": f"{day.date()}T{hour}Z", "home": str(home), "away": str(away)})
    if year < 2026:  # a few playoff games
        for k in range(2):
            day += pd.Timedelta(days=2)
            pair = rng.choice(teams, 2, replace=False)
            games.append({"game_id": year * 1_000_000 + 30_000 + 111 + k, "game_type": 3, "date": day,
                          "utc": f"{day.date()}T23:00:00Z", "home": str(pair[0]), "away": str(pair[1])})
    return games


def main(root: str):
    if os.path.exists(os.path.join(root, "data")):
        sys.exit(f"{root}/data already exists; refusing to overwrite real data")
    skaters, goalies = make_rosters()
    sched, team_games, goalie_rows, mp_rows = [], [], [], []
    skater_sum, skater_toi = {}, {}
    last_start, last_game_day = {}, {}
    strength = {(franchise(t), y): rng.normal(0, 0.12, 2) for t in TEAMS for y in SEASONS}

    for year in SEASONS:
        season = year * 10000 + year + 1
        for g in schedule(year):
            h, a = g["home"], g["away"]
            sh, sa = strength[(franchise(h), year)], strength[(franchise(a), year)]
            lam_h = 2.9 * np.exp(sh[0] - sa[1] + 0.05)
            lam_a = 2.9 * np.exp(sa[0] - sh[1])
            reg_h, reg_a = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
            real_h, real_a, fin_h, fin_a, dec = reg_h, reg_a, reg_h, reg_a, "REG"
            if reg_h == reg_a:
                home_wins = rng.random() < 0.5
                if g["game_type"] == 3 or rng.random() < 0.6:
                    dec = "OT"
                    real_h, real_a = (reg_h + 1, reg_a) if home_wins else (reg_h, reg_a + 1)
                    fin_h, fin_a = real_h, real_a
                else:
                    dec = "SO"
                    fin_h, fin_a = (reg_h + 1, reg_a) if home_wins else (reg_h, reg_a + 1)
            date_s = str(g["date"].date())

            # Skaters: 12 forwards and 6 defensemen dress; goals spread by shot rate.
            sog, lines = {}, {}
            for team, opp, real, is_home in [(h, a, real_h, 1), (a, h, real_a, 0)]:
                ros = skaters[(franchise(team), year)]
                fwd = [p for p in ros if p["pos"] != "D"]
                dmen = [p for p in ros if p["pos"] == "D"]
                dressed = list(rng.choice(fwd, 12, replace=False)) + list(rng.choice(dmen, 6, replace=False))
                rates = np.array([p["rate"] for p in dressed])
                shots = rng.poisson(rates)
                goals = rng.multinomial(real, rates / rates.sum()) if real else np.zeros(len(dressed), int)
                assists = rng.multinomial(int(real * 1.7), rates / rates.sum()) if real else np.zeros(len(dressed), int)
                shots = np.maximum(shots, goals)
                sog[team] = int(shots.sum())
                pp_g = np.minimum(goals, rng.binomial(goals, 0.2))
                toi = np.array([rng.normal(1250, 120) if p["pos"] == "D" else rng.normal(950, 150) for p in dressed]).clip(300)
                pp = (toi * rng.uniform(0, 0.15, len(dressed))).astype(int)
                sh_ = (toi * rng.uniform(0, 0.1, len(dressed))).astype(int)
                rows_s, rows_t = [], []
                for p, s_, g_, a_, ppg, t_, ppt, sht in zip(dressed, shots, goals, assists, pp_g, toi, pp, sh_):
                    rows_s.append({"gameId": g["game_id"], "playerId": p["id"], "skaterFullName": f"Player {p['id']}",
                                   "positionCode": p["pos"], "teamAbbrev": team, "opponentTeamAbbrev": opp,
                                   "homeRoad": "H" if is_home else "R", "gameDate": date_s, "goals": int(g_),
                                   "assists": int(a_), "points": int(g_ + a_), "shots": int(s_),
                                   "ppPoints": int(ppg + min(a_, 1) * (rng.random() < 0.2)), "ppGoals": int(ppg)})
                    rows_t.append({"gameId": g["game_id"], "playerId": p["id"], "timeOnIce": int(t_),
                                   "evTimeOnIce": int(t_) - int(ppt) - int(sht), "ppTimeOnIce": int(ppt),
                                   "shTimeOnIce": int(sht), "shifts": int(rng.integers(15, 30))})
                skater_sum.setdefault(season, []).extend(rows_s)
                skater_toi.setdefault(season, []).extend(rows_t)

            # Official tables (2019-20 on).
            if year >= 2019:
                sched.append({"game_id": g["game_id"], "season": season, "game_type": g["game_type"],
                              "competition_phase": "REGULAR" if g["game_type"] == 2 else "PLAYOFF",
                              "game_date": date_s, "scheduled_utc": g["utc"], "status": "OFF", "schedule_state": "OK",
                              "home_team_id": TEAMS[h], "away_team_id": TEAMS[a], "home_abbrev": h, "away_abbrev": a,
                              "home_final_score": float(fin_h), "away_final_score": float(fin_a), "decision_type": dec})
                for team, opp, rf, ra, ff, fa, is_home in [(h, a, real_h, real_a, fin_h, fin_a, True),
                                                           (a, h, real_a, real_h, fin_a, fin_h, False)]:
                    team_games.append({
                        "game_id": g["game_id"], "season": season, "game_type": g["game_type"], "game_date": date_s,
                        "scheduled_utc": g["utc"], "team_id": TEAMS[team], "opponent_id": TEAMS[opp],
                        "is_home": is_home, "final_score_for": ff, "final_score_against": fa,
                        "real_goals_for": rf, "real_goals_against": ra,
                        "shootout_credit_for": int(dec == "SO" and ff > fa), "shootout_credit_against": int(dec == "SO" and fa > ff),
                        "sog_for": sog[team], "sog_against": sog[opp], "won_final": bool(ff > fa), "decision_type": dec})

                # Goalies: two dress; the 1A starts about 60%, rarely on the second night of a back-to-back.
                for team, opp, is_home in [(h, a, 1), (a, h, 0)]:
                    fr = franchise(team)
                    g1, g2, g3 = goalies[(fr, year)]
                    backup = g3 if rng.random() < 0.06 else g2
                    b2b = fr in last_game_day and (g["date"] - last_game_day[fr]).days == 1
                    p1 = 0.15 if (b2b and last_start.get(fr) == g1["id"]) else 0.6
                    starter, other = (g1, backup) if rng.random() < p1 else (backup, g1)
                    last_start[fr], last_game_day[fr] = starter["id"], g["date"]
                    shots_ag, goals_ag = sog[opp], real_a if is_home else real_h
                    pulled = rng.random() < 0.05 and shots_ag > 10
                    split = int(shots_ag * 0.6) if pulled else shots_ag
                    ga1 = min(goals_ag, int(round(goals_ag * (0.7 if pulled else 1.0))))
                    for gk, start, sa_, ga_, toi_ in [(starter, True, split, ga1, 2400 if pulled else 3600),
                                                      (other, False, shots_ag - split, goals_ag - ga1, 1200 if pulled else 0)]:
                        sa_ = max(sa_, ga_)
                        goalie_rows.append({
                            "game_id": g["game_id"], "season": season, "game_type": g["game_type"], "game_date": date_s,
                            "scheduled_utc": g["utc"], "player_id": gk["id"], "player_name": f"Goalie {gk['id']}",
                            "team_id": TEAMS[team], "team_abbrev": team, "opponent_id": TEAMS[opp], "is_home": is_home,
                            "position": "G", "toi_total_sec": toi_, "starter": start, "saves": sa_ - ga_,
                            "shots_against": sa_, "goals_against": ga_, "participated": toi_ > 0})

            # MoneyPuck team rows (2018-19 on), situations as in the real file.
            for team, opp, gf, ga_, ha in [(h, a, real_h, real_a, "HOME"), (a, h, real_a, real_h, "AWAY")]:
                for sit, share in [("all", 1.0), ("5on5", 0.8), ("5on4", 0.1), ("4on5", 0.1), ("other", 0.0)]:
                    ice = 3600 * share * (0 if (sit in ("5on4", "4on5") and rng.random() < 0.03) else 1)
                    xgf, xga = rng.gamma(9, (lam_h if team == h else lam_a) / 9 * share + 1e-6), rng.gamma(9, (lam_a if team == h else lam_h) / 9 * share + 1e-6)
                    mp_rows.append({
                        "team": team, "season": year, "gameId": g["game_id"], "opposingTeam": opp, "home_or_away": ha,
                        "gameDate": int(g["date"].strftime("%Y%m%d")), "situation": sit, "iceTime": float(ice),
                        "xGoalsFor": float(xgf), "xGoalsAgainst": float(xga),
                        "flurryScoreVenueAdjustedxGoalsFor": float(xgf * rng.uniform(0.9, 1.1)),
                        "goalsFor": float(gf if sit == "all" else rng.binomial(gf, share)),
                        "goalsAgainst": float(ga_ if sit == "all" else rng.binomial(ga_, share)),
                        "shotsOnGoalFor": float(sog[team] * share), "shotsOnGoalAgainst": float(sog[opp] * share),
                        "shotAttemptsFor": float(sog[team] * 1.8 * share), "shotAttemptsAgainst": float(sog[opp] * 1.8 * share)})

    def write(df, rel):
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)

    write(pd.DataFrame(sched), f"{RUN}/games_schedule.parquet")
    write(pd.DataFrame(team_games), f"{RUN}/official_team_games.parquet")
    write(pd.DataFrame(goalie_rows), f"{RUN}/goalie_history_candidates.parquet")
    write(pd.DataFrame(mp_rows), MONEYPUCK)
    for season, rows in skater_sum.items():
        if season < 20192020:
            continue
        base = V1 if season in (20242025, 20252026) else REFETCH
        write(pd.DataFrame(rows), f"{base}/skater_summary_{season}.parquet")
        write(pd.DataFrame(skater_toi[season]), f"{base}/skater_timeonice_{season}.parquet")
    print(f"synthetic data: {len(sched)} games, {len(goalie_rows)} goalie lines, "
          f"{sum(len(v) for v in skater_sum.values())} skater lines -> {root}/data")


if __name__ == "__main__":
    main(sys.argv[1])
