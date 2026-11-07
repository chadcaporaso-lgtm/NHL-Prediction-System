"""Colab cell: weekly refresh of current-season data for live tracking.

Fetches, for 2026-27 from opening night to yesterday:
  1. Schedule, final scores and goalie lines from the NHL web API
     (api-web.nhle.com schedule and boxscore endpoints)
       games_schedule_live.parquet   one row per game
       team_games_live.parquet       two rows per completed game: final score,
                                     real goals (shootout credit removed), shots
       goalie_games_live.parquet     one row per goalie per completed game
  2. Skater game summaries and ice time (NHL stats API, same method as before)
  3. MoneyPuck team game-by-game file (seasons 2018 onward)
Then copies everything into 3 MB parts with a SHA-256 manifest under
NHL_Prediction_System_Claude/01_raw/_transfer_3mb/live_<date>/.

Run in Google Colab: paste into one cell and run. About 10-15 minutes.
"""

import hashlib
import io
import json
import os
import time
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd
import requests
from google.colab import drive

drive.mount("/content/drive")

TODAY = date.today().isoformat()
FIRST, LAST = date(2026, 9, 25), date.today() - timedelta(days=1)
SEASON = 20262027
ROOT = "/content/drive/MyDrive/NHL_Prediction_System_Claude/01_raw"
LIVE = f"{ROOT}/live/{TODAY}"
STAGE = f"{ROOT}/_transfer_3mb/live_{TODAY}"
os.makedirs(LIVE, exist_ok=True)
os.makedirs(STAGE, exist_ok=True)
session = requests.Session()
session.headers["User-Agent"] = "nhl-prediction-research/1.0"


def get(url, tries=5, as_json=True):
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=120)
            if r.status_code == 200:
                return r.json() if as_json else r.content
            print("  HTTP", r.status_code, url)
        except requests.RequestException as exc:
            print("  error", exc)
        time.sleep(2 ** attempt)
    raise RuntimeError("failed after retries: " + url)


def mmss(s):
    if not s:
        return 0
    m, sec = str(s).split(":")
    return int(m) * 60 + int(sec)


# ---- 1. schedule, scores and goalies ----
games = {}
d = FIRST
while d <= LAST:
    for day in get(f"https://api-web.nhle.com/v1/schedule/{d.isoformat()}").get("gameWeek", []):
        for g in day.get("games", []):
            if g.get("season") == SEASON and g.get("gameType") in (2, 3):
                g["_local_date"] = day["date"]  # evening games start after midnight UTC
                games[g["id"]] = g
    d += timedelta(days=7)
print("scheduled games found:", len(games))

sched, team_rows, goalie_rows, failed = [], [], [], []
for gid, g in sorted(games.items()):
    row = {"game_id": gid, "season": SEASON, "game_type": g["gameType"], "scheduled_utc": g["startTimeUTC"],
           "home_team_id": g["homeTeam"]["id"], "away_team_id": g["awayTeam"]["id"],
           "home_abbrev": g["homeTeam"]["abbrev"], "away_abbrev": g["awayTeam"]["abbrev"],
           "status": g.get("gameState"), "game_date": None, "home_final_score": None, "away_final_score": None,
           "decision_type": None}
    if g.get("gameState") in ("OFF", "FINAL") and g["_local_date"] <= LAST.isoformat():
        try:
            b = get(f"https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore")
            dec = (b.get("gameOutcome") or {}).get("lastPeriodType", "REG")
            h, a = b["homeTeam"], b["awayTeam"]
            row.update(game_date=b["gameDate"], home_final_score=h["score"], away_final_score=a["score"], decision_type=dec)
            for side, team, opp, is_home in [("homeTeam", h, a, 1), ("awayTeam", a, h, 0)]:
                won = team["score"] > opp["score"]
                so_credit = int(dec == "SO" and won)
                so_credit_against = int(dec == "SO" and not won)
                team_rows.append({
                    "game_id": gid, "season": SEASON, "game_type": g["gameType"], "game_date": b["gameDate"],
                    "scheduled_utc": g["startTimeUTC"], "team_id": team["id"], "opponent_id": opp["id"],
                    "is_home": is_home, "final_score_for": team["score"], "final_score_against": opp["score"],
                    "real_goals_for": team["score"] - so_credit, "real_goals_against": opp["score"] - so_credit_against,
                    "shootout_credit_for": so_credit, "shootout_credit_against": so_credit_against,
                    "sog_for": team.get("sog"), "sog_against": opp.get("sog"), "won_final": won, "decision_type": dec})
                for gl in b.get("playerByGameStats", {}).get(side, {}).get("goalies", []):
                    sa = gl.get("shotsAgainst")
                    sv = gl.get("saves")
                    if sa is None and gl.get("saveShotsAgainst"):
                        sv, sa = (int(x) for x in gl["saveShotsAgainst"].split("/"))
                    toi = mmss(gl.get("toi"))
                    goalie_rows.append({
                        "game_id": gid, "season": SEASON, "game_type": g["gameType"], "game_date": b["gameDate"],
                        "scheduled_utc": g["startTimeUTC"], "player_id": gl["playerId"],
                        "player_name": (gl.get("name") or {}).get("default"), "team_id": team["id"],
                        "team_abbrev": team["abbrev"], "opponent_id": opp["id"], "is_home": is_home,
                        "toi_total_sec": toi, "starter": bool(gl.get("starter")), "saves": sv or 0,
                        "shots_against": sa or 0, "goals_against": gl.get("goalsAgainst", 0), "participated": toi > 0})
        except Exception as exc:  # keep going; report at the end
            failed.append((gid, repr(exc)))
        time.sleep(0.1)
    sched.append(row)

sched, team_games, goalies = pd.DataFrame(sched), pd.DataFrame(team_rows), pd.DataFrame(goalie_rows)
sched.to_parquet(f"{LIVE}/games_schedule_live.parquet", index=False)
team_games.to_parquet(f"{LIVE}/team_games_live.parquet", index=False)
goalies.to_parquet(f"{LIVE}/goalie_games_live.parquet", index=False)
done = sched.home_final_score.notna()
starters = goalies[goalies.starter].groupby(["game_id", "team_id"]).size()
print(f"completed games: {int(done.sum())}  through {sched.loc[done, 'game_date'].max()}")
print("decision types:", sched.loc[done, "decision_type"].value_counts().to_dict())
print("team-games with exactly one starter:", int((starters == 1).sum()), "of", len(team_games))
print("goalie rows where saves + goals != shots:", int((goalies.saves + goalies.goals_against != goalies.shots_against).sum()))
print("boxscore failures:", failed[:5], len(failed))

# ---- 2. skater game data ----
BASE = "https://api.nhle.com/stats/rest/en/skater/{report}"
SORT = quote(json.dumps([{"property": "gameId", "direction": "ASC"}, {"property": "playerId", "direction": "ASC"}]))
for rep in ("summary", "timeonice"):
    rows, d0 = [], FIRST
    while d0 <= LAST:
        end = min(d0 + timedelta(days=6), LAST)
        cay = quote(f'seasonId={SEASON} and (gameTypeId=2 or gameTypeId=3) and gameDate>="{d0}" and gameDate<="{end}"')
        offset = 0
        while True:
            p = get(f"{BASE.format(report=rep)}?isAggregate=false&isGame=true&sort={SORT}"
                    f"&start={offset}&limit=100&cayenneExp={cay}")
            batch = p.get("data", [])
            rows.extend(batch)
            offset += len(batch)
            if len(batch) < 100:
                break
            time.sleep(0.15)
        d0 = end + timedelta(days=1)
    df = pd.DataFrame(rows)
    df.to_parquet(f"{LIVE}/skater_{rep}_{SEASON}.parquet", index=False)
    per_game = df.groupby("gameId").size()
    print(f"skater {rep}: {len(df)} rows, {per_game.size} games, {int((per_game != 36).sum())} without 36, "
          f"{int(df.duplicated(['gameId', 'playerId']).sum())} duplicates")

# ---- 3. MoneyPuck ----
mp = pd.read_csv(io.BytesIO(get("https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv", as_json=False)),
                 low_memory=False)
mp = mp[mp.season >= 2018]
mp.to_parquet(f"{LIVE}/moneypuck_all_teams_2018_plus.parquet", index=False)
print("MoneyPuck 2026-27 games:", mp[mp.season == 2026].gameId.nunique(), "through", mp.gameDate.max())

# ---- 4. stage for transfer ----
PART = 3 * 1024 * 1024
manifest = []
for name in sorted(os.listdir(LIVE)):
    data = open(f"{LIVE}/{name}", "rb").read()
    parts = []
    for i in range(0, len(data), PART):
        part = f"live_{TODAY}__{name}.part{i // PART:03d}"
        open(f"{STAGE}/{part}", "wb").write(data[i:i + PART])
        parts.append(part)
    manifest.append({"file": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "parts": parts})
json.dump(manifest, open(f"{STAGE}/manifest_live_{TODAY}.json", "w"), indent=2)
print("staged", len(manifest), "files in", sum(len(m["parts"]) for m in manifest), "parts")
for m in manifest:
    print(f"  {m['file']}: {m['size_bytes']:,} bytes, sha256 {m['sha256'][:16]}")
