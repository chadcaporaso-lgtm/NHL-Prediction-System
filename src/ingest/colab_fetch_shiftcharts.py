"""Colab cell: NHL shift charts -> who played with whom, per game.

The points model lacks real linemates (docs/models/points-model.md, v2). The
NHL publishes every player's shifts per game (free, no key):
  https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=<id>
Since 2026-11-09 the Claude environment allows api.nhle.com, so the session
runs src/ingest/fetch_shiftcharts.py directly; this cell is the fallback.

For every game in the skater files already in Drive (2019-20 to the newest
weekly refresh) it fetches the shift chart, keeps the raw shifts in Drive,
and reduces each game to two small tables:

  shift_toi     one row per skater-game: seconds on ice in all situations,
                even strength (equal skaters, both goalies in), 5-on-5, own
                power play and shorthanded, from the shifts alone. Checked
                against the box-score ice time in the Claude session.
  shift_pairs   one row per pair of teammates per game: seconds on ice
                together at 5-on-5 and on their power play (pairs with at
                least 20 seconds combined).

Strength is counted second by second from who is on the ice: skaters are the
players in that game's skater summary, everyone else in the chart is a
goalie. Power play = both goalies in net and more skaters than the opponent.
That is a count, not the penalty clock, so it differs slightly from the
official power-play time (a delayed penalty with the goalie pulled is not
counted).

Rerunnable: games already fetched are skipped (cache per season under
01_raw/shiftcharts/), so the same cell also tops up new games each week.
Output parts and a SHA-256 manifest go to
01_raw/_transfer_3mb/shifts_<date>/ (download in the Claude session, then
`python3 run.py restore <manifest>`).

Run in Google Colab: paste the whole file into one cell. The first run
fetches about 9,500 games (roughly 20-40 minutes); later runs take a minute.
Local tests import the two functions above the main block
(tests/test_shift_pairs.py); the main block only runs in Colab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIOD = 1200  # seconds; overtime periods are indexed the same way


def to_seconds(t) -> float:
    if not isinstance(t, str) or ":" not in t:
        return np.nan
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def parse_shifts(rows: list[dict], game_id: int) -> pd.DataFrame:
    """Raw shiftcharts rows -> game_id, player_id, team_id, period, start_s, end_s (seconds from game start)."""
    out = []
    for r in rows:
        if r.get("typeCode") != 517 or r.get("playerId") is None:
            continue  # 505 = goal events, not shifts
        a, b = to_seconds(r.get("startTime")), to_seconds(r.get("endTime"))
        if np.isnan(a) or np.isnan(b) or b <= a:
            continue
        p = int(r["period"])
        out.append((game_id, int(r["playerId"]), int(r["teamId"]), p, (p - 1) * PERIOD + a, (p - 1) * PERIOD + b))
    d = pd.DataFrame(out, columns=["game_id", "player_id", "team_id", "period", "start_s", "end_s"])
    return d.drop_duplicates(["player_id", "start_s", "end_s"])


def game_aggregates(sh: pd.DataFrame, skaters: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One game's shifts -> (per-skater seconds by strength, teammate pair seconds at 5v5 and on the power play).

    A shift covers seconds start_s .. end_s - 1, so a change at 10:00 does not
    put both players on the ice at the same second.
    """
    gid = int(sh.game_id.iloc[0])
    T = int(sh.end_s.max())
    players = sh[["player_id", "team_id"]].drop_duplicates("player_id")
    teams = sorted(players.team_id.unique())
    if len(teams) != 2:
        return pd.DataFrame(), pd.DataFrame()
    ids = players.player_id.to_numpy()
    col = {p: i for i, p in enumerate(ids)}
    on = np.zeros((T, len(ids)), dtype=bool)
    for p, a, b in sh[["player_id", "start_s", "end_s"]].itertuples(index=False):
        on[int(a):int(b), col[p]] = True
    is_sk = np.array([p in skaters for p in ids])
    team = players.team_id.to_numpy()
    sk = {t: on[:, (team == t) & is_sk].sum(1) for t in teams}
    gl = {t: on[:, (team == t) & ~is_sk].any(1) for t in teams}
    both_g = gl[teams[0]] & gl[teams[1]]
    ev = both_g & (sk[teams[0]] == sk[teams[1]])
    five = ev & (sk[teams[0]] == 5)
    pp = {t: both_g & (sk[t] > sk[u]) for t, u in (teams, teams[::-1])}
    tot, pairs = [], []
    for t, u in (teams, teams[::-1]):
        m = (team == t) & is_sk
        A, pid = on[:, m], ids[m]
        tot.append(pd.DataFrame({"game_id": gid, "team_id": t, "player_id": pid,
                                 "n_shifts": sh[sh.player_id.isin(pid)].groupby("player_id").size().reindex(pid).to_numpy(),
                                 "s_all": A.sum(0), "s_ev": A[ev].sum(0), "s_5v5": A[five].sum(0),
                                 "s_pp": A[pp[t]].sum(0), "s_sh": A[pp[u]].sum(0)}))
        A = A.astype(np.int32)
        s5 = A[five].T @ A[five]
        spp = A[pp[t]].T @ A[pp[t]]
        i, j = np.triu_indices(len(pid), k=1)
        keep = (s5[i, j] + spp[i, j]) >= 20
        pairs.append(pd.DataFrame({"game_id": gid, "team_id": t, "player_id": pid[i[keep]], "mate_id": pid[j[keep]],
                                   "s_5v5": s5[i, j][keep], "s_pp": spp[i, j][keep]}))
    return pd.concat(tot, ignore_index=True), pd.concat(pairs, ignore_index=True)


URL = "https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={}"


def fetch_all(skater_files: list[str], cache: str, workers: int = 4):
    """Fetch every game in the skater files (skipping games cached in `cache`), return (toi, pairs, fetch log)."""
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    import requests

    os.makedirs(cache, exist_ok=True)
    sk = pd.concat([pd.read_parquet(p, columns=["gameId", "playerId"]) for p in skater_files], ignore_index=True)
    sk.columns = ["game_id", "player_id"]
    skaters = sk.groupby("game_id").player_id.apply(set).to_dict()
    print(f"{len(skaters)} games in the skater files ({len(skater_files)} files)", flush=True)
    session = requests.Session()

    def fetch(gid):
        for attempt in range(5):
            try:
                r = session.get(URL.format(gid), timeout=30)
                if r.status_code == 200:
                    return gid, r.json().get("data", []), "ok"
                if r.status_code not in (429, 500, 502, 503, 504):
                    return gid, [], f"http {r.status_code}"
            except (requests.RequestException, ValueError):
                pass
            time.sleep(2 ** attempt)
        return gid, [], "failed"

    all_toi, all_pairs, all_log = [], [], []
    for season in sorted({g // 1_000_000 for g in skaters}):
        sid = f"{season}{season + 1}"
        f_raw, f_log = f"{cache}/shifts_{sid}.parquet", f"{cache}/fetch_log_{sid}.parquet"
        raw = pd.read_parquet(f_raw) if os.path.exists(f_raw) else pd.DataFrame()
        log = pd.read_parquet(f_log) if os.path.exists(f_log) else pd.DataFrame(columns=["game_id", "rows", "status"])
        done = set(log.loc[(log.status == "ok") & (log.rows > 0), "game_id"])
        todo = sorted(g for g in skaters if g // 1_000_000 == season and g not in done)
        new_raw, new_log = [], []
        with ThreadPoolExecutor(workers) as ex:
            for k, (gid, rows, status) in enumerate(ex.map(fetch, todo), 1):
                new_raw.append(parse_shifts(rows, gid))
                new_log.append((gid, len(new_raw[-1]), status))
                if k % 500 == 0:
                    print(f"  {sid}: {k}/{len(todo)}", flush=True)
        if todo:
            log = pd.concat([log[~log.game_id.isin(todo)], pd.DataFrame(new_log, columns=["game_id", "rows", "status"])],
                            ignore_index=True)
            raw = pd.concat([raw[~raw.game_id.isin(todo)] if len(raw) else raw, *new_raw], ignore_index=True)
            raw.to_parquet(f_raw, index=False)
            log.to_parquet(f_log, index=False)
        print(f"{sid}: {len(todo)} fetched now, {int(((log.status == 'ok') & (log.rows > 0)).sum())} games with shifts, "
              f"{int((log.rows == 0).sum())} empty", flush=True)
        log["season"] = int(sid)
        all_log.append(log)
        for gid, g in raw.groupby("game_id"):
            t, p = game_aggregates(g, skaters.get(gid, set()))
            all_toi.append(t)
            all_pairs.append(p)
    toi = pd.concat(all_toi, ignore_index=True)
    pairs = pd.concat(all_pairs, ignore_index=True)
    for c in ["n_shifts", "s_all", "s_ev", "s_5v5", "s_pp", "s_sh"]:
        toi[c] = toi[c].astype("int16")
    for c in ["s_5v5", "s_pp"]:
        pairs[c] = pairs[c].astype("int16")
    for d in (toi, pairs):
        d["team_id"] = d.team_id.astype("int16")
        for c in ["player_id", "mate_id"]:
            if c in d:
                d[c] = d[c].astype("int32")
    games = pd.concat(all_log, ignore_index=True)
    print(f"skater-games {len(toi):,}, pairs {len(pairs):,}, games with shifts {int((games.rows > 0).sum()):,} of {len(games):,}")
    return toi, pairs, games


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__" and _in_colab():
    import glob
    import hashlib
    import json
    import os
    from datetime import date

    from google.colab import drive

    drive.mount("/content/drive")
    MY = "/content/drive/MyDrive"
    RAW = f"{MY}/NHL_Prediction_System_Claude/01_raw"
    CACHE = f"{RAW}/shiftcharts"
    TODAY = date.today().isoformat()
    STAGE = f"{RAW}/_transfer_3mb/shifts_{TODAY}"
    os.makedirs(STAGE, exist_ok=True)
    src = [f"{RAW}/nhl_api_refetch/skater_summary_{s}.parquet" for s in ("20192020", "20202021", "20212022", "20222023", "20232024")]
    src += [f"{MY}/NHL_Prediction_System/data/raw/nhl_api/skater_summary_{s}.parquet" for s in ("20242025", "20252026")]
    src += sorted(glob.glob(f"{RAW}/live/*/skater_summary_20262027.parquet"))[-1:]
    toi, pairs, games = fetch_all(src, CACHE)
    out = {"shift_toi.parquet": toi, "shift_pairs.parquet": pairs, "shift_fetch_log.parquet": games}
    manifest = []
    for name, d in out.items():
        d.to_parquet(f"{CACHE}/{name}", index=False, compression="zstd")
        data = open(f"{CACHE}/{name}", "rb").read()
        parts = []
        for i in range(0, len(data), 3 * 1024 * 1024):
            part = f"shifts__{name}.part{i // (3 * 1024 * 1024):03d}"
            open(f"{STAGE}/{part}", "wb").write(data[i:i + 3 * 1024 * 1024])
            parts.append(part)
        manifest.append({"source": f"NHL_Prediction_System_Claude/01_raw/shiftcharts/{name}", "size_bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest(), "parts": parts})
    json.dump(manifest, open(f"{STAGE}/manifest_shifts_{TODAY}.json", "w"), indent=2)
    print(f"{len(manifest)} files, {sum(len(m['parts']) for m in manifest)} parts, "
          f"{sum(m['size_bytes'] for m in manifest) / 1e6:.1f} MB -> {STAGE}")
