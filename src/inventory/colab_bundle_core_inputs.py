"""Colab cell: bundle the core inputs a new Claude Code session needs.

A cloud session starts on a fresh machine with only the git repository. The
pipeline needs 19 audited input files (about 28 MB) plus the newest weekly
refresh, and optionally the owner-supplied odds files in 01_raw/external/
(about 7 MB; src/inventory/place_external.py puts them in data/external/). This cell copies them into 3 MB parts with a SHA-256 manifest under
NHL_Prediction_System_Claude/01_raw/_transfer_3mb/core_<date>/. The new
session downloads the parts and runs src/inventory/restore_core_inputs.py.
Originals are only read.
"""

import glob
import hashlib
import json
import os
from datetime import date

from google.colab import drive

drive.mount("/content/drive")
MY = "/content/drive/MyDrive"
TODAY = date.today().isoformat()
DEST = f"{MY}/NHL_Prediction_System_Claude/01_raw/_transfer_3mb/core_{TODAY}"
PART = 3 * 1024 * 1024
V2 = "NHL_Prediction_System_ChatGPT/data/normalized/official_incremental/20261103T160656Z_4414a3fb"
FILES = [f"{V2}/games_schedule.parquet", f"{V2}/goalie_history_candidates.parquet", f"{V2}/official_team_games.parquet",
         "NHL_Prediction_System_ChatGPT/data/evaluation/external_closing_comparison/20261026T210505Z_bed9e020/external_official_crosswalk.parquet"]
FILES += [f"NHL_Prediction_System/data/raw/nhl_api/skater_{r}_{s}.parquet" for s in ("20242025", "20252026") for r in ("summary", "timeonice")]
FILES += [f"NHL_Prediction_System_Claude/01_raw/nhl_api_refetch/skater_{r}_{s}.parquet"
          for s in ("20192020", "20202021", "20212022", "20222023", "20232024") for r in ("summary", "timeonice")]
FILES += ["NHL_Prediction_System_Claude/01_raw/moneypuck/moneypuck_all_teams_2018_plus_2026-11-07.parquet"]
# Owner-supplied odds (decisions 37-39), if present: put them in 01_raw/external/.
EXTERNAL = ["nhl_2026_27_odds.csv", "sportsbook-nhl-2021-2022.csv", "sportsbook-nhl-2022-2023.csv", "archive (8).zip"]
for name in EXTERNAL:
    hits = sorted(glob.glob(f"{MY}/NHL_Prediction_System_Claude/01_raw/external/{name}")
                  + glob.glob(f"{MY}/NHL_Prediction_System_Claude/01_raw/inspect/{name}"))
    if hits:
        FILES.append(os.path.relpath(hits[0], MY))
    else:
        print("not found (optional):", name)
live = sorted(glob.glob(f"{MY}/NHL_Prediction_System_Claude/01_raw/live/*"))
if live:
    FILES += [os.path.relpath(p, MY) for p in sorted(glob.glob(f"{live[-1]}/*.parquet"))]

os.makedirs(DEST, exist_ok=True)
manifest = []
for rel in FILES:
    data = open(f"{MY}/{rel}", "rb").read()
    stem = rel.replace("/", "__")
    parts = []
    for i in range(0, len(data), PART):
        name = f"core__{stem}.part{i // PART:03d}"
        open(f"{DEST}/{name}", "wb").write(data[i:i + PART])
        parts.append(name)
    manifest.append({"source": rel, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "parts": parts})
json.dump(manifest, open(f"{DEST}/manifest_core_{TODAY}.json", "w"), indent=2)
print(f"{len(manifest)} files, {sum(len(m['parts']) for m in manifest)} parts, "
      f"{sum(m['size_bytes'] for m in manifest) / 1e6:.1f} MB -> {DEST}")
