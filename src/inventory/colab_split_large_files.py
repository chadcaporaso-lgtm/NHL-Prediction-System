"""Colab cell: split large Drive files into parts the audit can download.

The Drive connection used for the audit downloads files up to 10 MB. This
cell copies each large file listed below into 9 MB parts under
NHL_Prediction_System_Claude/01_raw/_transfer/, and writes a manifest with the
SHA-256 of every original file. The audit reassembles the parts and checks
each checksum, so the files examined are byte-for-byte the originals.

The original files are only read, never changed.

Run in Google Colab:
  1. Paste this whole file into one cell.
  2. Run it and approve the Google Drive mount.
"""

import hashlib
import json
import os

from google.colab import drive

drive.mount("/content/drive")

MY_DRIVE = "/content/drive/MyDrive"
DEST = f"{MY_DRIVE}/NHL_Prediction_System_Claude/01_raw/_transfer"
PART_BYTES = 9 * 1024 * 1024

FILES = [
    # v1 production system: feature tables and MoneyPuck source
    "NHL_Prediction_System/data/features/game_features_T24h.parquet",
    "NHL_Prediction_System/data/features/game_features_T60m.parquet",
    "NHL_Prediction_System/data/features/player_features_T60m.parquet",
    "NHL_Prediction_System/data/raw/moneypuck/moneypuck_all_teams_2018_plus.parquet",
    # v2 research pipeline: latest research features and qualified odds
    "NHL_Prediction_System_ChatGPT/data/features/official_richer_incremental/20261103T160656Z_4414a3fb/reconstructed_research_features.parquet",
    "NHL_Prediction_System_ChatGPT/data/normalized/odds_qualified/20261019T011635Z_112d14f7/quote_observations.parquet",
    "NHL_Prediction_System_ChatGPT/data/normalized/odds_qualified/20261019T011635Z_112d14f7/paired_market_observations.parquet",
    "NHL_Prediction_System_ChatGPT/data/normalized/odds_qualified/20261019T011635Z_112d14f7/closing_benchmarks.parquet",
    "NHL_Prediction_System_ChatGPT/data/normalized/moneypuck/20261019T020708Z_32eea7e4/fresh_team_games_reconciled.parquet",
    "NHL_Prediction_System_ChatGPT/data/normalized/moneypuck/20261019T020708Z_32eea7e4/legacy_team_games_reconciled.parquet",
]

os.makedirs(DEST, exist_ok=True)
manifest = []
for rel in FILES:
    src = f"{MY_DRIVE}/{rel}"
    if not os.path.exists(src):
        print("NOT FOUND:", rel)
        manifest.append({"source": rel, "status": "not_found"})
        continue
    data = open(src, "rb").read()
    stem = rel.replace("/", "__")
    parts = []
    for i in range(0, len(data), PART_BYTES):
        name = f"{stem}.part{i // PART_BYTES:03d}"
        with open(f"{DEST}/{name}", "wb") as out:
            out.write(data[i : i + PART_BYTES])
        parts.append(name)
    manifest.append(
        {
            "source": rel,
            "status": "split",
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "parts": parts,
        }
    )
    print(f"{len(parts):3d} parts  {len(data) / 1e6:7.1f} MB  {rel}")

with open(f"{DEST}/transfer_manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)
print("Done. Manifest written to", f"{DEST}/transfer_manifest.json")
