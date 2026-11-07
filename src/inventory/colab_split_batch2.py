"""Colab cell: split the files from the 2026-11-07 gap fill into 3 MB parts.

Same method as colab_split_large_files.py, written to a separate folder so
nothing collides with the first transfer:
NHL_Prediction_System_Claude/01_raw/_transfer_3mb/batch2/ with
transfer_manifest_batch2.json (SHA-256 of every original file).
The originals are only read, never changed.
"""

import hashlib
import json
import os

from google.colab import drive

drive.mount("/content/drive")

MY_DRIVE = "/content/drive/MyDrive"
BASE = "NHL_Prediction_System_Claude/01_raw"
DEST = f"{MY_DRIVE}/{BASE}/_transfer_3mb/batch2"
PART_BYTES = 3 * 1024 * 1024

FILES = [f"{BASE}/nhl_api_refetch/skater_{r}_{s}.parquet"
         for s in ("20192020", "20202021", "20262027") for r in ("summary", "timeonice")]
FILES.append(f"{BASE}/moneypuck/moneypuck_all_teams_2018_plus_2026-11-07.parquet")

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
    manifest.append({"source": rel, "status": "split", "size_bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest(), "parts": parts})
    print(f"{len(parts):3d} parts  {len(data) / 1e6:7.1f} MB  {rel}")

with open(f"{DEST}/transfer_manifest_batch2.json", "w") as fh:
    json.dump(manifest, fh, indent=2)
print("Done. Total parts:", sum(len(m.get("parts", [])) for m in manifest))
