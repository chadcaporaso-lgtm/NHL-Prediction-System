"""Rebuild the core inputs in a new session from Drive downloads.

Claude downloads each part of a core bundle (colab_bundle_core_inputs.py)
with the Google Drive connector. Large results are saved by the harness as
JSON files under ~/.claude/projects/*/tool-results/; small ones appear only in
the session transcript (~/.claude/projects/*.jsonl). This script searches both
for parts by file title, reassembles each file, checks it against the
manifest's SHA-256, and writes it to data/drive/<original Drive path>.
A file goes to data/live/<date>/ when it came from 01_raw/live/<date>/.

Usage: python3 src/inventory/restore_core_inputs.py <manifest.json>
"""

from __future__ import annotations

import base64
import glob
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
manifest = json.load(open(sys.argv[1]))
want = {p for m in manifest for p in m["parts"]}
found: dict[str, bytes] = {}
pat = re.compile(r'\{\\?"content\\?":\\?"([A-Za-z0-9+/=]+)\\?",\\?"id\\?":\\?"[^"\\]+\\?",\\?"mimeType\\?":\\?"[^"\\]+\\?",\\?"title\\?":\\?"([^"\\]+)')
home = os.path.expanduser("~/.claude/projects")
for path in glob.glob(f"{home}/*/tool-results/*.txt") + glob.glob(f"{home}/*.jsonl") + glob.glob(f"{home}/*/*.jsonl"):
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        continue
    if not any(w in text for w in list(want)[:1]) and "core__" not in text:
        continue
    try:
        d = json.loads(text)
        if d.get("title") in want:
            found[d["title"]] = base64.b64decode(d["content"])
            continue
    except (ValueError, AttributeError):
        pass
    for m in pat.finditer(text):
        if m.group(2) in want:
            found[m.group(2)] = base64.b64decode(m.group(1))

ok = bad = missing = 0
for m in manifest:
    if not all(p in found for p in m["parts"]):
        print("MISSING parts for", m["source"], [p for p in m["parts"] if p not in found])
        missing += 1
        continue
    data = b"".join(found[p] for p in m["parts"])
    if hashlib.sha256(data).hexdigest() != m["sha256"]:
        print("CHECKSUM MISMATCH", m["source"])
        bad += 1
        continue
    live = re.search(r"01_raw/live/(\d{4}-\d{2}-\d{2})/(.+)$", m["source"])
    out = os.path.join(ROOT, f"data/live/{live.group(1)}/{live.group(2)}") if live else os.path.join(ROOT, "data/drive", m["source"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "wb").write(data)
    ok += 1
print(f"verified and written: {ok}, checksum mismatches: {bad}, missing: {missing}")
