#!/usr/bin/env python3
"""Build a manifest.csv for the team's real "Hey Tara" recordings, in the
same schema as the synthetic dataset's manifest, WITHOUT copying or modifying
the source audio -- this stays the team's raw, read-only ground truth.

Auto-discovers member*/ folders so new contributors (member5, member6, ...)
are picked up automatically. Within each member's folder, classifies every
directory that directly contains .wav files as positive ("hey tara" folder,
any naming: "hey tara", "Hey_Tara", "heytara", ...) or negative ("tara_alone"
folder, any naming, including nested wrapper folders from zip re-extracts)
by normalizing the folder name -- no per-member special-casing needed, so it
keeps working as folder-naming conventions keep drifting between members.

Flags likely-clipped files automatically (a run of >50 consecutive samples at
the digital ceiling), rather than hardcoding which files are bad -- so a
re-recorded file clears its own flag on the next run, and a new problem in a
new member's recordings gets caught the same way.

Usage:
    .venv/bin/python3 scripts/ingest_real_recordings.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import config

SOURCE_ROOT = config.REPO_ROOT / "orginal_recording" / "orginal dataset"
OUT_MANIFEST = config.REPO_ROOT / "orginal_recording" / "manifest.csv"

FIELDS = ["filepath", "label", "category", "text", "voice_id", "speaker_id",
          "sample_rate", "duration_sec", "split", "augmentation", "quality_flag"]


def normalize(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def classify_folder(name: str) -> tuple[str, str] | None:
    """Returns (label, category) or None if the folder name doesn't match a
    known pattern. Checked in order so "heyother" can't be mistaken for
    "heytara" (both start with "hey")."""
    n = normalize(name)
    if "heyother" in n:
        # "Hey <anything but Tara>" -- e.g. "Hey Siri", "Hey Pranav", "Hey Sir".
        # Fixes a real measured bug: real recordings previously had ONLY
        # "Hey Tara" (positive) and "Tara" alone (negative), so in the
        # real-voice domain "Hey" itself became a perfect predictor of
        # positive. This category is what teaches the model that "Hey" alone
        # isn't enough -- it needs to be specifically followed by "Tara".
        return "negative", "hey_other"
    if "heytara" in n:
        return "positive", "wake_word"
    if "taraalone" in n:
        return "negative", "confusable"
    return None


def clipping_run(data: np.ndarray, ceiling: float = 0.999) -> int:
    at_ceiling = np.abs(data) >= ceiling
    if not at_ceiling.any():
        return 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], at_ceiling.view(np.int8), [0]))))
    return int((idx[1::2] - idx[0::2]).max())


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"No source folder at {SOURCE_ROOT}")

    member_dirs = sorted(
        d for d in SOURCE_ROOT.iterdir()
        if d.is_dir() and re.match(r"member\d+$", d.name, re.IGNORECASE)
    )
    if not member_dirs:
        raise SystemExit(f"No member*/ folders found under {SOURCE_ROOT}")

    rows = []
    skipped_folders = []
    for member_dir in member_dirs:
        member_id = member_dir.name.lower()
        # every directory anywhere under this member that directly holds .wav
        # files is a class folder, classified by ITS OWN name -- this handles
        # member2's nested "tara_alone (1)/tara_alone/" wrapper for free, since
        # the wrapper itself holds no .wav files and is never classified.
        wav_dirs = sorted({p.parent for p in member_dir.rglob("*.wav")})
        for d in wav_dirs:
            classified = classify_folder(d.name)
            if classified is None:
                skipped_folders.append(str(d.relative_to(SOURCE_ROOT)))
                continue
            label, category = classified
            text = {"wake_word": "Hey Tara", "confusable": "Tara",
                     "hey_other": "Hey (other)"}[category]
            for wav in sorted(d.glob("*.wav")):
                data, sr = sf.read(str(wav))
                if data.ndim > 1:
                    data = data.mean(axis=1)
                run = clipping_run(data)
                quality_flag = "clipped" if run > 50 else ""
                rows.append({
                    "filepath": str(wav.relative_to(SOURCE_ROOT)),
                    "label": label,
                    "category": category,
                    "text": text,
                    "voice_id": member_id,
                    "speaker_id": "",
                    "sample_rate": sr,
                    "duration_sec": round(len(data) / sr, 3),
                    "split": "real_eval",
                    "augmentation": "{}",
                    "quality_flag": quality_flag,
                })

    if skipped_folders:
        print("Skipped folders (name didn't match 'hey tara' or 'tara alone' pattern):",
              file=sys.stderr)
        for f in skipped_folders:
            print(f"  {f}", file=sys.stderr)

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # ---------------- summary ----------------
    from collections import Counter
    print(f"\n{len(rows)} real recordings ingested from {len(member_dirs)} member(s)")
    print(f"Manifest: {OUT_MANIFEST}\n")

    by_member_label = Counter((r["voice_id"], r["label"]) for r in rows)
    print(f"{'member':10s} {'positive':>9s} {'negative':>9s}")
    for m in sorted({r['voice_id'] for r in rows}):
        print(f"{m:10s} {by_member_label[(m,'positive')]:9d} {by_member_label[(m,'negative')]:9d}")

    flagged = [r for r in rows if r["quality_flag"]]
    print(f"\nQuality flags: {len(flagged)} file(s)")
    for r in flagged:
        print(f"  [{r['quality_flag']}] {r['filepath']}  (member={r['voice_id']})")
    if flagged:
        print("\n  These are NOT excluded from the manifest, just flagged --")
        print("  filter on quality_flag=='' downstream, or re-record and rerun this script.")


if __name__ == "__main__":
    main()
