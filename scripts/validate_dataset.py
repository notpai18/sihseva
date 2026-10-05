#!/usr/bin/env python3
"""Post-generation QA for the 'Hey Tara' dataset.

Checks the properties that are easy to break silently and impossible to spot
by listening to a few clips: manifest/disk agreement, degenerate audio,
speaker leakage across splits, and statistical shortcuts (duration, loudness,
or augmentation intensity correlating with the label) that would let a model
score well for the wrong reason.

    .venv/bin/python3 scripts/validate_dataset.py [--data-dir data]

Exits non-zero if any FAIL-level check trips.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

import config

FAILURES: list[str] = []
WARNINGS: list[str] = []


def check(ok: bool, label: str, detail: str = "", warn_only: bool = False) -> None:
    if ok:
        print(f"  PASS  {label}")
        return
    (WARNINGS if warn_only else FAILURES).append(f"{label}: {detail}")
    print(f"  {'WARN' if warn_only else 'FAIL'}  {label}  -- {detail}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--audio-sample", type=int, default=400,
                     help="how many clips to open and inspect acoustically")
    args = ap.parse_args()

    manifest_path = args.data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"No manifest at {manifest_path} -- generate the dataset first.")
    rows = list(csv.DictReader(open(manifest_path)))
    print(f"Manifest: {len(rows)} rows\n")

    # ---------- integrity ----------
    print("Integrity")
    missing = [r["filepath"] for r in rows if not (args.data_dir / r["filepath"]).exists()]
    check(not missing, "every manifest row has a file on disk",
          f"{len(missing)} missing, e.g. {missing[:3]}")

    on_disk = {str(p.relative_to(args.data_dir)) for p in args.data_dir.rglob("*.wav")}
    listed = {r["filepath"] for r in rows}
    orphans = on_disk - listed
    check(not orphans, "no orphan wavs on disk (stale files from an earlier run)",
          f"{len(orphans)} orphans, e.g. {sorted(orphans)[:3]} -- regenerate with --clean")

    dup_paths = [p for p, n in Counter(r["filepath"] for r in rows).items() if n > 1]
    check(not dup_paths, "no duplicate filepaths in manifest",
          f"{len(dup_paths)} duplicated, e.g. {dup_paths[:3]}")

    # ---------- splits ----------
    print("\nSplits and leakage")
    voice_splits = defaultdict(set)
    for r in rows:
        if r["voice_id"]:
            voice_splits[r["voice_id"]].add(r["split"])
    leaked = [v for v, s in voice_splits.items() if len(s) > 1]
    check(not leaked, "speaker-disjoint splits (no voice spans two splits)",
          f"{len(leaked)} voices leak across splits, e.g. {leaked[:3]}")

    labels_per_voice = defaultdict(set)
    for r in rows:
        if r["voice_id"]:
            labels_per_voice[r["voice_id"]].add(r["label"])
    single_class = [v for v, l in labels_per_voice.items() if len(l) == 1]
    check(not single_class, "every voice contributes BOTH positives and negatives",
          f"{len(single_class)} voices are single-class, e.g. {single_class[:3]} "
          f"-- lets the model use speaker identity as a label shortcut")

    per_split = defaultdict(Counter)
    for r in rows:
        per_split[r["split"]][r["label"]] += 1
    for s in ("train", "val", "test"):
        pos, neg = per_split[s]["positive"], per_split[s]["negative"]
        print(f"        {s:5s} pos={pos:5d} neg={neg:5d} ratio 1:{neg/max(pos,1):.2f}")
    check(all(per_split[s]["positive"] > 0 and per_split[s]["negative"] > 0
              for s in ("train", "val", "test")),
          "all three splits contain both classes", "a split is single-class")

    n_test_voices = len({v for v, s in voice_splits.items() if "test" in s})
    check(n_test_voices >= 8, "test split has >=8 distinct speakers",
          f"only {n_test_voices} -- false-accept/reject rates will be very noisy",
          warn_only=True)

    # ---------- shortcut detection ----------
    print("\nShortcut detection (label must not be predictable from metadata alone)")
    durs = defaultdict(list)
    for r in rows:
        durs[r["label"]].append(float(r["duration_sec"]))
    all_durs = np.array([d for v in durs.values() for d in v])
    check(np.ptp(all_durs) < 1e-6, "clip duration is constant across the dataset",
          f"durations vary {all_durs.min():.2f}..{all_durs.max():.2f}s -- "
          f"length alone may separate the classes")

    clean_frac = {}
    for lab in ("positive", "negative"):
        sub = [r for r in rows if r["label"] == lab and r["voice_id"]]
        if sub:
            clean_frac[lab] = sum(json.loads(r["augmentation"]).get("clean", False)
                                   for r in sub) / len(sub)
    if len(clean_frac) == 2:
        gap = abs(clean_frac["positive"] - clean_frac["negative"])
        check(gap < 0.05,
              "unaugmented ('clean') fraction is the same for both classes",
              f"positive {clean_frac['positive']:.0%} vs negative "
              f"{clean_frac['negative']:.0%} -- 'looks clean' becomes a label cue")

    snr = defaultdict(list)
    for r in rows:
        a = json.loads(r["augmentation"])
        if "snr_db" in a:
            snr[r["label"]].append(a["snr_db"])
    if len(snr) == 2:
        gap = abs(np.mean(snr["positive"]) - np.mean(snr["negative"]))
        check(gap < 1.5, "noise SNR distribution is the same for both classes",
              f"positive mean {np.mean(snr['positive']):.1f} dB vs negative "
              f"{np.mean(snr['negative']):.1f} dB -- degradation level cues the label")

    # ---------- acoustics ----------
    print(f"\nAcoustics (inspecting {args.audio_sample} random clips)")
    rng = np.random.default_rng(0)
    sample = [rows[i] for i in rng.choice(len(rows), size=min(args.audio_sample, len(rows)),
                                           replace=False)]
    bad_finite, empty, wrong_sr, wrong_len = [], [], [], []
    peaks_by_label = defaultdict(list)
    expected_n = int(config.TARGET_SAMPLE_RATE * config.CLIP_DURATION_S)
    for r in sample:
        data, sr = sf.read(args.data_dir / r["filepath"])
        if not np.all(np.isfinite(data)):
            bad_finite.append(r["filepath"])
        if sr != config.TARGET_SAMPLE_RATE:
            wrong_sr.append(r["filepath"])
        if len(data) != expected_n:
            wrong_len.append(r["filepath"])
        peak = float(np.max(np.abs(data))) if len(data) else 0.0
        # the silence class is supposed to be inaudible; everything else isn't
        if r["category"] != "silence" and peak < 1e-4:
            empty.append(r["filepath"])
        peaks_by_label[r["label"]].append(peak)

    check(not bad_finite, "no NaN/Inf samples", f"{len(bad_finite)} bad, e.g. {bad_finite[:3]}")
    check(not empty, "no unexpectedly silent clips", f"{len(empty)} silent, e.g. {empty[:3]}")
    check(not wrong_sr, "sample rate is uniform", f"{len(wrong_sr)} wrong, e.g. {wrong_sr[:3]}")
    check(not wrong_len, "sample count matches CLIP_DURATION_S",
          f"{len(wrong_len)} wrong length, e.g. {wrong_len[:3]}")

    all_peaks = np.array([p for v in peaks_by_label.values() for p in v])
    check(all_peaks.std() > 0.02, "loudness actually varies across clips",
          f"peak std={all_peaks.std():.5f} -- every clip is at the same level, so "
          f"gain augmentation is being cancelled somewhere downstream")

    if len(peaks_by_label) == 2:
        gap = abs(np.mean(peaks_by_label["positive"]) - np.mean(peaks_by_label["negative"]))
        check(gap < 0.05, "loudness is not correlated with the label",
              f"mean peak differs by {gap:.3f} between classes")

    # ---------- summary ----------
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
    if WARNINGS:
        print(f"{len(WARNINGS)} warning(s):")
        for w in WARNINGS:
            print(f"  - {w}")
    if not FAILURES:
        print("All hard checks passed.")
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
