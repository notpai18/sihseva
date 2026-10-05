#!/usr/bin/env python3
"""Press Enter, speak, it records exactly 1.5s and saves it -- same format as
the team's existing real recordings. Records a full session (positives, then
negatives) for one person, then automatically folds it into the manifest and
retrains the demo model.

Usage:
    .venv/bin/python3 scripts/live_record.py --member member5
    .venv/bin/python3 scripts/live_record.py --member member5 --n-positive 10 --n-negative 5
    .venv/bin/python3 scripts/live_record.py --member member5 --no-train   # record only, train later
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

import config

SOURCE_ROOT = config.REPO_ROOT / "orginal_recording" / "orginal dataset"
SR = 16000
DURATION_S = 1.5

# Varied "Hey <not Tara>" phrases -- deliberately NOT the same word repeated.
# Fixes a measured bug: real recordings previously had ONLY "Hey Tara"
# (positive) and "Tara" alone (negative), so "Hey" itself became a perfect
# predictor of positive in the real-voice domain. Mixes real competing
# assistants (people say these around devices) with generic names/words.
HEY_OTHER_PROMPTS = [
    "Hey Siri", "Hey Google", "Hey Alexa", "Hey Cortana",
    "Hey Pranav", "Hey Sir", "Hey Buddy", "Hey Guys",
    "Hey Man", "Hey There", "Hey You", "Hey Bro",
]


def record_one(prompt_text: str) -> np.ndarray:
    while True:
        input(f"  [{prompt_text}] Press Enter, then say it immediately...")
        print("  recording...", end="", flush=True)
        audio = sd.rec(int(SR * DURATION_S), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        audio = audio[:, 0]
        peak = float(np.max(np.abs(audio)))
        print(f" done (peak={peak:.3f})")
        if peak < 0.02:
            print("  -> very quiet / looks like nothing was captured. Let's redo this one.")
            continue
        if peak >= 0.999:
            print("  -> possibly clipped (very loud). Keeping it, but consider redoing "
                  "quieter/farther from the mic if this repeats.")
        return audio


def record_batch(member: str, category_dir: str, label_word: str, n: int) -> None:
    out_dir = SOURCE_ROOT / member / category_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(f"{category_dir}_*.wav"))
    start_idx = len(existing)
    print(f"\n=== {member}: recording {n} clips of \"{label_word}\" ===")
    for i in range(n):
        idx = start_idx + i
        audio = record_one(f"{i+1}/{n}: say \"{label_word}\"")
        path = out_dir / f"{category_dir}_{idx:03d}.wav"
        sf.write(str(path), audio, SR, subtype="PCM_16")
        print(f"  saved {path.relative_to(SOURCE_ROOT)}")


def record_batch_varied(member: str, category_dir: str, phrases: list[str], n: int,
                         rng: np.random.Generator) -> None:
    """Like record_batch, but prompts a DIFFERENT phrase each time (cycling
    through `phrases`, reshuffled each pass) instead of repeating one word --
    the whole point of this category is covering many different "Hey X"
    variants, not many takes of the same one."""
    out_dir = SOURCE_ROOT / member / category_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(f"{category_dir}_*.wav"))
    start_idx = len(existing)
    print(f"\n=== {member}: recording {n} clips of varied \"Hey <not Tara>\" phrases ===")
    order: list[str] = []
    while len(order) < n:
        batch = list(phrases)
        rng.shuffle(batch)
        order.extend(batch)
    order = order[:n]
    for i, phrase in enumerate(order):
        idx = start_idx + i
        audio = record_one(f"{i+1}/{n}: say \"{phrase}\"")
        path = out_dir / f"{category_dir}_{idx:03d}.wav"
        sf.write(str(path), audio, SR, subtype="PCM_16")
        print(f"  saved {path.relative_to(SOURCE_ROOT)}  (said: \"{phrase}\")")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--member", required=True, help="e.g. member5")
    p.add_argument("--n-positive", type=int, default=10)
    p.add_argument("--n-negative", type=int, default=5)
    p.add_argument("--n-hey-other", type=int, default=8,
                    help="varied \"Hey <not Tara>\" negatives -- the single most "
                         "valuable category right now, see live_record.py header")
    p.add_argument("--no-train", action="store_true",
                    help="record only; skip auto ingest+retrain at the end")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Recording session for '{args.member}'.")
    print(f"Each clip is exactly {DURATION_S}s, starting the instant you press Enter --"
          f" say the word right away.\n")

    record_batch(args.member, "hey_tara", "Hey Tara", args.n_positive)
    record_batch(args.member, "tara_alone", "Tara", args.n_negative)
    record_batch_varied(args.member, "hey_other", HEY_OTHER_PROMPTS, args.n_hey_other,
                         np.random.default_rng(args.seed))

    print(f"\nDone recording: {args.n_positive} positive + {args.n_negative} negative "
          f"+ {args.n_hey_other} hey_other clips for {args.member}.")

    if args.no_train:
        print("Skipping auto-train (--no-train). Run these when ready:")
        print("  .venv/bin/python3 scripts/ingest_real_recordings.py")
        print("  .venv/bin/python3 scripts/train.py")
        return

    print("\nFolding into the manifest...")
    subprocess.run([sys.executable, str(config.REPO_ROOT / "scripts" / "ingest_real_recordings.py")],
                    check=True)
    print("\nRetraining the demo model (synthetic + all real recordings)...")
    subprocess.run([sys.executable, str(config.REPO_ROOT / "scripts" / "train.py")], check=True)
    print("\nDone. Test it with:")
    print("  .venv/bin/python3 scripts/live_demo.py")


if __name__ == "__main__":
    main()
