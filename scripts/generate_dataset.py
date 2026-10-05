#!/usr/bin/env python3
"""Generate a synthetic 'Hey Tara' wake-word dataset: positives, hard
(confusable) negatives, other-wake-word negatives, general-speech negatives,
and pure-noise/silence negatives -- with audio augmentation and a manifest CSV.

Usage:
    .venv/bin/python3 scripts/generate_dataset.py --help
    .venv/bin/python3 scripts/generate_dataset.py            # run with defaults
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import config
import augment
import noise_gen
import tts_engine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean", action="store_true",
                    help="delete --out-dir before generating, so a rerun (e.g. "
                         "after changing phrase lists or filename scheme) can't "
                         "leave stale files mixed in with the new ones")
    p.add_argument("--positive-variants-per-combo", type=int, default=2,
                    help="copies per (phrase, voice)")
    p.add_argument("--negative-variants-per-combo", type=int, default=2,
                    help="copies per (phrase, voice)")
    p.add_argument("--confusable-per-voice", type=int, default=8,
                    help="confusable phrases sampled per voice (out of %d available)"
                         % len(config.CONFUSABLE_PHRASES))
    p.add_argument("--other-wakeword-per-voice", type=int, default=4,
                    help="other-wakeword phrases sampled per voice (out of %d available)"
                         % len(config.OTHER_WAKEWORDS))
    p.add_argument("--random-speech-per-voice", type=int, default=10,
                    help="random-speech sentences sampled per voice (out of %d available)"
                         % len(config.RANDOM_SPEECH_SENTENCES))
    p.add_argument("--noise-clips-per-kind", type=int, default=40,
                    help="clips generated for each of the pure noise/silence kinds")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    # test-frac is whatever remains
    args = p.parse_args()
    if args.train_frac + args.val_frac >= 1.0:
        p.error("--train-frac + --val-frac must leave a positive test fraction")
    return args


def dedupe_by_phonemes(phrases: list[str], voice, label: str) -> list[str]:
    """Drop phrases that phonemize identically to one already in the list.

    Text variety is not acoustic variety: espeak renders "Hey Kara"/"Hey Cara"
    and "Hey Tora"/"Hey Torah" to the same phonemes, so they'd produce near
    identical audio stored under different filenames -- inflating the apparent
    dataset size with hidden duplicates. Running this at generation time keeps
    the guarantee even after someone edits the lists in config.py.
    """
    seen: dict[str, str] = {}
    kept: list[str] = []
    for phrase in phrases:
        key = "".join(tts_engine.phonemize(voice, phrase))
        if key in seen:
            print(f"  [dedupe] {label}: dropping {phrase!r} "
                  f"(same phonemes as {seen[key]!r}: {key!r})", file=sys.stderr)
            continue
        seen[key] = phrase
        kept.append(phrase)
    return kept


def cross_label_phoneme_check(positives: list[str], negative_lists: dict[str, list[str]],
                               voice) -> None:
    """Abort if any negative phonemizes identically to a positive.

    This is the label-poisoning guard: a phrase that sounds exactly like the
    wake word but is labeled negative teaches the detector to reject its own
    trigger, and nothing downstream would ever surface that as an error.
    """
    pos_phonemes = {"".join(tts_engine.phonemize(voice, p)): p for p in positives}
    for category, phrases in negative_lists.items():
        for phrase in phrases:
            key = "".join(tts_engine.phonemize(voice, phrase))
            if key in pos_phonemes:
                raise SystemExit(
                    f"ABORT: negative {phrase!r} (category {category}) phonemizes to "
                    f"{key!r}, identical to positive {pos_phonemes[key]!r}. "
                    f"Training on this would teach the model to reject the wake word. "
                    f"Fix the phrase lists in config.py."
                )


def phrase_tag(text: str) -> str:
    """Short, stable, collision-resistant tag for filenames. Plain hash() is
    randomized per-process in Python (PYTHONHASHSEED), so it isn't reproducible
    across runs with the same --seed and can silently collide across phrases."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]


def random_augment_chain(audio: np.ndarray, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Apply a randomized augmentation chain; returns (audio, params_used).
    A CLEAN_PROB fraction of calls skip augmentation entirely (raw TTS output),
    decided by a coin flip here rather than by caller-side copy index -- that
    keeps the clean fraction identical between positive and negative classes
    regardless of how many variants-per-combo each class requests, so
    "unaugmented-ness" can't become an accidental label cue."""
    clean = rng.random() < config.CLEAN_PROB
    params: dict = {"clean": clean}
    if clean:
        return audio.astype(np.float32), params

    lo, hi = config.AUGMENT_RANGES["speed_factor"]
    speed = rng.uniform(lo, hi)
    audio = augment.speed_perturb(audio, speed)
    params["speed_factor"] = round(float(speed), 4)

    lo, hi = config.AUGMENT_RANGES["pitch_semitones"]
    semitones = rng.uniform(lo, hi)
    audio = augment.pitch_shift(audio, sr, semitones)
    params["pitch_semitones"] = round(float(semitones), 3)

    lo, hi = config.AUGMENT_RANGES["gain_db"]
    gain = rng.uniform(lo, hi)
    audio = augment.apply_gain_db(audio, gain)
    params["gain_db"] = round(float(gain), 2)

    if rng.random() < 0.7:  # additive noise most of the time
        kind = rng.choice([k for k in config.NOISE_KINDS if k != "silence"])
        snr = float(rng.choice(config.AUGMENT_RANGES["snr_db_choices"]))
        noise = noise_gen.generate(kind, len(audio), sr, rng)
        audio = augment.mix_at_snr(audio, noise, snr, rng)
        params["noise_kind"] = kind
        params["snr_db"] = snr

    if rng.random() < config.AUGMENT_RANGES["reverb_prob"]:
        lo, hi = config.AUGMENT_RANGES["reverb_decay_s"]
        decay = rng.uniform(lo, hi)
        lo, hi = config.AUGMENT_RANGES["reverb_drr_db"]
        drr = rng.uniform(lo, hi)
        audio = augment.add_synthetic_reverb(audio, sr, decay, rng, drr_db=drr)
        params["reverb_decay_s"] = round(float(decay), 3)
        params["reverb_drr_db"] = round(float(drr), 2)

    if rng.random() < config.AUGMENT_RANGES["lowpass_prob"]:
        lo, hi = config.AUGMENT_RANGES["lowpass_cutoff_hz"]
        cutoff = rng.uniform(lo, hi)
        audio = augment.lowpass_filter(audio, sr, cutoff)
        params["lowpass_cutoff_hz"] = round(float(cutoff), 1)

    if rng.random() < config.AUGMENT_RANGES["clip_prob"]:
        drive = rng.uniform(1.2, 2.5)
        audio = augment.soft_clip(audio, drive=drive)
        params["clip_drive"] = round(float(drive), 2)

    lo, hi = config.AUGMENT_RANGES["leading_silence_ms"]
    lead = rng.uniform(lo, hi)
    lo, hi = config.AUGMENT_RANGES["trailing_silence_ms"]
    trail = rng.uniform(lo, hi)
    audio = augment.pad_with_silence(audio, sr, lead, trail)
    params["leading_silence_ms"] = round(float(lead), 1)
    params["trailing_silence_ms"] = round(float(trail), 1)

    return audio.astype(np.float32), params


def assign_voice_splits(voice_ids: list[str], rng: np.random.Generator,
                         train_frac: float, val_frac: float) -> dict[str, str]:
    """Speaker-disjoint split: every sample from a given voice stays in one split."""
    shuffled = list(voice_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    split_of = {}
    for i, vid in enumerate(shuffled):
        if i < n_train:
            split_of[vid] = "train"
        elif i < n_train + n_val:
            split_of[vid] = "val"
        else:
            split_of[vid] = "test"
    return split_of


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


class ManifestWriter:
    FIELDS = ["filepath", "label", "category", "text", "voice_id", "speaker_id",
              "sample_rate", "duration_sec", "split", "augmentation"]

    def __init__(self, path: Path):
        self.file = open(path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.count = 0

    def add(self, **row):
        self.writer.writerow(row)
        self.count += 1

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_dir: Path = args.out_dir
    if args.clean and out_dir.exists():
        print(f"--clean: removing existing {out_dir}", file=sys.stderr)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TTS voice roster...", file=sys.stderr)
    t0 = time.time()
    roster = tts_engine.load_voice_roster(rng)
    print(f"Loaded {len(roster)} voices in {time.time() - t0:.1f}s", file=sys.stderr)

    all_voice_ids = [v.voice_id for v in roster]
    voice_split = assign_voice_splits(all_voice_ids, rng, args.train_frac, args.val_frac)

    # Phrase-list hygiene, before generating anything: drop sound-alike
    # duplicates, then hard-fail if any negative sounds exactly like a positive.
    print("Checking phrase lists...", file=sys.stderr)
    ref_voice = roster[0]
    positives = dedupe_by_phonemes(config.POSITIVE_PHRASES, ref_voice, "positive")
    negatives = {
        "confusable": dedupe_by_phonemes(config.CONFUSABLE_PHRASES, ref_voice, "confusable"),
        "other_wakeword": dedupe_by_phonemes(config.OTHER_WAKEWORDS, ref_voice, "other_wakeword"),
        "random_speech": dedupe_by_phonemes(config.RANDOM_SPEECH_SENTENCES, ref_voice, "random_speech"),
    }
    cross_label_phoneme_check(positives, negatives, ref_voice)
    print(f"  {len(positives)} positive phrases, "
          f"{sum(len(v) for v in negatives.values())} negative phrases (deduped)",
          file=sys.stderr)

    target_sr = config.TARGET_SAMPLE_RATE
    split_choices = ["train", "val", "test"]
    split_probs = [args.train_frac, args.val_frac, 1.0 - args.train_frac - args.val_frac]

    with ManifestWriter(out_dir / "manifest.csv") as manifest:

        def synth_and_save(text: str, voice, category: str, label: str, variants: int,
                            combo_tag: str) -> None:
            for copy_idx in range(variants):
                length_scale = float(rng.uniform(0.9, 1.15))
                noise_scale = float(rng.uniform(0.5, 0.8))
                audio, sr = tts_engine.synthesize(voice, text, length_scale=length_scale,
                                                   noise_scale=noise_scale)
                audio = augment.resample(audio, sr, target_sr)
                audio, aug_params = random_augment_chain(audio, target_sr, rng)
                audio = augment.safety_limit(audio)
                audio = augment.place_in_window(audio, target_sr, config.CLIP_DURATION_S, rng)
                aug_params["length_scale"] = round(length_scale, 3)
                aug_params["noise_scale_tts"] = round(noise_scale, 3)

                fname = f"{category}__{combo_tag}__{copy_idx}.wav"
                rel_path = Path(label) / category / fname
                write_wav(out_dir / rel_path, audio, target_sr)
                manifest.add(
                    filepath=str(rel_path), label=label, category=category, text=text,
                    voice_id=voice.voice_id, speaker_id=voice.speaker_id,
                    sample_rate=target_sr, duration_sec=round(len(audio) / target_sr, 3),
                    split=voice_split[voice.voice_id],
                    augmentation=json.dumps(aug_params),
                )

        # ---------------- Positives ----------------
        # Every voice in the roster produces positives.
        print("Generating positives...", file=sys.stderr)
        n_pos = 0
        for voice in roster:
            for phrase in positives:
                tag = f"{voice.voice_id}_{phrase_tag(phrase)}"
                synth_and_save(phrase, voice, "wake_word", "positive",
                                args.positive_variants_per_combo, tag)
                n_pos += args.positive_variants_per_combo
        print(f"  {n_pos} positive files", file=sys.stderr)

        # ---------------- Word/phrase negatives ----------------
        # The SAME full roster produces negatives too (a random per-voice subset
        # of each phrase list, to control volume) -- every voice contributes to
        # both classes, so the model can't learn "this speaker == positive".
        def gen_negative_category(phrases: list[str], category: str, per_voice: int) -> int:
            count = 0
            k = min(per_voice, len(phrases))
            for voice in roster:
                sampled = rng.choice(phrases, size=k, replace=False)
                for phrase in sampled:
                    tag = f"{voice.voice_id}_{phrase_tag(phrase)}"
                    synth_and_save(phrase, voice, category, "negative",
                                    args.negative_variants_per_combo, tag)
                    count += args.negative_variants_per_combo
            return count

        print("Generating confusable negatives...", file=sys.stderr)
        n_conf = gen_negative_category(negatives["confusable"], "confusable",
                                        args.confusable_per_voice)
        print(f"  {n_conf} confusable files", file=sys.stderr)

        print("Generating other-wakeword negatives...", file=sys.stderr)
        n_oww = gen_negative_category(negatives["other_wakeword"], "other_wakeword",
                                       args.other_wakeword_per_voice)
        print(f"  {n_oww} other_wakeword files", file=sys.stderr)

        print("Generating random-speech negatives...", file=sys.stderr)
        n_rand = gen_negative_category(negatives["random_speech"], "random_speech",
                                        args.random_speech_per_voice)
        print(f"  {n_rand} random_speech files", file=sys.stderr)

        # ---------------- Noise / silence negatives ----------------
        print("Generating noise/silence negatives...", file=sys.stderr)
        n_noise = 0
        for kind in config.NOISE_KINDS:
            for i in range(args.noise_clips_per_kind):
                n_samples = int(target_sr * config.CLIP_DURATION_S)
                audio = noise_gen.generate(kind, n_samples, target_sr, rng)
                if kind != "silence":
                    audio = augment.normalize_peak(audio, peak=rng.uniform(0.2, 0.9))
                # silence sets its own (deliberately very low) level -- renormalizing
                # it up to an audible peak would defeat the point of the class.
                category = "silence" if kind == "silence" else "noise"
                fname = f"{kind}__{i:04d}.wav"
                rel_path = Path("negative") / category / fname
                write_wav(out_dir / rel_path, audio, target_sr)
                split = rng.choice(split_choices, p=split_probs)
                manifest.add(
                    filepath=str(rel_path), label="negative", category=category,
                    text="", voice_id="", speaker_id="", sample_rate=target_sr,
                    duration_sec=config.CLIP_DURATION_S, split=split,
                    augmentation=json.dumps({"noise_kind": kind}),
                )
                n_noise += 1
        print(f"  {n_noise} noise/silence files", file=sys.stderr)

        total = n_pos + n_conf + n_oww + n_rand + n_noise

    print(f"\nDone. {total} total clips written under {out_dir}", file=sys.stderr)
    print(f"Manifest: {out_dir / 'manifest.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()
