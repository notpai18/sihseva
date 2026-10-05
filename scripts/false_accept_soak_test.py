#!/usr/bin/env python3
"""False-accept soak test: run the wake-word detector over hours of REAL,
non-synthetic speech that was never trained on, and count how often it fires.

Every other accuracy number in this project (synthetic test split, real-eval,
LOSO) comes from short clips that are either the wake word or a deliberately
chosen negative phrase. None of them answer the question a deployed device
actually lives or dies on: "left running for hours next to a podcast, a TV,
or a conversation, how many times does it go off by mistake?" README.md's
caveat #5 calls this out as the single highest-value gap in the dataset --
this script is what closes it, using LibriSpeech (real audiobook narration,
public domain, CC BY 4.0) as a stand-in for "real speech the model has never
heard," rather than more synthetic TTS.

This deliberately reuses the EXACT SAME inference path as scripts/live_demo.py
(IncrementalMelExtractor, loudness_gain/apply_gain_to_log_mel, confirm-polls,
cooldown) rather than a simplified batch classifier -- a false-accept rate
measured on a different code path than the one actually deployed would not
mean anything.

Usage:
    .venv/bin/python3 scripts/false_accept_soak_test.py /path/to/librispeech/dev-clean
    .venv/bin/python3 scripts/false_accept_soak_test.py /path/to/audio --threshold 0.6
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import features
import live_demo
import model as model_module
from train import CHECKPOINT_PATH, load_clip

AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg"}


def find_audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def load_full_clip(path: Path, target_sr: int) -> np.ndarray:
    """Like train.load_clip, but returns the FULL file at its natural length
    (no pad/crop to a fixed window) -- this script simulates continuous
    streaming over a whole file, not isolated fixed-duration clips."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g)
    return audio.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio_root", type=str, help="directory of real speech audio (searched recursively)")
    p.add_argument("--threshold", type=float, default=None,
                    help="defaults to the checkpoint's recommended (balanced) threshold, "
                         "same as scripts/live_demo.py's default")
    p.add_argument("--poll-interval-s", type=float, default=0.2)
    p.add_argument("--confirm-polls", type=int, default=2)
    p.add_argument("--cooldown-s", type=float, default=1.5)
    p.add_argument("--max-files", type=int, default=None, help="cap for a quick partial run")
    p.add_argument("--report-every-s", type=float, default=300.0,
                    help="print a running false-accepts-per-hour estimate every N seconds "
                         "of audio processed, so a partial/interrupted run still tells you something")
    args = p.parse_args()

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    net = model_module.TinyWakeWordNet(n_mels=ckpt["n_mels"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    sr = ckpt["target_sr"]
    threshold = args.threshold if args.threshold is not None else ckpt.get("recommended_threshold", 0.5)
    _, n_frames = features.feature_shape(sr, ckpt["clip_duration_s"])

    files = find_audio_files(Path(args.audio_root))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No audio files ({sorted(AUDIO_EXTS)}) found under {args.audio_root}")

    print(f"Model: {model_module.count_params(net)} params, threshold={threshold:.3f}, "
          f"confirm_polls={args.confirm_polls}, cooldown_s={args.cooldown_s}")
    print(f"Scanning {len(files)} real audio files under {args.audio_root}\n")

    total_duration_s = 0.0
    total_detections = 0
    detection_log: list[tuple[str, float, float]] = []  # (filepath, offset_s, confidence)
    poll_samples = int(sr * args.poll_interval_s)
    t_start = time.time()
    next_report_at = args.report_every_s

    # ONE buffer, extractor, and cooldown clock for the WHOLE corpus, not one
    # per file. A real device resets from silence exactly once, at power-on
    # -- resetting per file would instead replay that one-time "buffer still
    # partly zero-padded" transient at the start of all 2703 files, which
    # measurably inflates false accepts (confirmed: an early version of this
    # script that reset per file found detections spiking specifically in the
    # 2.0-4.0s window of each file, i.e. exactly when the buffer first
    # finishes filling with real audio -- a transient that only happens once
    # on a real always-on device, not 2703 times).
    mel_extractor = features.IncrementalMelExtractor(sr=sr, n_frames=n_frames)
    ring = np.zeros(int(sr * ckpt["clip_duration_s"]), dtype=np.float32)
    consecutive_hits = 0
    last_trigger_time = -1e9  # global timeline, seconds

    for fi, path in enumerate(files):
        try:
            audio = load_full_clip(path, sr)
        except Exception as e:
            print(f"  [skip] {path.name}: {e}")
            continue

        pos = 0
        file_duration_s = len(audio) / sr

        while pos < len(audio):
            chunk = audio[pos:pos + poll_samples]
            pos += poll_samples
            offset_s = pos / sr                       # for the printed filepath:offset label
            global_t = total_duration_s + offset_s     # for cooldown/rate, continuous across files
            ring = np.concatenate([ring[len(chunk):], chunk]) if len(chunk) < len(ring) else chunk[-len(ring):]
            mel_extractor.push(chunk)

            peak_raw = float(np.max(np.abs(ring)))
            gain = live_demo.loudness_gain(peak_raw)
            mel = live_demo.apply_gain_to_log_mel(mel_extractor.frames, gain)
            mel_norm = (mel - ckpt["feat_mean"]) / ckpt["feat_std"]
            x = torch.from_numpy(mel_norm).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prob = torch.sigmoid(net(x)).item()

            if prob >= threshold:
                consecutive_hits += 1
            else:
                consecutive_hits = 0

            if (consecutive_hits >= args.confirm_polls
                    and (global_t - last_trigger_time) > args.cooldown_s):
                total_detections += 1
                last_trigger_time = global_t
                consecutive_hits = 0
                detection_log.append((str(path.relative_to(args.audio_root)), offset_s, prob))
                elapsed_hours = global_t / 3600.0
                rate = total_detections / elapsed_hours if elapsed_hours > 0 else float("nan")
                print(f"  [FALSE ACCEPT #{total_detections}] {path.relative_to(args.audio_root)} "
                      f"@ {offset_s:.1f}s (conf={prob:.3f})  "
                      f"-- running rate: {rate:.2f}/hour over {elapsed_hours:.2f}h so far")

        total_duration_s += file_duration_s
        if total_duration_s >= next_report_at:
            hours = total_duration_s / 3600.0
            rate = total_detections / hours if hours > 0 else float("nan")
            wall = time.time() - t_start
            print(f"-- progress: {hours:.2f}h of real audio processed "
                  f"({fi+1}/{len(files)} files, {wall:.0f}s wall time) -- "
                  f"{total_detections} false accepts so far = {rate:.2f}/hour --\n")
            next_report_at += args.report_every_s

    hours = total_duration_s / 3600.0
    rate = total_detections / hours if hours > 0 else float("nan")
    print(f"\n=== Done: {hours:.2f} hours of real speech ({len(files)} files) ===")
    print(f"Total false accepts: {total_detections}  ->  {rate:.2f} false accepts / hour "
          f"at threshold={threshold:.3f}")
    if detection_log:
        print("\nAll false accepts (file, offset, confidence):")
        for fp, off, conf in detection_log:
            print(f"  {fp} @ {off:.1f}s  conf={conf:.3f}")


if __name__ == "__main__":
    main()
