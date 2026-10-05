#!/usr/bin/env python3
"""Train the tiny wake-word CNN on the synthetic dataset, evaluate on synthetic
test AND the real recordings (the trustworthy number -- see README), and save
a checkpoint for scripts/infer.py to load.

Two real-recording numbers are reported and saved, and they answer different
questions -- don't conflate them:
  - "demo-voice fit" (real_eval_*): the deployed checkpoint trained on EVERY
    real speaker. Answers "does it work for the people it has heard" -- the
    number that matters for tomorrow's live demo.
  - "LOSO" (loso_*): leave-one-speaker-out cross-validation, a SEPARATE set of
    throwaway models each trained with one member held out entirely and
    evaluated only on that unseen member. Answers "does it work for someone
    new" -- the only honest generalization number in this file.

Usage:
    .venv/bin/python3 scripts/train.py
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

import augment
import config
import features
import model as model_module
import noise_gen

REAL_ROOT = config.REPO_ROOT / "orginal_recording" / "orginal dataset"
REAL_MANIFEST = config.REPO_ROOT / "orginal_recording" / "manifest.csv"
CHECKPOINT_PATH = config.REPO_ROOT / "scripts" / "wakeword_model.pt"


def load_clip(path: Path, target_sr: int, target_duration: float) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        from math import gcd
        g = gcd(sr, target_sr)
        from scipy.signal import resample_poly
        audio = resample_poly(audio, target_sr // g, sr // g)
    n_target = int(target_sr * target_duration)
    if len(audio) < n_target:
        pad = n_target - len(audio)
        audio = np.pad(audio, (pad // 2, pad - pad // 2))
    elif len(audio) > n_target:
        start = (len(audio) - n_target) // 2
        audio = audio[start:start + n_target]
    return audio.astype(np.float32)


def augment_real_clip(audio: np.ndarray, sr: int, duration: float,
                       rng: np.random.Generator) -> np.ndarray:
    """One randomized augmented copy of a real clip -- reuses the same
    building blocks as the synthetic pipeline (scripts/augment.py) so that
    oversampling a small real set produces genuine acoustic variation instead
    of N identical copies of the same recording."""
    speed = rng.uniform(0.9, 1.1)
    audio = augment.speed_perturb(audio, speed)
    semitones = rng.uniform(-1.5, 1.5)
    audio = augment.pitch_shift(audio, sr, semitones)
    audio = augment.apply_gain_db(audio, rng.uniform(-8.0, 6.0))
    if rng.random() < 0.6:
        kind = rng.choice(["white", "pink", "babble", "hum"])
        snr = float(rng.choice([8, 12, 16, 20]))
        noise = noise_gen.generate(kind, len(audio), sr, rng)
        audio = augment.mix_at_snr(audio, noise, snr, rng)
    audio = augment.safety_limit(audio)
    audio = augment.place_in_window(audio, sr, duration, rng)
    return audio


def make_partial_crop(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift a complete utterance so most of it falls OUTSIDE the window,
    leaving only a fragment at the edge -- labeled NEGATIVE.

    In streaming inference the sliding window spends most of its time seeing
    partial words ("Hey Ta...", "...ara"), but every training clip contained
    the complete phrase, so the model had never seen a fragment and produced
    arbitrary (often high) confidence on them -- measured at 0.62-0.66 for
    fragments holding only 13-40% of the word. Teaching fragments as negative
    is what makes the detector require the WHOLE phrase before firing.
    """
    n = len(audio)
    shift = int(n * rng.uniform(0.55, 0.85))
    out = np.zeros_like(audio)
    if rng.random() < 0.5:
        out[: n - shift] = audio[shift:]
    else:
        out[shift:] = audio[: n - shift]
    return out


def build_real_speech_negatives(root: Path, n_clips: int, sr: int, duration: float,
                                 rng: np.random.Generator,
                                 held_out_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Random windows of REAL (non-synthetic) speech -- e.g. a LibriSpeech
    split -- labeled negative. Closes the gap README.md's caveat 5 predicted:
    every other negative category in this pipeline (confusable, other_wakeword,
    random_speech) is synthetic TTS, so the model had never been taught to
    reject what actual human narration or conversation sounds like.

    Measured effect of that gap before this fix: running this checkpoint
    continuously against 5.4 hours of real audiobook speech it had never
    heard produced ~120 false accepts/hour -- about one false trigger every
    30 seconds, confirmed on a CONTINUOUS stream (not per-clip resets, which
    were checked separately and ruled out as the cause). Far too high to
    deploy; this is the single highest-value fix available given real speech
    data is now on hand.

    Speaker-disjoint by construction, the same principle the synthetic
    dataset's train/val/test split already uses: `root`'s immediate
    subdirectories are treated as speaker IDs (matches LibriSpeech's
    <speaker>/<chapter>/*.flac layout), shuffled and split 80/20. Only the
    80% pool contributes training clips -- the other 20% is returned as
    `held_out_speaker_dirs` so the caller can re-run the false-accept soak
    test against speech NONE of this run's training data came from, rather
    than re-testing on the same speakers just fixed.
    """
    speaker_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not speaker_dirs:
        raise ValueError(f"no speaker subdirectories found under {root}")
    shuffled = list(speaker_dirs)
    rng.shuffle(shuffled)
    n_held_out = max(1, int(round(len(shuffled) * held_out_frac)))
    held_out_dirs = shuffled[:n_held_out]
    train_dirs = shuffled[n_held_out:]

    audio_files = [p for d in train_dirs for p in d.rglob("*.flac")]
    if not audio_files:
        raise ValueError(f"no .flac files found under the training-pool speakers in {root}")

    t0 = time.time()
    n_samples = int(sr * duration)
    feats = []
    for _ in range(n_clips):
        path = audio_files[rng.integers(len(audio_files))]
        audio, file_sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(file_sr, sr)
            audio = resample_poly(audio, sr // g, file_sr // g)
        if len(audio) <= n_samples:
            clip = np.pad(audio, (0, n_samples - len(audio)))
        else:
            start = int(rng.integers(0, len(audio) - n_samples + 1))
            clip = audio[start:start + n_samples]
        feats.append(features.log_mel_spectrogram(clip.astype(np.float32), sr))
    print(f"  real-speech negatives: {n_clips} clips sampled from {len(train_dirs)} speakers "
          f"({[d.name for d in held_out_dirs]} held out for post-fix testing) "
          f"in {time.time()-t0:.0f}s")
    return (np.stack(feats).astype(np.float32), np.zeros(n_clips, dtype=np.float32),
            [d.name for d in held_out_dirs])


def build_noise_negatives(n_clips: int, sr: int, duration: float,
                           rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Cheap, generated-on-the-fly non-speech negatives: silence at many levels
    (including true digital zero) and every noise bed we can synthesize.

    The trained model scored 0.397 on pure silence and 0.548 on -30 dBFS room
    tone -- above the firing threshold -- because noise/silence was only ~2.5%
    of the training set. This class costs nothing to generate, so there is no
    reason for it to be rare.
    """
    t0 = time.time()
    feats = []
    n_samples = int(sr * duration)
    kinds = [k for k in config.NOISE_KINDS if k != "silence"]
    for i in range(n_clips):
        r = rng.random()
        if r < 0.15:
            audio = np.zeros(n_samples, dtype=np.float32)           # true digital zero
        elif r < 0.45:
            floor_db = rng.uniform(-70, -25)                         # room tone, wide level range
            audio = (rng.normal(0, 1, n_samples) * (10 ** (floor_db / 20))).astype(np.float32)
        else:
            kind = str(rng.choice(kinds))
            audio = noise_gen.generate(kind, n_samples, sr, rng)
            audio = augment.apply_gain_db(audio, rng.uniform(-45, -6))
        feats.append(features.log_mel_spectrogram(augment.safety_limit(audio), sr))
    print(f"  noise/silence negatives: {n_clips} generated in {time.time()-t0:.0f}s")
    return np.stack(feats).astype(np.float32), np.zeros(n_clips, dtype=np.float32)


def build_augmented_real_features(rows: list[dict], root: Path, target_sr: int,
                                   target_duration: float, n_repeats: int,
                                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n_repeats DIFFERENT augmented versions of each real clip (not exact
    duplicates), so the small real set contributes actual acoustic range."""
    t0 = time.time()
    feats, labels = [], []
    n_partial = 0
    for r in rows:
        base_audio = load_clip(root / r["filepath"], target_sr, target_duration)
        label = 1.0 if r["label"] == "positive" else 0.0
        for _ in range(n_repeats):
            audio = augment_real_clip(base_audio.copy(), target_sr, target_duration, rng)
            feats.append(features.log_mel_spectrogram(audio, target_sr))
            labels.append(label)
        # for positives, also emit edge-truncated fragments as NEGATIVES so the
        # model learns to require the complete phrase, not just a piece of it
        if label == 1.0:
            for _ in range(max(1, n_repeats // 3)):
                audio = augment_real_clip(base_audio.copy(), target_sr, target_duration, rng)
                feats.append(features.log_mel_spectrogram(make_partial_crop(audio, rng), target_sr))
                labels.append(0.0)
                n_partial += 1
    print(f"  real (augmented): {len(rows)} clips x {n_repeats} repeats "
          f"(+{n_partial} partial-word negatives) = {len(feats)} in {time.time()-t0:.0f}s")
    return np.stack(feats).astype(np.float32), np.array(labels, dtype=np.float32)


def build_partial_negatives(rows: list[dict], root: Path, target_sr: int,
                             target_duration: float, n_per_clip: int,
                             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Edge-truncated fragments of synthetic positives, labeled negative."""
    t0 = time.time()
    feats = []
    for r in rows:
        base = load_clip(root / r["filepath"], target_sr, target_duration)
        for _ in range(n_per_clip):
            feats.append(features.log_mel_spectrogram(make_partial_crop(base, rng), target_sr))
    print(f"  synthetic partial-word negatives: {len(feats)} in {time.time()-t0:.0f}s")
    return np.stack(feats).astype(np.float32), np.zeros(len(feats), dtype=np.float32)


def spec_augment(x: torch.Tensor, n_freq_masks: int = 1, freq_mask_max: int = 6,
                  n_time_masks: int = 1, time_mask_max: int = 20) -> torch.Tensor:
    """Randomly zero out small time/frequency bands of a batch of log-mel
    spectrograms (x: (batch, 1, n_mels, n_frames)), applied fresh each batch
    during training only. Standard regularizer for small speech datasets --
    forces the model to not over-rely on any single narrow time/frequency
    region, which is exactly the overfitting risk with only ~4 real speakers."""
    x = x.clone()
    n_mels, n_frames = x.shape[-2], x.shape[-1]
    for b in range(x.shape[0]):
        for _ in range(n_freq_masks):
            width = int(torch.randint(0, freq_mask_max + 1, (1,)))
            if width == 0:
                continue
            start = int(torch.randint(0, max(n_mels - width, 1), (1,)))
            x[b, :, start:start + width, :] = 0.0
        for _ in range(n_time_masks):
            width = int(torch.randint(0, time_mask_max + 1, (1,)))
            if width == 0:
                continue
            start = int(torch.randint(0, max(n_frames - width, 1), (1,)))
            x[b, :, :, start:start + width] = 0.0
    return x


def build_features(rows: list[dict], root: Path, target_sr: int, target_duration: float,
                    label_str: str) -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    feats, labels = [], []
    for i, r in enumerate(rows):
        audio = load_clip(root / r["filepath"], target_sr, target_duration)
        mel = features.log_mel_spectrogram(audio, target_sr)
        feats.append(mel)
        labels.append(1.0 if r["label"] == "positive" else 0.0)
        if (i + 1) % 1000 == 0:
            print(f"  {label_str}: {i+1}/{len(rows)} ({time.time()-t0:.0f}s elapsed)")
    print(f"  {label_str}: {len(rows)} clips featurized in {time.time()-t0:.0f}s")
    return np.stack(feats).astype(np.float32), np.array(labels, dtype=np.float32)


def evaluate(net: nn.Module, X: torch.Tensor, y: np.ndarray, name: str) -> dict:
    net.eval()
    with torch.no_grad():
        logits = net(X)
        probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(np.float32)
    y = np.asarray(y)
    acc = float((preds == y).mean())
    pos_mask, neg_mask = y == 1, y == 0
    frr = float((preds[pos_mask] == 0).mean()) if pos_mask.any() else float("nan")
    far = float((preds[neg_mask] == 1).mean()) if neg_mask.any() else float("nan")
    print(f"  [{name}] n={len(y)}  acc={acc:.4f}  FRR={frr:.4f} (miss real wake word)  "
          f"FAR={far:.4f} (false trigger)")
    return {"n": len(y), "acc": acc, "frr": frr, "far": far}


def train_net(Xt: torch.Tensor, yt: torch.Tensor, Xv: torch.Tensor, y_val: np.ndarray,
              n_mels: int, epochs: int, seed: int = 0, batch_size: int = 64,
              lr: float = 1e-3, pos_weight: float = 0.5,
              verbose: bool = True) -> tuple[nn.Module, float]:
    """Train one TinyWakeWordNet to convergence, keeping the best-val-acc
    epoch's weights (val accuracy bounces several points epoch to epoch, so
    the last epoch is not necessarily the best one).

    Shared by the demo checkpoint's training run and every leave-one-speaker-
    out fold below -- both use identical architecture, optimizer, and
    schedule, and differ only in which data they were trained on. `seed`
    controls only the model's weight initialization (torch.manual_seed);
    dataset construction/augmentation randomness is controlled separately by
    whatever np.random.Generator the caller used to build Xt/yt.
    """
    torch.manual_seed(seed)
    net = model_module.TinyWakeWordNet(n_mels=n_mels)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # Penalize false accepts harder than false rejects. A wake word that
    # occasionally needs a second try is a minor annoyance; one that fires at
    # random speech is unusable -- and the measured failure was firing too
    # readily (FAR 0.10-0.20, and room tone alone crossing threshold).
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    n = Xt.shape[0]
    best_val_acc, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            xb = spec_augment(xb)
            opt.zero_grad()
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(net(Xv)).numpy()
        val_acc = float(((val_probs >= 0.5).astype(np.float32) == y_val).mean())
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            best_epoch = epoch + 1
        if verbose:
            print(f"  epoch {epoch+1:2d}/{epochs}  train_loss={total_loss/n:.4f}  val_acc={val_acc:.4f}")
    if verbose:
        print(f"Training done in {time.time()-t0:.0f}s")
    if best_state is not None:
        net.load_state_dict(best_state)
        if verbose:
            print(f"Restored best checkpoint (epoch {best_epoch}, val_acc={best_val_acc:.4f})")
    return net, best_val_acc


def run_loso_eval(X_train_base: np.ndarray, y_train_base: np.ndarray,
                   Xv: torch.Tensor, y_val: np.ndarray, real_rows: list[dict],
                   mean: float, std: float, oversample: int, epochs: int) -> dict:
    """Leave-one-speaker-out: for each real member, train a FRESH, throwaway
    model on synthetic data + generated negatives + every OTHER member's real
    recordings, then evaluate only on that member's clips, which this fold's
    model never saw in any form. This is the only number in this file that
    measures generalization to an unseen voice.

    The main demo checkpoint (see main()) deliberately folds every member
    into training, which is the right call for a live demo but means its
    real_eval_* numbers are a "fit to known voices" claim, not generalization
    -- do not present those as a generalization number. This function exists
    so there is always an honest number to put next to it.

    Each fold reuses X_train_base/y_train_base (synthetic + partial + noise
    negatives, identical across folds) and only rebuilds the real-recording
    portion, so this costs ~one extra training run per member, not six full
    pipeline reruns. Fold models are discarded after evaluation -- only the
    numbers are kept.
    """
    members = sorted({r["voice_id"] for r in real_rows})
    print(f"\n=== Leave-one-speaker-out generalization check ({len(members)} folds) ===")
    per_member: dict[str, dict] = {}
    for i, held_out in enumerate(members):
        fold_train_rows = [r for r in real_rows if r["voice_id"] != held_out]
        held_out_rows = [r for r in real_rows if r["voice_id"] == held_out]
        rng = np.random.default_rng(100 + i)  # deterministic per member, not hash()-based

        print(f"\n  -- fold: holding out {held_out} ({len(held_out_rows)} clips) --")
        X_real_fold, y_real_fold = build_augmented_real_features(
            fold_train_rows, REAL_ROOT, 16000, config.CLIP_DURATION_S, oversample, rng)
        X_held, y_held = build_features(held_out_rows, REAL_ROOT, 16000,
                                         config.CLIP_DURATION_S, f"held-out {held_out}")

        X_fold_train = np.concatenate([X_train_base, X_real_fold], axis=0)
        y_fold_train = np.concatenate([y_train_base, y_real_fold], axis=0)
        X_fold_train_norm = (X_fold_train - mean) / std
        X_held_norm = (X_held - mean) / std

        Xt = torch.from_numpy(X_fold_train_norm).unsqueeze(1)
        yt = torch.from_numpy(y_fold_train)
        Xh = torch.from_numpy(X_held_norm).unsqueeze(1)

        net, _ = train_net(Xt, yt, Xv, y_val, n_mels=X_fold_train_norm.shape[1],
                            epochs=epochs, seed=200 + i, verbose=False)
        result = evaluate(net, Xh, y_held, f"held-out speaker {held_out} (never trained on)")
        per_member[held_out] = result

    n_total = sum(r["n"] for r in per_member.values())
    acc = sum(r["acc"] * r["n"] for r in per_member.values()) / n_total
    frr_vals = [r["frr"] for r in per_member.values() if not np.isnan(r["frr"])]
    far_vals = [r["far"] for r in per_member.values() if not np.isnan(r["far"])]
    frr = float(np.mean(frr_vals)) if frr_vals else float("nan")
    far = float(np.mean(far_vals)) if far_vals else float("nan")
    print(f"\n  LOSO summary (n={n_total} clips pooled across {len(members)} "
          f"unseen-per-fold speakers): acc={acc:.4f}  FRR={frr:.4f}  FAR={far:.4f}")
    print("  This is the honest generalization number -- report THIS one when asked "
          "'how does it do on a new voice', not the demo-voice-fit number above.")
    return {"acc": acc, "frr": frr, "far": far, "n": n_total, "per_member": per_member}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-speech-negatives-dir", type=str, default=None,
                   help="directory of real (non-synthetic) speech laid out as "
                        "<speaker>/<chapter>/*.flac (matches LibriSpeech), sampled as "
                        "extra negative training clips -- see build_real_speech_negatives "
                        "docstring for why this matters. Off by default so this script "
                        "still runs with just the repo's own data.")
    p.add_argument("--real-speech-negatives-count", type=int, default=3000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading synthetic manifest...")
    syn_rows = list(csv.DictReader(open(config.DATA_DIR / "manifest.csv")))
    train_rows = [r for r in syn_rows if r["split"] == "train"]
    val_rows = [r for r in syn_rows if r["split"] == "val"]
    test_rows = [r for r in syn_rows if r["split"] == "test"]
    print(f"  train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

    print("\nFeaturizing (log-mel spectrograms)...")
    X_train, y_train = build_features(train_rows, config.DATA_DIR, config.TARGET_SAMPLE_RATE,
                                       config.CLIP_DURATION_S, "train")
    X_val, y_val = build_features(val_rows, config.DATA_DIR, config.TARGET_SAMPLE_RATE,
                                   config.CLIP_DURATION_S, "val")
    X_test, y_test = build_features(test_rows, config.DATA_DIR, config.TARGET_SAMPLE_RATE,
                                     config.CLIP_DURATION_S, "test")

    # --- extra negatives that cost nothing to make, fixing measured failures ---
    # Built once, here, BEFORE folding in real recordings, so this exact
    # synthetic + generated-negative base (X_train_base/y_train_base) can be
    # reused unchanged by every leave-one-speaker-out fold later -- no reason
    # to regenerate 2000+ noise clips six times over.
    print("\nAdding generated negatives...")
    rng = np.random.default_rng(0)
    syn_pos_rows = [r for r in train_rows if r["label"] == "positive"]
    X_part, y_part = build_partial_negatives(syn_pos_rows, config.DATA_DIR,
                                              config.TARGET_SAMPLE_RATE,
                                              config.CLIP_DURATION_S, 1, rng)
    X_noise, y_noise = build_noise_negatives(2000, config.TARGET_SAMPLE_RATE,
                                              config.CLIP_DURATION_S, rng)
    X_train_base = np.concatenate([X_train, X_part, X_noise], axis=0)
    y_train_base = np.concatenate([y_train, y_part, y_noise], axis=0)

    held_out_speech_speakers: list[str] = []
    if args.real_speech_negatives_dir:
        X_speech, y_speech, held_out_speech_speakers = build_real_speech_negatives(
            Path(args.real_speech_negatives_dir), args.real_speech_negatives_count,
            config.TARGET_SAMPLE_RATE, config.CLIP_DURATION_S, rng)
        X_train_base = np.concatenate([X_train_base, X_speech], axis=0)
        y_train_base = np.concatenate([y_train_base, y_speech], axis=0)

    # --- Fold in ALL real recordings for the DEMO checkpoint (no holdout) ---
    # A model trained on synthetic TTS voices alone does not transfer to real
    # speech (see the first training run: 86% synthetic test acc, 45% real
    # acc). An early leave-one-speaker-out run confirmed why: with only 3
    # training speakers it didn't generalize to a 4th, unseen voice (48% acc
    # there too). So for the DEPLOYED demo checkpoint, every recorded member
    # is folded into training -- the right target for "responds correctly to
    # the voices it HAS heard", which is what a live demo needs.
    #
    # IMPORTANT: this means real_eval_acc/frr/far below is a fit-to-known-
    # voices number, not a generalization number -- performance on a
    # genuinely new, unrecorded voice is a SEPARATE question, answered by
    # run_loso_eval() further down, not by this block.
    print("\nFeaturizing REAL recordings (folding in ALL speakers for the demo)...")
    real_rows = list(csv.DictReader(open(REAL_MANIFEST)))
    real_rows = [r for r in real_rows if not r["quality_flag"]]
    print(f"  real speakers: {sorted({r['voice_id'] for r in real_rows})} ({len(real_rows)} clips)")

    # clean (unaugmented) copies, used only for the final fit-check report
    X_real, y_real = build_features(real_rows, REAL_ROOT, 16000,
                                     config.CLIP_DURATION_S, "real-clean")

    # augmented copies for training -- each repeat is a genuinely different
    # speed/pitch/gain/noise perturbation of the same recording, not a byte
    # duplicate, so the model gets real acoustic range instead of 15x
    # memorizing one exact spectrogram per clip.
    OVERSAMPLE = 15
    X_real_train, y_real_train = build_augmented_real_features(
        real_rows, REAL_ROOT, 16000, config.CLIP_DURATION_S, OVERSAMPLE, rng)

    X_train_full = np.concatenate([X_train_base, X_real_train], axis=0)
    y_train_full = np.concatenate([y_train_base, y_real_train], axis=0)
    n_pos, n_neg = int(y_train_full.sum()), int((y_train_full == 0).sum())
    print(f"  final train set: {len(y_train_full)} clips -- {n_pos} pos / {n_neg} neg "
          f"= 1:{n_neg/max(n_pos,1):.2f}")

    # normalize using train-set statistics
    mean, std = X_train_full.mean(), X_train_full.std() + 1e-6
    X_train_norm = (X_train_full - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    Xt = torch.from_numpy(X_train_norm).unsqueeze(1)
    yt = torch.from_numpy(y_train_full)
    Xv = torch.from_numpy(X_val).unsqueeze(1)
    Xte = torch.from_numpy(X_test).unsqueeze(1)

    epochs = 18  # a few more than before -- goal now is fitting the known demo voices well
    print(f"\nTraining demo checkpoint ({epochs} epochs, batch_size=64)...")
    net, best_val_acc = train_net(Xt, yt, Xv, y_val, n_mels=X_train_norm.shape[1], epochs=epochs)
    print(f"Model: {model_module.count_params(net)} parameters "
          f"({model_module.count_params(net) * 4 / 1024:.1f} KB as float32)")

    print("\n=== Evaluation ===")
    evaluate(net, Xt, y_train_full, "train (synthetic + all real speakers)")
    evaluate(net, Xv, y_val, "synthetic val")
    evaluate(net, Xte, y_test, "synthetic test")

    # NOT a generalization check -- these speakers were trained on. This is
    # "does it correctly respond to the voices it will actually hear tomorrow",
    # which is the number that matters for a live demo. It is NOT evidence the
    # model works for anyone else -- see the LOSO check right after this.
    X_real_fit = (X_real - mean) / std
    Xr = torch.from_numpy(X_real_fit).unsqueeze(1)
    print()
    real_result = evaluate(net, Xr, y_real,
                            "real recordings (TRAINED ON -- demo-voice fit, not generalization)")

    # --- honest generalization number: leave-one-speaker-out ---
    # Fewer epochs than the demo model (this is a diagnostic across 5 fold
    # models, not a deployed checkpoint) but otherwise identical architecture/
    # optimizer, via the same train_net() used above.
    loso_result = run_loso_eval(X_train_base, y_train_base, Xv, y_val, real_rows,
                                 mean, std, oversample=OVERSAMPLE, epochs=14)

    # --- pick the live threshold from data instead of guessing ---
    # Live streaming polls ~5x/second, so the false-accept rate that matters is
    # per-poll and needs to be very low; recall can be looser because a genuine
    # utterance gets ~9 consecutive chances as it slides through the window.
    print("\n=== Threshold calibration (validation set) ===")
    net.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(net(Xv)).numpy()
    pos_mask, neg_mask = y_val == 1, y_val == 0
    recommended = 0.5
    print(f"  {'thresh':>7s} {'FRR':>7s} {'FAR':>7s}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        frr = float((val_probs[pos_mask] < t).mean())
        far = float((val_probs[neg_mask] >= t).mean())
        print(f"  {t:7.2f} {frr:7.3f} {far:7.3f}")
    for t in np.arange(0.50, 0.999, 0.01):   # lowest threshold meeting FAR <= 2%
        if float((val_probs[neg_mask] >= t).mean()) <= 0.02:
            recommended = round(float(t), 2)
            break
    rec_frr = float((val_probs[pos_mask] < recommended).mean())
    rec_far = float((val_probs[neg_mask] >= recommended).mean())
    print(f"  -> strict threshold {recommended:.2f} (FRR={rec_frr:.3f}, FAR={rec_far:.3f}) "
          f"-- minimizes false accepts, at the cost of a high per-poll miss rate")

    # A SECOND threshold, chosen differently: minimize FRR+FAR jointly instead
    # of chasing FAR alone. The strict threshold above measured FRR=0.521 on
    # the last run -- more than half of genuine utterances missed on any given
    # poll. The "~9 consecutive chances as the window slides through" argument
    # for tolerating that assumes those chances are roughly independent; in
    # practice consecutive polls overlap almost entirely in audio content, so
    # errors are correlated and that argument is weaker than it looks. This is
    # the threshold to actually use for a live demo; the strict one above
    # stays available for whoever wants to prioritize false-accepts instead.
    grid = np.arange(0.05, 0.99, 0.01)
    frr_grid = np.array([float((val_probs[pos_mask] < t).mean()) for t in grid])
    far_grid = np.array([float((val_probs[neg_mask] >= t).mean()) for t in grid])
    balanced_idx = int(np.argmin(frr_grid + far_grid))
    balanced = round(float(grid[balanced_idx]), 2)
    bal_frr, bal_far = float(frr_grid[balanced_idx]), float(far_grid[balanced_idx])
    print(f"  -> balanced threshold {balanced:.2f} (FRR={bal_frr:.3f}, FAR={bal_far:.3f}) "
          f"-- minimizes FRR+FAR jointly; use this one for the live demo")

    # latency measurement, single clip, CPU
    net.eval()
    single = Xte[:1]
    with torch.no_grad():
        for _ in range(10):
            net(single)  # warmup
        t0 = time.time()
        for _ in range(200):
            net(single)
        latency_ms = (time.time() - t0) / 200 * 1000
    print(f"\nSingle-clip CPU inference latency: {latency_ms:.2f} ms")
    print(f"Model size: {model_module.count_params(net) * 4 / 1024:.1f} KB (float32)")

    print("\n=== Summary: two different real-voice claims, don't conflate them ===")
    print(f"  demo-voice fit   (trained-on speakers): acc={real_result['acc']:.4f} "
          f"FRR={real_result['frr']:.4f} FAR={real_result['far']:.4f}")
    print(f"  LOSO generalization (unseen speakers):  acc={loso_result['acc']:.4f} "
          f"FRR={loso_result['frr']:.4f} FAR={loso_result['far']:.4f} (n={loso_result['n']})")
    if held_out_speech_speakers:
        print(f"\n  Real-speech negatives were trained on a subset of "
              f"{args.real_speech_negatives_dir}; speakers {held_out_speech_speakers} were "
              f"held out entirely -- re-run scripts/false_accept_soak_test.py against ONLY "
              f"those speaker folders for an honest post-fix false-accept measurement.")

    torch.save({
        "state_dict": net.state_dict(),
        "n_mels": X_train_norm.shape[1],
        "feat_mean": float(mean),
        "feat_std": float(std),
        "target_sr": config.TARGET_SAMPLE_RATE,
        "clip_duration_s": config.CLIP_DURATION_S,
        "real_eval_acc": real_result["acc"],
        "real_eval_frr": real_result["frr"],
        "real_eval_far": real_result["far"],
        "loso_acc": loso_result["acc"],
        "loso_frr": loso_result["frr"],
        "loso_far": loso_result["far"],
        "loso_n": loso_result["n"],
        "latency_ms": latency_ms,
        "recommended_threshold": balanced,      # live_demo.py's default -- see calibration block
        "strict_threshold": recommended,        # minimizes FAR alone; higher miss rate
        "balanced_threshold": balanced,
    }, CHECKPOINT_PATH)
    print(f"\nSaved checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
