#!/usr/bin/env python3
"""Demo inference script: feed it a .wav file, get a prediction + confidence
+ measured latency. This is what to run live at the hackathon.

Usage:
    .venv/bin/python3 scripts/infer.py path/to/clip.wav [more.wav ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

import config
import features
import model as model_module
from train import load_clip, CHECKPOINT_PATH


def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    net = model_module.TinyWakeWordNet(n_mels=ckpt["n_mels"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def predict(net, ckpt, wav_path: Path) -> tuple[float, float]:
    """Returns (probability of wake word, latency_ms) for one clip, timing
    the full path: load audio -> features -> model -> probability."""
    t0 = time.time()
    audio = load_clip(wav_path, ckpt["target_sr"], ckpt["clip_duration_s"])
    mel = features.log_mel_spectrogram(audio, ckpt["target_sr"])
    mel = (mel - ckpt["feat_mean"]) / ckpt["feat_std"]
    x = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(net(x)).item()
    latency_ms = (time.time() - t0) * 1000
    return prob, latency_ms


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: infer.py path/to/clip.wav [more.wav ...]", file=sys.stderr)
        raise SystemExit(1)

    net, ckpt = load_model()
    print(f"Model: {model_module.count_params(net)} params "
          f"({model_module.count_params(net)*4/1024:.1f} KB) | "
          f"demo-voice-fit acc={ckpt.get('real_eval_acc', float('nan')):.3f} "
          f"(trained-on speakers) | "
          f"unseen-speaker acc={ckpt.get('loso_acc', float('nan')):.3f} "
          f"(LOSO, n={ckpt.get('loso_n', 0)})\n")

    for arg in sys.argv[1:]:
        path = Path(arg)
        prob, latency_ms = predict(net, ckpt, path)
        verdict = "HEY TARA DETECTED" if prob >= 0.5 else "not detected"
        print(f"{path.name:35s} -> {verdict:20s} (confidence={prob:.3f}, "
              f"latency={latency_ms:.1f} ms)")


if __name__ == "__main__":
    main()
