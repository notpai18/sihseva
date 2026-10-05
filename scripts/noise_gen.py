"""Synthetic background-noise generators used for negative samples and augmentation.

Kept dependency-free (numpy/scipy only) so the pipeline works with no internet
access beyond the one-time TTS voice download. For production-grade training you
should eventually mix in real recorded noise (e.g. MUSAN, DEMAND, ESC-50) instead
of relying on these synthetic beds alone -- see README.md.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter


def white_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, n_samples).astype(np.float32)


def _filtered_noise(n_samples: int, rng: np.random.Generator, power: float) -> np.ndarray:
    """Noise with a 1/f^power spectral tilt (power=1 -> pink, power=2 -> brown)."""
    white = rng.normal(0.0, 1.0, n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    scale = 1.0 / np.power(freqs, power / 2.0)
    scale[0] = scale[1] if len(scale) > 1 else 1.0
    shaped = np.fft.irfft(spectrum * scale, n=n_samples)
    shaped = shaped / (np.max(np.abs(shaped)) + 1e-9)
    return shaped.astype(np.float32)


def pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    return _filtered_noise(n_samples, rng, power=1.0)


def brown_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    return _filtered_noise(n_samples, rng, power=2.0)


def babble_noise(n_samples: int, sr: int, rng: np.random.Generator, n_voices: int = 6) -> np.ndarray:
    """Crude crowd-murmur bed: several band-limited noise streams summed together."""
    mix = np.zeros(n_samples, dtype=np.float64)
    for _ in range(n_voices):
        voice = rng.normal(0.0, 1.0, n_samples)
        low = rng.uniform(150, 400)
        high = rng.uniform(1500, 3200)
        b, a = butter(4, [low / (sr / 2), high / (sr / 2)], btype="band")
        voice = lfilter(b, a, voice)
        amp_env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * np.arange(n_samples) / sr + rng.uniform(0, np.pi))
        mix += voice * amp_env
    mix = mix / (np.max(np.abs(mix)) + 1e-9)
    return mix.astype(np.float32)


def hum_noise(n_samples: int, sr: int, rng: np.random.Generator, base_freq: float = 50.0) -> np.ndarray:
    """Mains-hum style tonal noise (fan / appliance / electrical hum)."""
    t = np.arange(n_samples) / sr
    sig = np.zeros(n_samples, dtype=np.float64)
    for harmonic in (1, 2, 3):
        sig += rng.uniform(0.3, 1.0) / harmonic * np.sin(2 * np.pi * base_freq * harmonic * t + rng.uniform(0, 2 * np.pi))
    sig += 0.05 * rng.normal(0.0, 1.0, n_samples)
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    return sig.astype(np.float32)


def silence(n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Near-silence: an idle room, not digital zero.

    Varies both level (-70..-40 dBFS) and spectral tint, because real rooms are
    never quiet the same way twice -- a single fixed Gaussian floor would make
    this class 40 copies of one sound, which teaches the model nothing beyond
    that one noise shape.
    """
    floor_db = rng.uniform(-70.0, -40.0)
    tint = rng.choice(["white", "pink", "brown"])
    if tint == "white":
        base = rng.normal(0.0, 1.0, n_samples)
    else:
        base = _filtered_noise(n_samples, rng, power=1.0 if tint == "pink" else 2.0)
    base = base / (np.std(base) + 1e-9)

    if rng.random() < 0.4:  # faint appliance/electrical hum in the background
        hum = hum_noise(n_samples, sr, rng, base_freq=float(rng.choice([50.0, 60.0])))
        base = base + hum * rng.uniform(0.1, 0.5)
        base = base / (np.std(base) + 1e-9)

    return (base * (10 ** (floor_db / 20.0))).astype(np.float32)


GENERATORS = {
    "white": lambda n, sr, rng: white_noise(n, rng),
    "pink": lambda n, sr, rng: pink_noise(n, rng),
    "brown": lambda n, sr, rng: brown_noise(n, rng),
    "babble": lambda n, sr, rng: babble_noise(n, sr, rng),
    "hum": lambda n, sr, rng: hum_noise(n, sr, rng),
    "silence": lambda n, sr, rng: silence(n, sr, rng),
}


def generate(kind: str, n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    if kind not in GENERATORS:
        raise ValueError(f"Unknown noise kind '{kind}', choose from {list(GENERATORS)}")
    return GENERATORS[kind](n_samples, sr, rng)
