"""Numpy/scipy-only audio augmentation utilities (no ffmpeg/sox dependency).

All functions operate on float32 mono arrays in [-1, 1] at a given sample rate.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import butter, lfilter, resample_poly


def _rational_resample(audio: np.ndarray, ratio: float, max_denominator: int = 1000) -> np.ndarray:
    """resample_poly with up/down taken from a small rational approximation of
    `ratio` (output_len / input_len), instead of using the raw sample counts as
    up/down. Using raw lengths as up/down forces a polyphase filter with tens of
    thousands of taps (they're usually coprime-ish); rationalizing first keeps
    up/down small (usually < 200) for ~100x less compute at imperceptible
    (<0.1%) accuracy cost.
    """
    frac = Fraction(ratio).limit_denominator(max_denominator)
    if frac.numerator == 0:
        frac = Fraction(1, max_denominator)
    return resample_poly(audio, frac.numerator, frac.denominator).astype(np.float32)


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
    """Change playback speed (affects both duration and pitch together),
    a standard, cheap augmentation used widely in ASR/KWS training."""
    return _rational_resample(audio, 1.0 / factor)


def pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch while keeping duration roughly constant: resample to change
    pitch, then resample back to the original length."""
    if abs(semitones) < 1e-6:
        return audio.astype(np.float32)
    ratio = 2 ** (semitones / 12.0)
    stretched = _rational_resample(audio, 1.0 / ratio)
    restored = _rational_resample(stretched, ratio)
    if len(restored) > len(audio):
        restored = restored[: len(audio)]
    elif len(restored) < len(audio):
        restored = np.pad(restored, (0, len(audio) - len(restored)))
    return restored.astype(np.float32)


def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return (audio * (10 ** (gain_db / 20.0))).astype(np.float32)


def normalize_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = np.max(np.abs(audio)) + 1e-9
    return (audio / max_abs * peak).astype(np.float32)


def safety_limit(audio: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """Scale down only if the signal would clip; otherwise leave amplitude
    untouched. Unlike normalize_peak, this preserves intentional gain/loudness
    variation instead of forcing every clip to the same peak level."""
    max_abs = np.max(np.abs(audio))
    if max_abs > ceiling:
        return (audio / max_abs * ceiling).astype(np.float32)
    return audio.astype(np.float32)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float,
                rng: np.random.Generator) -> np.ndarray:
    """Additively mix `noise` into `signal` at the given SNR (dB), tiling/cropping
    noise to match signal length."""
    if len(noise) < len(signal):
        reps = int(np.ceil(len(signal) / len(noise)))
        noise = np.tile(noise, reps)
    start = 0 if len(noise) == len(signal) else rng.integers(0, len(noise) - len(signal) + 1)
    noise = noise[start:start + len(signal)]

    sig_power = np.mean(signal ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise_scaled = noise * np.sqrt(target_noise_power / noise_power)
    mixed = signal + noise_scaled
    return safety_limit(mixed)


def add_synthetic_reverb(audio: np.ndarray, sr: int, decay_s: float,
                          rng: np.random.Generator, drr_db: float = 6.0) -> np.ndarray:
    """Convolve with a synthetic exponentially-decaying noise impulse response
    to emulate room reflections.

    `drr_db` is the direct-to-reverberant ratio: the energy of the direct path
    relative to all reflections. Real rooms sit around 0-15 dB. Scaling the
    reflection tail explicitly is what keeps this a room rather than a smear --
    normalizing the whole IR instead leaves the direct path at ~0.1% of total
    energy, which convolves speech with a third of a second of noise and wipes
    out the consonant onsets the detector needs.

    Swap in real room-impulse-response recordings for higher fidelity.
    """
    ir_len = max(2, int(decay_s * sr))
    t = np.arange(ir_len) / sr
    envelope = np.exp(-t / max(decay_s / 4, 1e-3))
    reflections = rng.normal(0.0, 1.0, ir_len) * envelope
    reflections[0] = 0.0

    direct_energy = 1.0
    target_reflection_energy = direct_energy / (10 ** (drr_db / 10.0))
    reflections *= np.sqrt(target_reflection_energy / (np.sum(reflections ** 2) + 1e-12))

    ir = reflections
    ir[0] = 1.0
    wet = np.convolve(audio, ir, mode="full")[: len(audio)]
    return safety_limit(wet)


def lowpass_filter(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    nyq = sr / 2.0
    cutoff_hz = min(cutoff_hz, nyq * 0.95)
    b, a = butter(4, cutoff_hz / nyq, btype="low")
    return lfilter(b, a, audio).astype(np.float32)


def soft_clip(audio: np.ndarray, drive: float = 1.5) -> np.ndarray:
    """Cheap-microphone-style clipping distortion via tanh saturation.
    Keep drive modest -- on a near-full-scale signal, drive >3 approaches a
    square wave, which is destruction rather than a plausible mic artifact."""
    return np.tanh(audio * drive).astype(np.float32)


def place_in_window(audio: np.ndarray, sr: int, seconds: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Pad/crop to an exact duration with the content placed at a RANDOM offset.

    Random rather than centered placement leaves no positional prior for the
    model to exploit: a detector trained only on centered clips learns "the
    wake word lives in the middle of the buffer", which does not survive
    streaming inference where a sliding window sees the phrase at any offset.
    """
    n_target = int(sr * seconds)
    if len(audio) < n_target:
        pad_total = n_target - len(audio)
        pad_left = int(rng.integers(0, pad_total + 1))
        return np.pad(audio, (pad_left, pad_total - pad_left)).astype(np.float32)
    if len(audio) > n_target:
        start = int(rng.integers(0, len(audio) - n_target + 1))
        return audio[start:start + n_target].astype(np.float32)
    return audio.astype(np.float32)


def pad_with_silence(audio: np.ndarray, sr: int, leading_ms: float, trailing_ms: float) -> np.ndarray:
    lead = np.zeros(int(sr * leading_ms / 1000), dtype=np.float32)
    trail = np.zeros(int(sr * trailing_ms / 1000), dtype=np.float32)
    return np.concatenate([lead, audio, trail]).astype(np.float32)
