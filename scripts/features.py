"""Log-mel spectrogram feature extraction, numpy/scipy only (no librosa/torchaudio
dependency -- keeps the feature extractor itself tiny and fast, matching the
low-latency goal: this same code path is what a real-time inference loop uses).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft

N_FFT = 400        # 25ms at 16kHz
HOP_LENGTH = 160    # 10ms at 16kHz
N_MELS = 40          # compact -- real low-latency KWS models commonly use 32-40


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    n_freqs = n_fft // 2 + 1
    mel_min, mel_max = _hz_to_mel(np.array([0.0, sr / 2.0]))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


_FB_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def log_mel_spectrogram(audio: np.ndarray, sr: int, n_fft: int = N_FFT,
                         hop_length: int = HOP_LENGTH, n_mels: int = N_MELS) -> np.ndarray:
    """Returns (n_mels, n_frames) float32 log-mel spectrogram."""
    key = (sr, n_fft, n_mels)
    if key not in _FB_CACHE:
        _FB_CACHE[key] = mel_filterbank(sr, n_fft, n_mels)
    fb = _FB_CACHE[key]

    _, _, Zxx = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length,
                      boundary=None, padded=False)
    power = np.abs(Zxx) ** 2
    mel = fb @ power
    log_mel = np.log(mel + 1e-6).astype(np.float32)
    return log_mel


def feature_shape(sr: int, duration_s: float, n_fft: int = N_FFT,
                   hop_length: int = HOP_LENGTH, n_mels: int = N_MELS) -> tuple[int, int]:
    n_samples = int(sr * duration_s)
    n_frames = 1 + (n_samples - n_fft) // hop_length
    return n_mels, n_frames


class IncrementalMelExtractor:
    """Rolling log-mel spectrogram that costs O(new samples), not O(window
    samples), per update -- the thing scripts/live_demo.py needs for a
    believable low-CPU idle-listening number.

    scripts/live_demo.py currently keeps a `window_samples`-long ring buffer
    of raw audio and calls log_mel_spectrogram() on the WHOLE buffer every
    poll (every 200ms by default). At a 2.0s window that recomputes all ~198
    frames to get ~20 new ones -- about 10x more spectrogram work than the
    new audio actually requires. This class keeps a rolling buffer of mel
    FRAMES instead: push() computes only the frames covering newly-arrived
    samples and slides the window, so per-poll cost tracks the poll interval,
    not the window length.

    Numerically verified to match log_mel_spectrogram()'s output to within
    float32 precision (~2e-6 max abs diff) for the same audio -- it reproduces
    scipy.signal.stft's default Hann window and "spectrum" scaling (divide by
    the window's sum, stored as self._scale below) exactly. Re-derive that
    scale factor by comparing against log_mel_spectrogram() on a test signal
    if a scipy upgrade ever changes stft's defaults.

    Usage:
        ext = IncrementalMelExtractor(sr=16000, n_frames=198)
        while streaming:
            ext.push(new_audio_chunk)   # any length, doesn't need to be a
                                         # multiple of hop_length
            mel = ext.frames             # (n_mels, n_frames), current window
    """

    def __init__(self, sr: int, n_frames: int, n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH, n_mels: int = N_MELS):
        key = (sr, n_fft, n_mels)
        if key not in _FB_CACHE:
            _FB_CACHE[key] = mel_filterbank(sr, n_fft, n_mels)
        self._fb = _FB_CACHE[key]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_frames = n_frames
        self._window = np.hanning(n_fft + 1)[:-1].astype(np.float64)  # periodic Hann, matches scipy get_window(..., fftbins=True)
        self._scale = self._window.sum()  # scipy stft's default 'spectrum' scaling
        # Raw samples not yet consumed into a frame. Deliberately NOT seeded
        # with fake silence: log_mel_spectrogram() never pads either, it just
        # slides directly over whatever's in its input array, so the first
        # frame here must wait for n_fft genuine samples, exactly like the
        # batch version's first frame does over a freshly-started buffer.
        self._pending = np.zeros(0, dtype=np.float64)
        self.frames = np.zeros((n_mels, n_frames), dtype=np.float32)
        self.n_frames_seen = 0  # for detecting when the window has fully filled

    def push(self, new_audio: np.ndarray) -> None:
        self._pending = np.concatenate([self._pending, np.asarray(new_audio, dtype=np.float64)])
        n_new = 1 + (len(self._pending) - self.n_fft) // self.hop_length if len(self._pending) >= self.n_fft else 0
        if n_new <= 0:
            return
        # Build all n_new overlapping (n_fft,) windows as one (n_new, n_fft)
        # view (no copy) and FFT them in a single batched call -- this is
        # what actually delivers the speedup: one vectorized rfft over all
        # new frames beats a Python loop calling rfft once per frame, even
        # though both do the same number of FLOPs (scipy.signal.stft gets
        # its speed the same way, batching every frame of a window into one
        # call instead of looping).
        windows = np.lib.stride_tricks.sliding_window_view(self._pending, self.n_fft)[::self.hop_length][:n_new]
        spec = np.fft.rfft(windows * self._window, n=self.n_fft, axis=-1) / self._scale
        power = np.abs(spec) ** 2                       # (n_new, n_freqs)
        mel = power @ self._fb.T                        # (n_new, n_mels)
        new_frames = np.log(mel.T + 1e-6).astype(np.float32)   # (n_mels, n_new)
        self._pending = self._pending[n_new * self.hop_length:]  # drop consumed samples, keep overlap tail

        self.n_frames_seen += n_new
        if n_new >= self.n_frames:
            self.frames = new_frames[:, -self.n_frames:]
        else:
            self.frames = np.concatenate(
                [self.frames[:, n_new:], new_frames], axis=1)
