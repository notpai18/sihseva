"""Real voice-activity detection via Silero VAD (MIT license, ONNX) -- used to
gate what actually gets streamed to the cloud ASR server.

Why this exists: a peak/energy threshold can't tell "someone talking quietly"
from "the fan is on" or "empty room tone." That was the direct cause of
hallucinated transcripts during the live demo -- Moonshine (like most
end-to-end ASR models) has no built-in "there is no speech here" output, so
it confidently transcribes whatever audio it's handed, background noise and
pre-roll room tone included. Silero VAD is a real, tiny (~1MB) neural
classifier trained specifically to tell speech from non-speech, so it's used
here both to decide what audio is worth sending at all (scripts/live_demo.py's
streaming loop) and to trim leading silence off the pre-roll buffer before the
very first frame of an utterance goes out.

This talks to the ONNX model directly via onnxruntime rather than through the
`silero_vad` pip package's own Python wrapper, because that wrapper
unconditionally imports torchaudio at module load time, and the torchaudio
build that resolves in this environment fails to load (a torch/torchaudio
version mismatch unrelated to Silero itself -- see the OSError about
libcudart at import time). importlib.util.find_spec locates the installed
package's bundled .onnx file without executing its __init__.py, which
sidesteps the broken import while still reusing the exact model file the pip
package installed -- no separate download or vendoring needed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import onnxruntime as ort

FRAME_SAMPLES = 512     # fixed by the model: 16kHz only, exactly 32ms per call
CONTEXT_SAMPLES = 64    # samples of preceding audio stitched onto each frame --
                         # required by this model version; omitting it silently
                         # produces near-zero probabilities on real speech (measured).
DEFAULT_THRESHOLD = 0.5  # Silero's own documented operating point


def _locate_model() -> Path:
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("silero-vad not installed -- pip install silero-vad")
    path = Path(list(spec.submodule_search_locations)[0]) / "data" / "silero_vad.onnx"
    if not path.exists():
        raise RuntimeError(f"Silero VAD model file not found at {path}")
    return path


class SileroVAD:
    """Streaming voice-activity detector. One instance per audio stream --
    its state and context are temporal (a rolling summary of recent audio)
    and must not be shared across unrelated streams; call reset() between
    genuinely separate utterances.

    Usage:
        vad = SileroVAD()
        while streaming:
            prob = vad.process(new_audio_chunk)  # any chunk length
            is_speech = prob >= DEFAULT_THRESHOLD
    """

    def __init__(self, sr: int = 16000):
        if sr != 16000:
            raise ValueError("this ONNX export only supports 16kHz")
        self.sr = sr
        self._sess = ort.InferenceSession(str(_locate_model()), providers=["CPUExecutionProvider"])
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def process(self, audio: np.ndarray) -> float:
        """Feed new audio (any length, buffered across calls). Returns the
        MAX speech probability across whatever complete 32ms frames this
        call completed, or 0.0 if not enough new audio has accumulated yet
        to complete one -- matches the "was there speech in this poll"
        question scripts/live_demo.py actually needs to ask, since a poll
        interval (200ms) spans several 32ms VAD frames."""
        self._pending = np.concatenate([self._pending, np.asarray(audio, dtype=np.float32)])
        max_prob = 0.0
        while len(self._pending) >= FRAME_SAMPLES:
            frame = self._pending[:FRAME_SAMPLES][None, :]
            self._pending = self._pending[FRAME_SAMPLES:]
            x = np.concatenate([self._context, frame], axis=1)
            out, self._state = self._sess.run(None, {
                "input": x, "state": self._state, "sr": np.array(self.sr, dtype=np.int64),
            })
            self._context = x[:, -CONTEXT_SAMPLES:]
            max_prob = max(max_prob, float(out[0, 0]))
        return max_prob


def trim_leading_silence(vad: SileroVAD, audio: np.ndarray,
                          threshold: float = DEFAULT_THRESHOLD,
                          keep_before_s: float = 0.1) -> np.ndarray:
    """Return `audio` from a little before wherever speech actually starts,
    dropping leading silence/room noise -- this is what keeps the pre-roll
    buffer sent to the ASR server from being mostly dead air.

    Runs the ENTIRE `audio` through `vad` (not stopping at the first speech
    frame found) so its state/context end up correctly reflecting the end of
    this buffer -- `vad` is meant to keep being used for whatever live audio
    follows immediately after, and that continuation is only temporally
    contiguous with the state if nothing here was skipped.

    If no frame in `audio` reaches `threshold` (e.g. a marginal trigger),
    returns `audio` unchanged rather than trimming everything away -- the
    caller's own endpointing will end the utterance quickly regardless.
    """
    n = len(audio)
    first_speech_sample = None
    pos = 0
    while pos + FRAME_SAMPLES <= n:
        prob = vad.process(audio[pos:pos + FRAME_SAMPLES])
        if first_speech_sample is None and prob >= threshold:
            first_speech_sample = pos
        pos += FRAME_SAMPLES
    if first_speech_sample is None:
        return audio
    start = max(0, first_speech_sample - int(keep_before_s * vad.sr))
    return audio[start:]
