"""Thin wrapper around piper-tts for synthesizing text -> float32 numpy audio,
with a roster of loaded voices (single-speaker models + sampled IDs from the
multi-speaker libritts_r model) to draw from for diversity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from piper import PiperVoice, SynthesisConfig

import config


@dataclass
class VoiceHandle:
    voice_id: str          # unique id used in filenames/manifest, e.g. "en_US-amy-medium" or "libritts_r-spk042"
    model_name: str        # underlying piper model file (without extension)
    speaker_id: Optional[int]
    piper_voice: PiperVoice


def load_voice_roster(rng: np.random.Generator) -> list[VoiceHandle]:
    roster: list[VoiceHandle] = []

    for name in config.SINGLE_SPEAKER_VOICES:
        model_path = config.VOICES_DIR / f"{name}.onnx"
        config_path = config.VOICES_DIR / f"{name}.onnx.json"
        pv = PiperVoice.load(model_path, config_path)
        roster.append(VoiceHandle(voice_id=name, model_name=name, speaker_id=None, piper_voice=pv))

    multi_name = config.MULTI_SPEAKER_VOICE
    multi_model_path = config.VOICES_DIR / f"{multi_name}.onnx"
    multi_config_path = config.VOICES_DIR / f"{multi_name}.onnx.json"
    multi_pv = PiperVoice.load(multi_model_path, multi_config_path)
    num_speakers = multi_pv.config.num_speakers or 1
    n_sample = min(config.MULTI_SPEAKER_NUM_IDS_TO_SAMPLE, num_speakers)
    sampled_ids = rng.choice(num_speakers, size=n_sample, replace=False)
    for spk_id in sampled_ids:
        roster.append(VoiceHandle(
            voice_id=f"libritts_r-spk{spk_id:04d}",
            model_name=multi_name,
            speaker_id=int(spk_id),
            piper_voice=multi_pv,
        ))

    l2_name = config.L2ARCTIC_VOICE
    l2_model_path = config.VOICES_DIR / f"{l2_name}.onnx"
    l2_config_path = config.VOICES_DIR / f"{l2_name}.onnx.json"
    l2_pv = PiperVoice.load(l2_model_path, l2_config_path)
    speaker_map = l2_pv.config.speaker_id_map
    for code in config.L2ARCTIC_HINDI_SPEAKER_CODES:
        roster.append(VoiceHandle(
            voice_id=f"l2arctic-hindi-{code}",
            model_name=l2_name,
            speaker_id=int(speaker_map[code]),
            piper_voice=l2_pv,
        ))

    return roster


def phonemize(voice: VoiceHandle, text: str) -> list[str]:
    """Phoneme sequence espeak-ng produces for `text`. Used to detect phrases
    that look different but sound identical."""
    chunks = list(voice.piper_voice.synthesize(text, SynthesisConfig(speaker_id=voice.speaker_id)))
    return [p for c in chunks for p in c.phonemes]


def synthesize(voice: VoiceHandle, text: str, length_scale: float = 1.0,
               noise_scale: float = 0.667, noise_w_scale: float = 0.8) -> tuple[np.ndarray, int]:
    syn_config = SynthesisConfig(
        speaker_id=voice.speaker_id,
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
    )
    chunks = list(voice.piper_voice.synthesize(text, syn_config))
    if not chunks:
        raise RuntimeError(f"No audio produced for text={text!r} voice={voice.voice_id!r}")
    sr = chunks[0].sample_rate
    audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    return audio, sr
