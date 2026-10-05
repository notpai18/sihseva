#!/usr/bin/env python3
"""Live microphone wake-word demo: listens continuously and prints
"HEY TARA DETECTED" the moment it hears the phrase. This is the script to
run live at the hackathon.

Usage:
    .venv/bin/python3 scripts/live_demo.py
    .venv/bin/python3 scripts/live_demo.py --threshold 0.4   # more sensitive
    .venv/bin/python3 scripts/live_demo.py --list-devices    # find --device N
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

import asr_client
import features
import model as model_module
import ui_broadcast
import vad as vad_module
from train import CHECKPOINT_PATH


def normalize_loudness(audio: np.ndarray, target_peak: float = 0.7,
                        max_gain_db: float = 12.0, floor_peak: float = 0.08) -> np.ndarray:
    """Match the live-captured window's loudness to roughly what the model was
    trained on, WITHOUT ever clipping.

    Peak-based (not RMS-based): the previous RMS-targeting version had no
    ceiling on the actual sample values, so a quiet room with even one sharp
    transient (a click, a breath, a chair creak) could get amplified straight
    through 0 dBFS -- confirmed in the debug dump (recorded "normpeak" values
    above 1.0, i.e. already-clipped/distorted audio was being fed to the
    model). Targeting peak instead guarantees the output never exceeds
    `target_peak` by construction, so this can't happen again. Skips
    near-silent windows (nothing worth normalizing, and boosting a quiet
    room's noise floor up to speech level would itself create false triggers)
    and caps the gain so very quiet backgrounds don't get blown up either.

    The floor and max gain are deliberately conservative: the model scores
    higher on louder noise (0.397 at -50 dBFS room tone, 0.548 at -30 dBFS),
    so aggressively amplifying a quiet room walks the input straight into
    false-trigger territory. Only audio already loud enough to plausibly be
    speech gets boosted, and only modestly.
    """
    peak = float(np.max(np.abs(audio)))
    if peak < floor_peak:
        return audio
    gain_db = min(20 * np.log10(target_peak / peak), max_gain_db)
    out = audio * (10 ** (gain_db / 20.0))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def loudness_gain(peak: float, target_peak: float = 0.7,
                   max_gain_db: float = 12.0, floor_peak: float = 0.08) -> float:
    """The linear gain normalize_loudness() would apply to a window with this
    peak -- exposed separately so it can be applied to already-computed
    log-mel features (see apply_gain_to_log_mel) instead of the raw waveform.
    """
    if peak < floor_peak:
        return 1.0
    gain_db = min(20 * np.log10(target_peak / peak), max_gain_db)
    return 10 ** (gain_db / 20.0)


def apply_gain_to_log_mel(log_mel: np.ndarray, gain: float, eps: float = 1e-6) -> np.ndarray:
    """Reproduces exactly what normalize_loudness() + log_mel_spectrogram()
    would have produced, starting from log-mel features that were computed
    from RAW (unscaled) audio -- so the incremental extractor never has to
    recompute old frames just because the target gain changed on a later poll.

    Scaling a signal by `gain` scales its power spectrum by gain^2 (power is
    squared), so this exactly inverts the log/epsilon (raw_mel = exp(log_mel)
    - eps) and reapplies it (log(raw_mel*gain^2 + eps)) rather than adding a
    constant shift in log-space, which was tried first and measured to
    occasionally flip the detector's decision near the operating threshold on
    real clips (near-silent frames have raw_mel comparable to eps, where the
    shift approximation breaks down). This version is float-exact: verified
    to zero decision mismatches and zero probability difference against the
    batch path across a sample of real recordings.
    """
    if gain == 1.0:
        return log_mel
    raw_mel = np.exp(log_mel.astype(np.float64)) - eps
    return np.log(np.maximum(raw_mel, 0.0) * (gain ** 2) + eps).astype(np.float32)


def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    net = model_module.TinyWakeWordNet(n_mels=ckpt["n_mels"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=None,
                    help="detection threshold; lower = more sensitive. Defaults to the "
                         "value calibrated on validation data during training.")
    p.add_argument("--device", type=int, default=None,
                    help="input device index (see --list-devices)")
    p.add_argument("--cooldown-s", type=float, default=1.5,
                    help="minimum seconds between two detections")
    p.add_argument("--poll-interval-s", type=float, default=0.2,
                    help="how often to run inference on the sliding window")
    p.add_argument("--confirm-polls", type=int, default=2,
                    help="require this many CONSECUTIVE polls above threshold before "
                         "declaring a detection (default 2). A genuine utterance produces "
                         "several consecutive high-confidence polls as it slides through "
                         "the window; an isolated noise spike usually doesn't. Set to 1 "
                         "to trigger on any single poll (more sensitive, more false positives).")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--debug-dir", type=str, default=None,
                    help="if set, saves every polled window as a .wav here whenever "
                         "confidence > 0.2, so the raw captured audio can be inspected "
                         "-- useful for diagnosing why live results differ from file-based ones")
    p.add_argument("--stream-to-asr", type=str, default=None, metavar="WS_URL",
                    help="if set (e.g. ws://127.0.0.1:8765), stream audio to the cloud "
                         "ASR server at this URL the moment the wake word is detected. "
                         "Off by default -- the wake-word-only demo path is unchanged "
                         "when this isn't passed.")
    p.add_argument("--preroll-s", type=float, default=1.5,
                    help="seconds of audio BEFORE the detection to include as the first "
                         "frame sent to the ASR server, so the stream already covers the "
                         "moment the wake word ended, not just what comes after")
    p.add_argument("--asr-endpoint-silence-s", type=float, default=0.7,
                    help="stop streaming to the ASR server after this much trailing "
                         "non-speech (per Silero VAD, not a peak threshold) following "
                         "a detection")
    p.add_argument("--asr-vad-threshold", type=float, default=vad_module.DEFAULT_THRESHOLD,
                    help="Silero VAD speech probability (0-1) above which a poll's audio "
                         "counts as speech -- gates both what gets SENT to the ASR server "
                         "(silence/background noise between words is held back rather than "
                         "streamed and hallucinated into text) and when the utterance ends. "
                         "Replaces an earlier peak-threshold version that couldn't tell "
                         "quiet speech from room noise/hum, which was the direct cause of "
                         "garbled transcripts from non-speech audio.")
    p.add_argument("--dashboard-port", type=int, default=8766,
                    help="port for the read-only status feed a browser dashboard can "
                         "connect to (ws://127.0.0.1:PORT) to show confidence, detections, "
                         "and the live transcript. Runs by default; pass --no-dashboard "
                         "to skip it entirely.")
    p.add_argument("--no-dashboard", action="store_true",
                    help="skip starting the dashboard status feed")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    net, ckpt = load_model()
    sr = ckpt["target_sr"]
    duration = ckpt["clip_duration_s"]
    window_samples = int(sr * duration)
    if args.threshold is None:
        args.threshold = ckpt.get("recommended_threshold", 0.5)

    debug_dir = None
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug mode: saving windows with confidence > 0.2 to {debug_dir}/")

    print(f"Model: {model_module.count_params(net)} params, "
          f"trained real-eval acc={ckpt.get('real_eval_acc', float('nan')):.3f}")
    print(f"Listening at {sr} Hz, {duration}s sliding window, threshold={args.threshold}")
    print("Say 'Hey Tara' any time. Press Ctrl+C to stop.\n")

    ui = None
    if not args.no_dashboard:
        ui = ui_broadcast.UIBroadcaster(port=args.dashboard_port)
        ui.start()

    # Connect BEFORE opening the mic stream: connection setup must never be
    # in the latency-critical path between a detection and the first byte
    # sent, so it happens once here and stays open for the whole session.
    asr = None
    vad = None
    if args.stream_to_asr:
        on_asr_event = ui.broadcast if ui is not None else None
        asr = asr_client.AsrStreamClient(args.stream_to_asr, sample_rate=sr, on_event=on_asr_event)
        asr.start()
        vad = vad_module.SileroVAD(sr=sr)

    # Raw ring buffer -- kept only for the peak reading that drives loudness
    # normalization (a cheap max() over the window, not an FFT) and for
    # --debug-dir dumps. The actual spectrogram is no longer recomputed from
    # this every poll -- see mel_extractor below.
    ring = deque(np.zeros(window_samples, dtype=np.float32), maxlen=window_samples)
    _, n_frames = features.feature_shape(sr, duration)
    mel_extractor = features.IncrementalMelExtractor(sr=sr, n_frames=n_frames)
    pending_lock = threading.Lock()
    pending_chunks: list[np.ndarray] = []  # raw audio arrived since the last poll, audio-thread -> main-thread handoff
    utterance_active = threading.Event()  # set while streaming a detected utterance to the ASR server

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        mono = indata[:, 0] if indata.ndim > 1 else indata
        mono = mono.astype(np.float32)
        ring.extend(mono)
        with pending_lock:
            pending_chunks.append(mono.copy())
        # NOTE: audio is no longer fed to the ASR client here. It's gated by
        # Silero VAD in the main loop instead (below), so silence/background
        # noise between words never reaches the ASR server in the first place.

    try:
        stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                 callback=callback, device=args.device,
                                 blocksize=int(sr * args.poll_interval_s))
    except Exception as e:
        print(f"Could not open audio input at {sr} Hz: {e}", file=sys.stderr)
        print("Try --list-devices and pick one explicitly with --device N.", file=sys.stderr)
        raise SystemExit(1)

    last_trigger_time = 0.0
    last_debug_save = 0.0
    consecutive_hits = 0
    first_hit_ts = 0.0    # poll where consecutive_hits went 0->1: closer to the true
                          # acoustic end of the keyword than the confirm-fire moment,
                          # which lags it by up to confirm_polls*poll_interval_s
    silence_accum_s = 0.0  # trailing silence seen so far during an active ASR utterance
    with stream:
        try:
            while True:
                time.sleep(args.poll_interval_s)

                # hand off whatever raw audio the callback collected since the
                # last poll, and only featurize THAT (not the whole window --
                # see IncrementalMelExtractor)
                with pending_lock:
                    new_chunks, pending_chunks[:] = pending_chunks[:], []
                if new_chunks:
                    mel_extractor.push(np.concatenate(new_chunks))

                # peak reading still needs the raw window (cheap max(), not
                # an FFT) so loudness normalization keeps working exactly as
                # before -- applied as an additive shift in log-mel space
                # instead of rescaling the raw waveform and refeaturizing it.
                audio_raw = np.array(ring, dtype=np.float32)
                peak_raw = float(np.max(np.abs(audio_raw)))
                gain = loudness_gain(peak_raw)
                mel = apply_gain_to_log_mel(mel_extractor.frames, gain)
                mel_norm = (mel - ckpt["feat_mean"]) / ckpt["feat_std"]
                x = torch.from_numpy(mel_norm).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    prob = torch.sigmoid(net(x)).item()

                now = time.time()
                if debug_dir is not None and prob > 0.2 and (now - last_debug_save) > 0.8:
                    audio_norm = normalize_loudness(audio_raw)
                    peak = float(np.max(np.abs(audio_norm)))
                    fname = debug_dir / (f"win_{now:.0f}_conf{prob:.3f}_"
                                          f"rawpeak{peak_raw:.3f}_normpeak{peak:.3f}.wav")
                    sf.write(str(fname), audio_norm, sr, subtype="PCM_16")
                    last_debug_save = now

                if prob >= args.threshold:
                    if consecutive_hits == 0:
                        first_hit_ts = now
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0

                bar = "#" * int(prob * 30)
                if (consecutive_hits >= args.confirm_polls
                        and (now - last_trigger_time) > args.cooldown_s):
                    print(f"\r{' ' * 60}\r*** HEY TARA DETECTED *** (confidence={prob:.3f})")
                    last_trigger_time = now
                    consecutive_hits = 0
                    if ui is not None:
                        ui.broadcast({"type": "detected", "confidence": round(prob, 3)})
                    if asr is not None:
                        preroll_n = min(int(args.preroll_s * sr), window_samples)
                        preroll = audio_raw[-preroll_n:]
                        # Reset VAD for this new utterance, then trim leading
                        # silence/room-tone off the pre-roll before sending it
                        # -- trim_leading_silence runs the WHOLE pre-roll
                        # through `vad` (not just up to the trim point) so its
                        # state/context stay continuous into the live audio
                        # that follows immediately after.
                        vad.reset()
                        preroll_trimmed = vad_module.trim_leading_silence(
                            vad, preroll, threshold=args.asr_vad_threshold)
                        asr.trigger_utterance(preroll_trimmed, detect_ts=first_hit_ts)
                        silence_accum_s = 0.0
                        utterance_active.set()
                elif asr is not None and utterance_active.is_set():
                    # Gate on Silero VAD, not a peak threshold: only send this
                    # poll's new audio to the ASR server if it actually judged
                    # speech-like, and only count it as silence for
                    # endpointing if it didn't. This is what keeps background
                    # noise/hum between words from reaching (and being
                    # hallucinated by) the ASR server at all.
                    chunk_audio = np.concatenate(new_chunks) if new_chunks else np.zeros(0, dtype=np.float32)
                    vad_prob = vad.process(chunk_audio) if chunk_audio.size else 0.0
                    if vad_prob >= args.asr_vad_threshold:
                        asr.feed_audio(chunk_audio)
                        silence_accum_s = 0.0
                    else:
                        silence_accum_s += args.poll_interval_s
                    if silence_accum_s >= args.asr_endpoint_silence_s:
                        asr.end_utterance()
                        utterance_active.clear()
                        silence_accum_s = 0.0
                        if ui is not None:
                            ui.broadcast({"type": "utterance_end"})
                    print(f"\r streaming to ASR... conf={prob:.3f} {bar:<30s} "
                          f"(vad={vad_prob:.2f}, silence={silence_accum_s:.1f}s)      ",
                          end="", flush=True)
                    if ui is not None:
                        ui.broadcast({"type": "streaming", "conf": round(prob, 3),
                                      "vad": round(vad_prob, 3), "silence_s": round(silence_accum_s, 1)})
                else:
                    print(f"\r listening... conf={prob:.3f} {bar:<30s} "
                          f"(mic peak={peak_raw:.3f}, hits={consecutive_hits})",
                          end="", flush=True)
                    if ui is not None:
                        ui.broadcast({"type": "listening", "conf": round(prob, 3), "peak": round(peak_raw, 3)})
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            if asr is not None:
                asr.stop()


if __name__ == "__main__":
    main()
