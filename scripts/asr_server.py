#!/usr/bin/env python3
"""Local "cloud" ASR server: accepts a persistent WebSocket connection from
scripts/asr_client.py, receives streamed raw PCM16 audio, and transcribes it
with Moonshine (open-source, MIT-licensed for English) -- chosen over Vosk
for materially better accuracy and over Kyutai STT for not needing a GPU or a
Rust runtime to get its streaming behavior (see README.md for the comparison).

This process stands in for "the cloud" in the hybrid edge/cloud architecture:
on a real deployment it would run on a remote server, but for this hackathon
it's a second local process on the same laptop, talking over localhost --
the wire protocol doesn't care which, so pointing scripts/live_demo.py at an
actual remote host later is a one-line change (just the --stream-to-asr URL).

Wire protocol, one persistent connection carrying many utterances:
  client -> server:
    text  {"event": "utt_start", "utt_id": N, "client_ts": T}
    binary  raw PCM16LE mono 16kHz bytes (any chunk size; pre-roll audio
            first, then live audio, no framing required)
    text  {"event": "utt_end", "utt_id": N, "client_ts": T}
  server -> client:
    text  {"event": "first_frame_ack", "utt_id": N, "server_recv_ts": T}
          -- sent the INSTANT the first binary frame of an utterance is
          received, before it's handed to Moonshine, so this timestamp isn't
          inflated by decode time. This is the latency measurement anchor:
          client_detect_ts vs. server_recv_ts, one shared clock (same
          machine), no clock sync needed.
    text  {"event": "partial", "utt_id": N, "line_id": I, "text": "..."}
    text  {"event": "final", "utt_id": N, "line_id": I, "text": "...",
           "moonshine_latency_ms": M}

Usage:
    .venv/bin/python3 scripts/asr_server.py
    .venv/bin/python3 scripts/asr_server.py --model-arch base_streaming
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import numpy as np
from moonshine_voice import ModelArch, get_model_for_language
from moonshine_voice.transcriber import (
    LineCompleted,
    LineTextChanged,
    Error as MoonshineErrorEvent,
    Transcriber,
)
import websockets

import config

# *_STREAMING architectures are Moonshine's "Ergodic Streaming Encoder"
# models built specifically for incremental, low-latency use -- the plain
# TINY/BASE archs are batch/offline and not what a live pipeline wants.
MODEL_ARCH_CHOICES = {
    "tiny_streaming": ModelArch.TINY_STREAMING,
    "base_streaming": ModelArch.BASE_STREAMING,
    "small_streaming": ModelArch.SMALL_STREAMING,
    "medium_streaming": ModelArch.MEDIUM_STREAMING,
}


def pcm16_bytes_to_float_list(data: bytes) -> list[float]:
    """Raw PCM16LE bytes -> the plain Python float list Stream.add_audio()
    wants. Kept as a standalone function so scripts/asr_client.py can use the
    exact inverse (float_to_pcm16) and the two stay obviously symmetric."""
    samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return samples.tolist()


class UtteranceSession:
    """Per-connection state for the ONE utterance currently in flight (a
    connection carries many of these over its lifetime, one at a time, since
    scripts/live_demo.py only ever has one active detection at once)."""

    def __init__(self, transcriber: Transcriber, utt_id: int):
        self.utt_id = utt_id
        self.stream = transcriber.create_stream()
        self.first_frame_seen = False
        self.events: list[dict] = []  # drained after each add_audio/stop call
        self.stream.add_listener(self._on_event)

    def _on_event(self, event) -> None:
        # Fires synchronously, inside add_audio()/stop() -- just buffer, the
        # caller sends these over the websocket afterwards (see NOTE below).
        if isinstance(event, LineTextChanged) and not event.line.is_complete:
            self.events.append({
                "event": "partial", "utt_id": self.utt_id,
                "line_id": event.line.line_id, "text": event.line.text,
            })
        elif isinstance(event, LineCompleted):
            self.events.append({
                "event": "final", "utt_id": self.utt_id,
                "line_id": event.line.line_id, "text": event.line.text,
                "moonshine_latency_ms": event.line.last_transcription_latency_ms,
            })
        elif isinstance(event, MoonshineErrorEvent):
            print(f"[asr_server] Moonshine error on utt {self.utt_id}: {event.error}")


async def handle_connection(websocket) -> None:
    peer = websocket.remote_address
    print(f"[asr_server] client connected: {peer}")
    transcriber: Transcriber = websocket.transcriber  # stashed by main()
    session: UtteranceSession | None = None

    try:
        async for message in websocket:
            if isinstance(message, (bytes, bytearray)):
                if session is None:
                    continue  # stray audio with no utt_start -- ignore, don't crash the connection
                if not session.first_frame_seen:
                    session.first_frame_seen = True
                    # NOTE: sent before touching Moonshine at all, so decode
                    # time never leaks into the measured latency.
                    await websocket.send(json.dumps({
                        "event": "first_frame_ack", "utt_id": session.utt_id,
                        "server_recv_ts": time.time(),
                    }))
                samples = pcm16_bytes_to_float_list(bytes(message))
                # add_audio()/the implicit update_transcription() it triggers
                # are synchronous CPU work; run off the event loop thread so
                # one slow pass doesn't stall other connections' I/O.
                await asyncio.to_thread(session.stream.add_audio, samples, config.TARGET_SAMPLE_RATE)
                for evt in session.events:
                    await websocket.send(json.dumps(evt))
                session.events.clear()
                continue

            # text frame -> control JSON
            msg = json.loads(message)
            if msg["event"] == "utt_start":
                session = UtteranceSession(transcriber, msg["utt_id"])
                session.stream.start()
            elif msg["event"] == "utt_end" and session is not None:
                await asyncio.to_thread(session.stream.stop)
                for evt in session.events:
                    await websocket.send(json.dumps(evt))
                session.events.clear()
                session = None
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"[asr_server] client disconnected: {peer}")


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model-arch", choices=sorted(MODEL_ARCH_CHOICES), default="small_streaming",
                    help="Moonshine streaming architecture -- bigger = more accurate, "
                         "a bit more CPU per pass. English only publishes small/medium "
                         "streaming (no base/tiny streaming for en) -- 'small_streaming' "
                         "is the latency-favoring choice; use 'medium_streaming' (the "
                         "catalog's own recommended default for English) for more "
                         "accuracy if CPU allows. Not constrained by the 256KB edge "
                         "budget: this process is the 'cloud' side.")
    args = p.parse_args()

    print(f"[asr_server] fetching/loading Moonshine model ({args.model_arch}, English, "
          f"MIT-licensed)...")
    model_path, model_arch = get_model_for_language(
        "en", MODEL_ARCH_CHOICES[args.model_arch], cache_root=config.MODELS_DIR)
    transcriber = Transcriber(model_path, model_arch=model_arch)
    print(f"[asr_server] model ready: {model_path}")

    async def handler(websocket):
        websocket.transcriber = transcriber
        await handle_connection(websocket)

    print(f"[asr_server] listening on ws://{args.host}:{args.port}")
    try:
        async with websockets.serve(handler, args.host, args.port):
            await asyncio.Future()  # run forever
    finally:
        transcriber.close()


if __name__ == "__main__":
    asyncio.run(main())
