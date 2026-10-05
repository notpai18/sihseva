"""Persistent-connection streaming client for scripts/asr_server.py, used by
scripts/live_demo.py's --stream-to-asr path.

Design choices, both driven by "keep the demo-day failure surface small":
  - The synchronous `websockets.sync.client` API, not asyncio-in-a-thread.
    It gives a persistent connection, thread-safe sends, and a blocking
    receive loop with none of asyncio's event-loop bookkeeping -- the
    simplest mechanism that actually does the job.
  - Two threads share the one connection: this thread (whichever calls
    trigger_utterance/feed_audio/end_utterance -- typically live_demo.py's
    poll loop or its mic callback) sends directly under a lock, and a
    dedicated background thread only ever reads (`for message in ws: ...`).
    websockets' sync client supports concurrent send-from-one-thread /
    recv-from-another by design, so this needs no queue in between.

The connection is opened once, in start(), and stays open across every
detection for the life of the process -- opening it fresh per detection
would put a TCP+TLS handshake directly in the path of the metric this whole
feature exists to make small.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional

import numpy as np
from websockets.sync.client import connect


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """float32 audio in [-1, 1] -> raw PCM16LE bytes, the wire format
    scripts/asr_server.py expects. Inverse of its pcm16_bytes_to_float_list."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class AsrStreamClient:
    def __init__(self, url: str, sample_rate: int = 16000,
                 on_event: Optional[Callable[[dict], None]] = None):
        """`on_event`, if given, is called with a plain dict for every parsed
        server message (on the receive thread) -- e.g. scripts/live_demo.py
        wires this to scripts/ui_broadcast.py so a browser dashboard sees the
        same partial/final transcript and latency events the console prints.
        Must not block or raise; kept fire-and-forget on purpose."""
        self.url = url
        self.sample_rate = sample_rate
        self._on_event = on_event
        self._ws = None
        self._send_lock = threading.Lock()
        self._recv_thread: threading.Thread | None = None
        self._running = False
        self._next_utt_id = 0
        self._current_utt_id: int | None = None
        # utt_id -> the detection timestamp we'll report latency against,
        # once this utterance's first_frame_ack comes back.
        self._pending_detect_ts: dict[int, float] = {}

    def start(self) -> None:
        """Open the persistent connection and start the receive thread.
        Call once at program startup, well before any detection can fire."""
        self._ws = connect(self.url, open_timeout=5, legacy=True)
        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        print(f"[asr_client] connected to {self.url}")

    def _recv_loop(self) -> None:
        try:
            for message in self._ws:
                self._handle_message(message)
        except Exception as e:
            if self._running:
                print(f"[asr_client] receive loop ended ({e}); "
                      f"no auto-reconnect -- restart the demo if this wasn't intentional.")

    def _handle_message(self, raw: str) -> None:
        msg = json.loads(raw)
        ev = msg.get("event")
        if ev == "first_frame_ack":
            detect_ts = self._pending_detect_ts.pop(msg["utt_id"], None)
            if detect_ts is not None:
                delta_ms = (msg["server_recv_ts"] - detect_ts) * 1000.0
                print(f"\n[latency] utt {msg['utt_id']}: keyword-end -> ASR received "
                      f"= {delta_ms:.1f} ms")
                self._emit({"type": "latency", "ms": round(delta_ms, 1)})
        elif ev == "partial":
            print(f"\r[asr partial] {msg['text']:<70s}", end="", flush=True)
            self._emit({"type": "asr_partial", "text": msg["text"]})
        elif ev == "final":
            ms = msg.get("moonshine_latency_ms", "?")
            print(f"\n[asr final]   {msg['text']}  (moonshine pass: {ms} ms)")
            self._emit({"type": "asr_final", "text": msg["text"], "moonshine_latency_ms": ms})

    def _emit(self, event: dict) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:
                pass

    def trigger_utterance(self, preroll_audio: np.ndarray, detect_ts: float) -> int:
        """Call the moment a detection fires. Sends utt_start immediately
        followed by the pre-roll buffer as the first binary frame -- this is
        what makes the measured 'keyword end -> ASR received' delta reflect
        reality: the server's first_frame_ack timestamp lands on audio that
        already covers the moment the wake word ended, not audio starting
        fresh from the trigger."""
        self._next_utt_id += 1
        utt_id = self._next_utt_id
        self._current_utt_id = utt_id
        self._pending_detect_ts[utt_id] = detect_ts
        with self._send_lock:
            self._ws.send(json.dumps({
                "event": "utt_start", "utt_id": utt_id, "client_ts": time.time(),
            }))
            self._ws.send(float_to_pcm16(preroll_audio))
        return utt_id

    def feed_audio(self, chunk: np.ndarray) -> None:
        """Call with live audio (any chunk size) while an utterance is active."""
        if self._current_utt_id is None or chunk.size == 0:
            return
        with self._send_lock:
            self._ws.send(float_to_pcm16(chunk))

    def end_utterance(self) -> None:
        """Call once endpointing (trailing silence) decides the utterance is
        over. Does NOT close the connection -- it's reused for the next
        detection."""
        if self._current_utt_id is None:
            return
        with self._send_lock:
            self._ws.send(json.dumps({
                "event": "utt_end", "utt_id": self._current_utt_id, "client_ts": time.time(),
            }))
        self._current_utt_id = None

    @property
    def utterance_active(self) -> bool:
        return self._current_utt_id is not None

    def stop(self) -> None:
        self._running = False
        try:
            self._ws.close()
        except Exception:
            pass
