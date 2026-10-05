"""Tiny WebSocket broadcaster so a browser-based dashboard can show what
scripts/live_demo.py is doing in real time: confidence level, detection
moments, streaming state, and the ASR transcript as it arrives.

This is deliberately separate from scripts/asr_client.py's connection to
scripts/asr_server.py -- that one is live_demo.py acting as a WEBSOCKET
CLIENT talking to the cloud ASR. This one is live_demo.py acting as a
WEBSOCKET SERVER, broadcasting read-only status to any number of browser
tabs that connect to watch. Read-only by design: a dashboard should never be
able to feed commands back into the pipeline over this channel.

Message schema (one JSON object per line, no framing needed -- one
WebSocket text message per event):
    {"type": "listening", "conf": 0.02, "peak": 0.01}
        -- sent every poll while idle
    {"type": "detected", "confidence": 0.96}
        -- sent the instant the wake word fires
    {"type": "streaming", "conf": 0.56, "vad": 0.87, "silence_s": 0.0}
        -- sent every poll while an utterance is being streamed to the ASR
    {"type": "utterance_end"}
        -- sent when endpointing closes the utterance
    {"type": "asr_partial", "text": "..."}
    {"type": "asr_final", "text": "...", "moonshine_latency_ms": 349}
    {"type": "latency", "ms": 212.0}
        -- the keyword-end -> ASR-received measurement
"""
from __future__ import annotations

import asyncio
import json
import socket
import threading
from typing import Any


def _local_lan_ips() -> list[str]:
    """Best-effort list of this machine's non-loopback IPv4 addresses, so
    start() can print a URL another laptop on the same WiFi/LAN can actually
    connect to (0.0.0.0 itself is only valid for BINDING, never for a client
    to connect to). Opens no real connection -- the UDP "connect" to a
    public IP never sends a packet, it just asks the OS which local
    interface/address would be used to reach it."""
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ips.add(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


class UIBroadcaster:
    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        # 0.0.0.0, not 127.0.0.1: a dashboard running on a SEPARATE laptop
        # (over the same WiFi/LAN) needs to reach this from outside the
        # machine -- binding to loopback only would silently refuse every
        # connection that isn't from this same host.
        self.host = host
        self.port = port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        print(f"[ui_broadcast] dashboard feed listening on port {self.port}")
        print(f"[ui_broadcast]   same machine:  ws://127.0.0.1:{self.port}")
        for ip in _local_lan_ips():
            print(f"[ui_broadcast]   other devices: ws://{ip}:{self.port}  (same WiFi/LAN only)")

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def handler(websocket):
            self._clients.add(websocket)
            try:
                async for _ in websocket:  # dashboard never sends anything meaningful; drain and ignore
                    pass
            except Exception:
                pass
            finally:
                self._clients.discard(websocket)

        async def serve():
            import websockets
            async with websockets.serve(handler, self.host, self.port):
                self._ready.set()
                await asyncio.Future()

        self._loop.run_until_complete(serve())

    def broadcast(self, event: dict[str, Any]) -> None:
        """Thread-safe: call from live_demo.py's main polling thread (or any
        thread). No-ops silently if the server isn't up yet or no dashboard
        is connected -- this must never be able to slow down or break the
        detection loop."""
        if self._loop is None:
            return
        payload = json.dumps(event)

        async def _send_all():
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

        try:
            asyncio.run_coroutine_threadsafe(_send_all(), self._loop)
        except Exception:
            pass
