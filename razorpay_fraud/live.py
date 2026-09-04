"""A live console: the detector actually running, pushing alerts as they fire.

This is not playback of a saved run. A background thread consumes the payment
stream in wall-clock time, feeds each payment through the same
:class:`~razorpay_fraud.features.StreamingFeaturizer` and
:class:`~razorpay_fraud.detectors.RuleDetector` the offline pipeline uses, and
publishes events the moment they happen. The browser holds one Server-Sent
Events connection and renders whatever arrives.

Why SSE and not WebSockets: the traffic here is entirely one-way
(server pushes, browser renders), SSE is a plain HTTP response that the standard
library serves without a dependency, and it reconnects on its own. Controls go
back over ordinary POSTs, which are rare enough that a second protocol would be
overkill.

Threading model
---------------
One engine thread owns all detector state and is the only writer. The HTTP
handlers only read published events through :class:`EventBus`, which is a
bounded deque behind a condition variable. Nothing else touches the featurizer,
so there is no lock around the hot path and the detector runs at the same speed
it does offline.

A late-joining browser is handed a snapshot (counters, the timeline so far, the
last few alerts) and then only the events after it, so opening a second tab
mid-run shows the same picture as the first.

The server binds to localhost. It reports simulated payments and carries their
ground-truth labels in the payload so the page can show live precision -- both
of which make it a demonstration tool, not something to expose to a network.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .detectors import RuleDetector
from .features import StreamingFeaturizer
from .schema import Transaction
from .simulator import Simulator, SimulatorConfig
from .stream import AUDIT_FEATURES

#: Cost-optimal threshold selected on the dev split by the offline pipeline.
DEFAULT_THRESHOLD = 0.0065

#: Simulated seconds between heartbeat ticks.
TICK_EVERY_SIM_S = 60.0

#: Longest single sleep, so speed changes and pauses are picked up promptly.
SLEEP_SLICE_S = 0.1


class EventBus:
    """Bounded, sequence-numbered fan-out to any number of SSE clients."""

    def __init__(self, maxlen: int = 4000) -> None:
        self._events: deque[dict] = deque(maxlen=maxlen)
        self._seq = 0
        self._cv = threading.Condition()

    def publish(self, event: dict) -> None:
        with self._cv:
            self._seq += 1
            event["seq"] = self._seq
            self._events.append(event)
            self._cv.notify_all()

    @property
    def seq(self) -> int:
        with self._cv:
            return self._seq

    def since(self, after_seq: int, timeout: float = 15.0) -> list[dict]:
        """Events newer than ``after_seq``; blocks until some exist or timeout.

        Returning an empty list is normal and meaningful -- the caller sends an
        SSE comment to keep the connection alive.
        """
        with self._cv:
            if not self._events or self._events[-1]["seq"] <= after_seq:
                self._cv.wait(timeout)
            return [e for e in self._events if e["seq"] > after_seq]


class LiveEngine:
    """Owns the detector and drives it in wall-clock time."""

    def __init__(
        self,
        *,
        config: SimulatorConfig | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        speed: float = 300.0,
        loop: bool = True,
    ) -> None:
        self.config = config or SimulatorConfig()
        self.threshold = threshold
        self.detector = RuleDetector()
        self.bus = EventBus()
        self.loop = loop

        self._lock = threading.Lock()
        self._speed = speed
        self._paused = False
        self._stop = threading.Event()
        self._restart = threading.Event()

        self.phase = "starting"
        self.dataset = None
        self.warm_total = 0
        self.warm_done = 0
        self.sim_origin = 0.0
        self.sim_now = 0.0
        self.sim_span = 0.0

        self.n_processed = 0
        self.n_alerts = 0
        self.n_true_alerts = 0
        self.n_fraud_seen = 0
        self.rule_tally: dict[str, int] = {}
        self.recent_alerts: deque[dict] = deque(maxlen=120)
        self.timeline: list[dict] = []
        self._bin_index = -1
        self._bin_count = 0
        self._bin_fraud = 0
        self._throughput = 0.0

    # ------------------------------------------------------------- controls
    @property
    def speed(self) -> float:
        with self._lock:
            return self._speed

    def set_speed(self, speed: float) -> None:
        with self._lock:
            self._speed = max(1.0, min(20_000.0, float(speed)))
        self._publish_status()

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = bool(paused)
        self._publish_status()

    def restart(self) -> None:
        self._restart.set()

    def stop(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------- snapshot
    def _status(self) -> dict:
        return {
            "phase": self.phase,
            "speed": self.speed,
            "paused": self.paused,
            "threshold": self.threshold,
            "sim_elapsed_s": self.sim_now - self.sim_origin if self.sim_origin else 0.0,
            "sim_span_s": self.sim_span,
            # Absolute unix timestamps so the page can render stream time in
            # IST, and the server's own wall clock so a viewer can see the two
            # advancing independently.
            "sim_now_ts": self.sim_now,
            "sim_start_ts": self.sim_origin,
            "server_unix": time.time(),
            "n_processed": self.n_processed,
            "n_alerts": self.n_alerts,
            "n_true_alerts": self.n_true_alerts,
            "n_fraud_seen": self.n_fraud_seen,
            "throughput": round(self._throughput, 1),
            "warm_done": self.warm_done,
            "warm_total": self.warm_total,
        }

    def _publish_status(self) -> None:
        self.bus.publish({"type": "status", "status": self._status()})

    def snapshot(self) -> dict:
        meta = dict(self.dataset.meta) if self.dataset else {}
        return {
            "seq": self.bus.seq,
            "status": self._status(),
            "meta": meta,
            "timeline": list(self.timeline),
            "alerts": list(self.recent_alerts),
            "rule_tally": dict(self.rule_tally),
            "audit_features": list(AUDIT_FEATURES),
        }

    # ------------------------------------------------------------- internals
    def _sleep(self, seconds: float) -> None:
        """Sleep in slices so pause, speed and stop stay responsive."""
        remaining = seconds
        while remaining > 0 and not self._stop.is_set() and not self._restart.is_set():
            slice_s = min(SLEEP_SLICE_S, remaining)
            time.sleep(slice_s)
            remaining -= slice_s

    def _wait_while_paused(self) -> None:
        while self.paused and not self._stop.is_set() and not self._restart.is_set():
            time.sleep(0.05)

    def _alert_payload(self, txn: Transaction, score: float, values: dict) -> dict:
        reasons = [r.as_dict() for r in self.detector.reasons(values)]
        return {
            "payment_id": txn.payment_id,
            "sim_ts": txn.created_at,
            "sim_elapsed_s": txn.created_at - self.sim_origin,
            "score": round(score, 4),
            "amount_inr": round(txn.amount_inr, 2),
            "card_id": txn.card_id,
            "device_id": txn.device_id,
            "merchant_id": txn.merchant_id,
            "reasons": reasons,
            "top_rule": reasons[0]["rule"] if reasons else "-",
            "features": {k: round(values[k], 4) for k in AUDIT_FEATURES},
            # Ground truth travels with the event only because this is a
            # simulation and the page shows live precision. A real deployment
            # would not have these fields at decision time.
            "is_fraud": txn.is_fraud,
            "pattern": txn.pattern,
            "episode_id": txn.episode_id,
        }

    def _close_bin(self) -> None:
        if self._bin_index < 0:
            return
        entry = {
            "m": self._bin_index,
            "n": self._bin_count,
            "f": self._bin_fraud,
        }
        self.timeline.append(entry)
        if len(self.timeline) > 4000:
            del self.timeline[:1000]
        self.bus.publish({"type": "bin", "bin": entry})

    def _reset_run(self) -> None:
        self.n_processed = 0
        self.n_alerts = 0
        self.n_true_alerts = 0
        self.n_fraud_seen = 0
        self.rule_tally = {}
        self.recent_alerts.clear()
        self.timeline = []
        self._bin_index = -1
        self._bin_count = 0
        self._bin_fraud = 0
        self.sim_now = self.sim_origin

    # ------------------------------------------------------------------ main
    def run(self) -> None:
        while not self._stop.is_set():
            self._restart.clear()

            if self.dataset is None:
                self.phase = "generating"
                self._publish_status()
                self.dataset = Simulator(self.config).generate()

            dev = [t for t in self.dataset.transactions if t.created_at < self.dataset.split_ts]
            live = [t for t in self.dataset.transactions if t.created_at >= self.dataset.split_ts]
            if not live:
                self.phase = "done"
                self._publish_status()
                return

            self.sim_origin = live[0].created_at
            self.sim_span = live[-1].created_at - live[0].created_at
            self._reset_run()

            # Warm-up. A detector deployed on Wednesday does not start with
            # empty sliding windows -- it starts with Tuesday's state. Skipping
            # this would hand every card and merchant an empty history and
            # quietly change the alerts.
            self.phase = "warming"
            self.warm_total = len(dev)
            self.warm_done = 0
            self._publish_status()
            featurizer = StreamingFeaturizer()
            for i, txn in enumerate(dev, 1):
                if self._stop.is_set() or self._restart.is_set():
                    break
                featurizer.process(txn)
                if i % 8000 == 0:
                    self.warm_done = i
                    self._publish_status()
            if self._stop.is_set():
                return
            if self._restart.is_set():
                continue
            self.warm_done = self.warm_total

            self.phase = "streaming"
            self._publish_status()

            prev_ts = live[0].created_at
            pending_wall = 0.0
            next_tick = self.sim_origin + TICK_EVERY_SIM_S
            window_start_wall = time.perf_counter()
            window_start_n = 0

            for txn in live:
                if self._stop.is_set() or self._restart.is_set():
                    break
                self._wait_while_paused()

                pending_wall += (txn.created_at - prev_ts) / self.speed
                prev_ts = txn.created_at
                if pending_wall >= 0.004:
                    self._sleep(pending_wall)
                    pending_wall = 0.0

                row = featurizer.process(txn)
                score = self.detector.score(row.values)

                self.sim_now = txn.created_at
                self.n_processed += 1
                self.n_fraud_seen += txn.is_fraud

                bin_index = int((txn.created_at - self.sim_origin) // 60.0)
                if bin_index != self._bin_index:
                    self._close_bin()
                    self._bin_index = bin_index
                    self._bin_count = 0
                    self._bin_fraud = 0
                self._bin_count += 1
                self._bin_fraud += txn.is_fraud

                if score >= self.threshold:
                    payload = self._alert_payload(txn, score, row.values)
                    self.n_alerts += 1
                    self.n_true_alerts += txn.is_fraud
                    rule = payload["top_rule"]
                    self.rule_tally[rule] = self.rule_tally.get(rule, 0) + 1
                    self.recent_alerts.append(payload)
                    self.bus.publish({"type": "alert", "alert": payload})

                if txn.created_at >= next_tick:
                    now_wall = time.perf_counter()
                    span = now_wall - window_start_wall
                    if span > 0:
                        self._throughput = (self.n_processed - window_start_n) / span
                    window_start_wall = now_wall
                    window_start_n = self.n_processed
                    while next_tick <= txn.created_at:
                        next_tick += TICK_EVERY_SIM_S
                    self._publish_status()

            if self._stop.is_set():
                return
            if self._restart.is_set():
                continue

            self._close_bin()
            self.phase = "complete"
            self._publish_status()

            if not self.loop:
                return
            # Idle at the end until someone restarts, rather than looping
            # straight back and making the numbers unreadable.
            while not self._stop.is_set() and not self._restart.is_set():
                time.sleep(0.1)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
PAGE_PATH = Path(__file__).with_name("live_page.html")


def make_handler(engine: LiveEngine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep the console readable
            pass

        # ---------------------------------------------------------- helpers
        def _cors(self) -> None:
            """Allow any origin.

            The published console is served from static hosting and offers to
            connect to a detector running on the viewer's own machine, which is
            a cross-origin request. This server binds to localhost and serves
            nothing but synthetic payments, so there is no secret for a hostile
            page to read -- but that is the reason it is safe here, not a
            general one. Do not copy this onto a server that holds real data.
            """
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        # -------------------------------------------------------------- GET
        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                try:
                    html = PAGE_PATH.read_bytes()
                except OSError:
                    self._send_bytes(b"live_page.html is missing", "text/plain; charset=utf-8")
                    return
                self._send_bytes(html, "text/html; charset=utf-8")
                return
            if path == "/api/state":
                self._send_json(engine.snapshot())
                return
            if path == "/api/stream":
                self._stream()
                return
            self._send_json({"error": "not found"}, 404)

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            # Chunked encoding is what HTTP/1.1 would otherwise negotiate, and
            # an SSE body has no known length, so opt out explicitly.
            self.send_header("Transfer-Encoding", "identity")
            self._cors()
            self.end_headers()

            last = engine.bus.seq
            try:
                # Hand a late joiner the full picture before the increments.
                self._write_event({"type": "snapshot", "snapshot": engine.snapshot()})
                while not engine._stop.is_set():
                    events = engine.bus.since(last, timeout=10.0)
                    if events:
                        for event in events:
                            self._write_event(event)
                        last = events[-1]["seq"]
                    else:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return  # the browser navigated away; nothing to clean up

        def _write_event(self, event: dict) -> None:
            body = json.dumps(event, separators=(",", ":"))
            self.wfile.write(b"data: " + body.encode("utf-8") + b"\n\n")
            self.wfile.flush()

        def do_OPTIONS(self):  # noqa: N802
            """Preflight for the cross-origin POST to /api/control."""
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self._cors()
            self.end_headers()

        # ------------------------------------------------------------- POST
        def do_POST(self):  # noqa: N802
            if self.path.split("?", 1)[0] != "/api/control":
                self._send_json({"error": "not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body: dict[str, Any] = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send_json({"error": "bad json"}, 400)
                return

            if "speed" in body:
                engine.set_speed(body["speed"])
            if "paused" in body:
                engine.set_paused(body["paused"])
            if body.get("restart"):
                engine.restart()
            self._send_json({"ok": True, "status": engine._status()})

    return Handler


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8800,
    config: SimulatorConfig | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    speed: float = 300.0,
) -> None:
    """Run the live console until interrupted."""
    engine = LiveEngine(config=config, threshold=threshold, speed=speed)
    thread = threading.Thread(target=engine.run, name="detector", daemon=True)
    thread.start()

    class Server(ThreadingHTTPServer):
        # On Windows SO_REUSEADDR does not mean "rebind after TIME_WAIT", it
        # means "bind even though someone else already has this port". Two
        # detectors then share 8800 and requests land on whichever, which looks
        # exactly like the running server ignoring your code changes. Refuse
        # instead, and say so. Elsewhere the flag is wanted: without it a
        # restart fails while the old socket lingers in TIME_WAIT.
        allow_reuse_address = os.name != "nt"

    try:
        server = Server((host, port), make_handler(engine))
    except OSError as exc:
        engine.stop()
        raise SystemExit(
            f"cannot bind {host}:{port} -- {exc}. "
            f"Another detector is probably already running; stop it, or pass "
            f"--port with a free port."
        ) from exc
    server.daemon_threads = True
    url = f"http://{host}:{port}/"
    print(f"live fraud console on {url}")
    print(f"  threshold {threshold:.4f} · {speed:.0f}x · detector running in a background thread")
    print("  Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        engine.stop()
        server.shutdown()
        server.server_close()
