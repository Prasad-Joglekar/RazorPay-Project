"""Streaming replay: the shape the system would actually run in.

The featurizer already maintains state incrementally, so "make it streaming" is
not a rewrite -- it is this file. Payments are consumed in timestamp order, one
at a time, and each one is scored against state built only from its
predecessors. Nothing is recomputed in batch, and no future payment is visible.

Mapping to production
---------------------
Every piece here has a direct counterpart in a real deployment, which is worth
being concrete about rather than gesturing at:

    this module                  production
    -------------------------    -----------------------------------------
    replay() over a sorted list  Kafka consumer on a payments topic,
                                 partitioned by card_id so one card's
                                 events are ordered within a partition
    StreamingFeaturizer dicts    Redis hashes / RocksDB state store, keyed
                                 by entity, with the same TTL sweep
    AlertRecord -> JSONL         alert topic + case-management queue
    threshold from dev split     config value, re-tuned on a schedule

The one thing that genuinely changes is state ownership. Here it is a single
process, so all entities are local. Partitioning by ``card_id`` keeps the
card-level windows correct but splits device- and IP-level state across
workers, which needs either a second partitioning of the same stream or a
shared store. That is a real design cost, and it is called out in the README
rather than glossed over.

Latency below is *processing* latency -- time spent in feature extraction plus
scoring -- not end-to-end pipeline latency, which would include broker hops.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

from .detectors import Detector, Reason
from .features import FeatureRow, StreamingFeaturizer
from .schema import Transaction

#: Features worth writing into every alert. The full vector is available too,
#: but these are the ones an analyst reads first.
AUDIT_FEATURES: tuple[str, ...] = (
    "card_cnt_30s",
    "card_cnt_5m",
    "card_distinct_merchants_5m",
    "card_fail_ratio_5m",
    "card_tiny_ratio_5m",
    "card_geo_speed_kmph",
    "device_distinct_cards_5m",
    "device_cnt_5m",
    "device_distinct_merchants_5m",
    "device_fail_ratio_5m",
    "ip_distinct_cards_1h",
    "merchant_rate_z",
    "merchant_tiny_ratio_5m",
    "merchant_fail_ratio_5m",
    "amount_inr",
)


@dataclass(slots=True)
class AlertRecord:
    """One alert, with everything needed to defend or dismiss it later.

    The audit trail is not decoration. A fraud decision that blocks a customer's
    payment has to be explainable after the fact -- to the merchant, to the
    customer, and to whoever reviews the dispute. This record carries the score,
    the named rules that fired with their own sub-scores, human-readable
    detail, and the exact feature values behind them, all captured at decision
    time rather than reconstructed later (by which point the sliding windows
    have moved on and the values are unrecoverable).
    """

    payment_id: str
    created_at: float
    detector: str
    score: float
    threshold: float
    merchant_id: str
    card_id: str
    device_id: str
    amount_inr: float
    reasons: list[Reason] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    # Ground truth, recorded only so offline analysis can join on it. A live
    # deployment simply would not have these fields.
    is_fraud: bool = False
    pattern: str | None = None
    episode_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "payment_id": self.payment_id,
            "created_at": self.created_at,
            "detector": self.detector,
            "score": round(self.score, 5),
            "threshold": round(self.threshold, 5),
            "merchant_id": self.merchant_id,
            "card_id": self.card_id,
            "device_id": self.device_id,
            "amount_inr": round(self.amount_inr, 2),
            "reasons": [r.as_dict() for r in self.reasons],
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "label": {
                "is_fraud": self.is_fraud,
                "pattern": self.pattern,
                "episode_id": self.episode_id,
            },
        }

    def explain(self) -> str:
        """One-paragraph plain-English rendering, for the demo and for logs."""
        head = (
            f"[{self.score:.2f}] {self.payment_id}  Rs {self.amount_inr:,.2f}  "
            f"card={self.card_id} merchant={self.merchant_id}"
        )
        if not self.reasons:
            return head
        lines = [head]
        for reason in self.reasons:
            lines.append(f"    - {reason.rule} ({reason.score:.2f}): {reason.detail}")
        return "\n".join(lines)


@dataclass(slots=True)
class ReplayResult:
    scores: list[float]
    alerts: list[AlertRecord]
    n_processed: int
    elapsed_s: float
    latencies_us: list[float]

    @property
    def throughput(self) -> float:
        return self.n_processed / self.elapsed_s if self.elapsed_s else 0.0

    def latency_percentile(self, pct: float) -> float:
        if not self.latencies_us:
            return 0.0
        ordered = sorted(self.latencies_us)
        idx = min(len(ordered) - 1, max(0, int(round(pct / 100.0 * (len(ordered) - 1)))))
        return ordered[idx]

    def summary(self) -> dict:
        return {
            "n_processed": self.n_processed,
            "n_alerts": len(self.alerts),
            "alert_rate": round(len(self.alerts) / self.n_processed, 6)
            if self.n_processed
            else 0.0,
            "elapsed_s": round(self.elapsed_s, 3),
            "throughput_per_s": round(self.throughput, 1),
            "processing_latency_us": {
                "p50": round(self.latency_percentile(50), 1),
                "p95": round(self.latency_percentile(95), 1),
                "p99": round(self.latency_percentile(99), 1),
                "max": round(self.latency_percentile(100), 1),
            },
        }


def replay(
    transactions: Sequence[Transaction],
    detector: Detector,
    threshold: float,
    *,
    featurizer: StreamingFeaturizer | None = None,
    measure_latency: bool = True,
    on_alert: Callable[[AlertRecord], None] | None = None,
    collect_alerts: bool = True,
) -> ReplayResult:
    """Consume the stream one payment at a time, scoring as we go.

    ``featurizer`` may be passed in already warmed on earlier traffic -- which
    is exactly what evaluating the test split requires. A detector deployed on
    Wednesday does not start with empty sliding windows; it starts with
    Tuesday's state. Rebuilding the featurizer for the test split alone would
    quietly hand every card and merchant an empty history and change the
    numbers.
    """
    featurizer = featurizer or StreamingFeaturizer()
    scores: list[float] = []
    alerts: list[AlertRecord] = []
    latencies: list[float] = []

    wall_start = time.perf_counter()
    for txn in transactions:
        t0 = time.perf_counter() if measure_latency else 0.0
        row = featurizer.process(txn)
        score = detector.score(row.values)
        if measure_latency:
            latencies.append((time.perf_counter() - t0) * 1e6)
        scores.append(score)

        if score >= threshold:
            record = AlertRecord(
                payment_id=txn.payment_id,
                created_at=txn.created_at,
                detector=detector.name,
                score=score,
                threshold=threshold,
                merchant_id=txn.merchant_id,
                card_id=txn.card_id,
                device_id=txn.device_id,
                amount_inr=txn.amount_inr,
                reasons=detector.reasons(row.values),
                features={k: row.values[k] for k in AUDIT_FEATURES},
                is_fraud=txn.is_fraud,
                pattern=txn.pattern,
                episode_id=txn.episode_id,
            )
            if on_alert is not None:
                on_alert(record)
            if collect_alerts:
                alerts.append(record)
    elapsed = time.perf_counter() - wall_start

    return ReplayResult(
        scores=scores,
        alerts=alerts,
        n_processed=len(transactions),
        elapsed_s=elapsed,
        latencies_us=latencies,
    )


def warm_and_replay(
    warmup: Sequence[Transaction],
    evaluate: Sequence[Transaction],
    detector: Detector,
    threshold: float,
) -> ReplayResult:
    """Build state on ``warmup`` (scores discarded), then score ``evaluate``.

    This is how the held-out test split must be run: state carries across the
    split boundary, predictions do not.
    """
    featurizer = StreamingFeaturizer()
    for txn in warmup:
        featurizer.process(txn)
    return replay(evaluate, detector, threshold, featurizer=featurizer)


def write_alerts_jsonl(alerts: Iterable[AlertRecord], path: str | Path) -> int:
    """Persist the audit trail. One JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for alert in alerts:
            fh.write(json.dumps(alert.as_dict(), separators=(",", ":")))
            fh.write("\n")
            n += 1
    return n


def paced_replay(
    transactions: Sequence[Transaction],
    detector: Detector,
    threshold: float,
    *,
    speed: float = 600.0,
    featurizer: StreamingFeaturizer | None = None,
    on_progress: Callable[[dict], None] | None = None,
    progress_every_s: float = 600.0,
) -> Iterator[AlertRecord]:
    """Replay in wall-clock time at ``speed``x, yielding alerts as they fire.

    Purely for the demo video: it makes the stream visibly a stream. At the
    default 600x, one simulated hour takes six wall-clock seconds. Detection is
    identical to :func:`replay` -- only the pacing differs.

    ``on_progress`` is called every ``progress_every_s`` *simulated* seconds
    with running counters, so a caller can show traffic flowing between alerts.
    """
    featurizer = featurizer or StreamingFeaturizer()
    if not transactions:
        return
    origin_sim = transactions[0].created_at
    origin_wall = time.perf_counter()
    n_seen = 0
    n_alerts = 0
    n_true = 0
    next_tick = origin_sim + progress_every_s

    for txn in transactions:
        target = (txn.created_at - origin_sim) / speed
        drift = target - (time.perf_counter() - origin_wall)
        if drift > 0:
            time.sleep(drift)

        # Heartbeat on the *simulated* clock, so the cadence is the same
        # whatever speed the replay runs at. Without this the demo shows
        # nothing at all during quiet traffic, which looks like a hang rather
        # than like a detector correctly staying silent.
        if on_progress is not None and txn.created_at >= next_tick:
            on_progress(
                {
                    "sim_ts": txn.created_at,
                    "sim_elapsed_s": txn.created_at - origin_sim,
                    "wall_elapsed_s": time.perf_counter() - origin_wall,
                    "n_processed": n_seen,
                    "n_alerts": n_alerts,
                    "n_true_alerts": n_true,
                    "payments_per_min": n_seen
                    / max(1e-9, (txn.created_at - origin_sim) / 60.0),
                }
            )
            while next_tick <= txn.created_at:
                next_tick += progress_every_s

        row = featurizer.process(txn)
        score = detector.score(row.values)
        n_seen += 1
        if score >= threshold:
            n_alerts += 1
            n_true += txn.is_fraud
            yield AlertRecord(
                payment_id=txn.payment_id,
                created_at=txn.created_at,
                detector=detector.name,
                score=score,
                threshold=threshold,
                merchant_id=txn.merchant_id,
                card_id=txn.card_id,
                device_id=txn.device_id,
                amount_inr=txn.amount_inr,
                reasons=detector.reasons(row.values),
                features={k: row.values[k] for k in AUDIT_FEATURES},
                is_fraud=txn.is_fraud,
                pattern=txn.pattern,
                episode_id=txn.episode_id,
            )
