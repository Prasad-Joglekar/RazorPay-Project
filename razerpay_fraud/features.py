"""Incremental, causal feature extraction over the payment stream.

Two properties are non-negotiable here, because everything downstream depends
on them:

**Causality.** The feature vector for a payment is computed from that payment
and the payments strictly before it. Never from later ones. This is what makes
the offline precision/recall numbers a fair estimate of live behaviour, and it
is enforced by a test (test_features.py::test_prefix_causality) that replays a
prefix of the stream and asserts the feature vectors are bit-identical to the
full replay.

**O(1) amortised work per payment.** Nothing is recomputed from scratch. Each
entity (card / device / IP / merchant) owns a set of SlidingAggregate windows:
a deque plus running aggregates, with a two-pointer eviction that pops expired
events off the front and decrements the aggregates as it goes. Counting a
1-hour distinct-card cardinality by rescanning the window would be O(window)
per event -- at a few thousand payments per minute that is the difference
between a detector and a batch job.

Deliberate exclusions
---------------------
Two features that would score very well here are left out on purpose, because
they are artefacts of the simulator rather than signal that would survive
contact with production traffic:

* *"first time we have ever seen this card"* -- every card-testing attack in
  the simulator uses a freshly minted ``card_ct_*`` id, so this feature would
  be a near-perfect label. In reality a large share of legitimate traffic is
  also first-sighting (new customers), so the feature would collapse in
  production.
* *instrument type* (``method``) -- 100% of simulated attacks are card
  payments, versus ~42% of legitimate ones. Some of that is real (card testing
  is card-specific by definition), but the simulator exaggerates it into a
  giveaway.

The current payment's own ``status`` is also excluded. Failure ratios are
computed from *prior* payments only, so the detector stays deployable
**pre-authorisation** -- at scoring time you do not yet know whether this
attempt will decline. See ``_history_ratio``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .schema import TINY_AMOUNT_PAISE, Transaction

# Window lengths in seconds.
W_30S = 30.0
W_1M = 60.0
W_5M = 300.0
W_1H = 3600.0

# Priors for Laplace-style ratio smoothing. A card with one prior failure out
# of one prior payment is not a 100%-failure card; without smoothing, ratio
# features are pure noise in the low-count regime where attacks actually start.
PRIOR_FAIL_RATE = 0.07
PRIOR_TINY_RATE = 0.05
PRIOR_STRENGTH = 3.0

# Below this distance, a change in location is city-level GPS/IP jitter, not
# travel. Without this floor two payments 2 s apart 15 km apart would imply
# 27,000 km/h and every busy card would look teleporting.
MIN_GEO_DISTANCE_KM = 100.0
MIN_GEO_DT_S = 60.0

# Payment traffic is Indian, so "hour of day" means the IST hour (UTC+05:30).
# A UTC hour would put the evening peak just after midnight local time and make
# the feature mean something different from what its name says.
IST_OFFSET_S = 19_800

# Entity state older than this is dropped by the periodic sweep. Bounds memory
# on a long-running stream; must exceed the longest window.
STATE_TTL_S = 2 * W_1H
GC_EVERY_N_EVENTS = 20_000


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r_lat1, r_lat2 = math.radians(lat1), math.radians(lat2)
    d_lat = r_lat2 - r_lat1
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(r_lat1) * math.cos(r_lat2) * math.sin(d_lon / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(a)))


def _history_ratio(numer: float, denom: float, prior: float) -> float:
    """Ratio shrunk toward a prior, so tiny denominators do not scream."""
    return (numer + PRIOR_STRENGTH * prior) / (denom + PRIOR_STRENGTH)


class RunningStats:
    """Welford mean/variance. Used for a card's lifetime ticket size."""

    __slots__ = ("n", "mean", "_m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self._m2 / (self.n - 1))

    def zscore(self, x: float, *, min_n: int = 3) -> float:
        """Robust-ish z-score; 0.0 until there is enough history to mean it."""
        if self.n < min_n:
            return 0.0
        denom = max(self.std, 0.15 * abs(self.mean), 1.0)
        return (x - self.mean) / denom


class SlidingAggregate:
    """A time window over one entity, with aggregates kept incrementally.

    ``track`` names the distinct-cardinality counters to maintain (e.g.
    ``("merchant", "device")``). Each event supplies its key values in the same
    order. Cardinality is ``len`` of a reference-counted dict, so it is O(1) to
    read and O(1) amortised to maintain.

    The window is half-open: ``(now - window, now]``.
    """

    __slots__ = ("window", "track", "events", "count", "sum_amount", "n_failed", "n_tiny", "_distinct")

    def __init__(self, window: float, track: tuple[str, ...] = ()) -> None:
        self.window = window
        self.track = track
        self.events: deque = deque()
        self.count = 0
        self.sum_amount = 0
        self.n_failed = 0
        self.n_tiny = 0
        self._distinct: dict[str, dict[str, int]] = {name: {} for name in track}

    def advance(self, now: float) -> None:
        """Evict everything that has fallen out of the window (two-pointer)."""
        cutoff = now - self.window
        events = self.events
        while events and events[0][0] <= cutoff:
            _, amount, failed, tiny, keys = events.popleft()
            self.count -= 1
            self.sum_amount -= amount
            self.n_failed -= failed
            self.n_tiny -= tiny
            for name, key in zip(self.track, keys):
                bucket = self._distinct[name]
                remaining = bucket[key] - 1
                if remaining:
                    bucket[key] = remaining
                else:
                    del bucket[key]

    def add(self, ts: float, amount: int, failed: bool, tiny: bool, keys: tuple[str, ...]) -> None:
        self.events.append((ts, amount, int(failed), int(tiny), keys))
        self.count += 1
        self.sum_amount += amount
        self.n_failed += int(failed)
        self.n_tiny += int(tiny)
        for name, key in zip(self.track, keys):
            bucket = self._distinct[name]
            bucket[key] = bucket.get(key, 0) + 1

    def distinct(self, name: str) -> int:
        return len(self._distinct[name])

    def distinct_including(self, name: str, key: str) -> int:
        """Cardinality if ``key`` were added -- lets us count the current
        payment without mutating the window before the history features are
        read."""
        bucket = self._distinct[name]
        return len(bucket) + (0 if key in bucket else 1)


class EwmaRate:
    """Exponentially-weighted baseline of an entity's per-minute event count.

    Only *completed* minute buckets feed the baseline, and gaps are zero-filled
    -- without the zero-fill an entity that went quiet for six hours would come
    back with its old busy baseline intact and look normal during an attack.

    The variance is tracked alongside the mean so callers get a z-score rather
    than a ratio, and the deviation is floored at the Poisson standard
    deviation ``sqrt(mean)``: for count data, a merchant averaging 4/min varies
    by about +/-2/min for free, and calling that an anomaly is how you generate
    false positives all night.
    """

    __slots__ = ("alpha", "bucket_s", "warmup", "_bucket", "_count", "mean", "var", "n_closed")

    #: Beyond this many empty buckets, stop looping and just reset the baseline.
    MAX_ZERO_FILL = 1440

    def __init__(self, alpha: float = 0.06, bucket_s: float = W_1M, warmup: int = 20) -> None:
        self.alpha = alpha
        self.bucket_s = bucket_s
        self.warmup = warmup
        self._bucket: int | None = None
        self._count = 0
        self.mean = 0.0
        self.var = 0.0
        self.n_closed = 0

    def _close_bucket(self, value: float) -> None:
        if self.n_closed == 0:
            self.mean = value
            self.var = 0.0
        else:
            alpha = self.alpha
            delta = value - self.mean
            self.mean += alpha * delta
            self.var = (1.0 - alpha) * (self.var + alpha * delta * delta)
        self.n_closed += 1

    def observe(self, ts: float) -> None:
        """Record an event at ``ts``, closing and zero-filling as needed."""
        bucket = int(ts // self.bucket_s)
        if self._bucket is None:
            self._bucket = bucket
            self._count = 1
            return
        if bucket == self._bucket:
            self._count += 1
            return
        self._close_bucket(float(self._count))
        gap = bucket - self._bucket - 1
        if gap > self.MAX_ZERO_FILL:
            self.mean = 0.0
            self.var = 0.0
            self.n_closed = max(self.n_closed, self.warmup)
        else:
            for _ in range(gap):
                self._close_bucket(0.0)
        self._bucket = bucket
        self._count = 1

    @property
    def ready(self) -> bool:
        return self.n_closed >= self.warmup

    def zscore(self, observed: float) -> float:
        """How anomalous ``observed`` events-per-bucket is. 0.0 until warm."""
        if not self.ready:
            return 0.0
        denom = max(math.sqrt(self.var), math.sqrt(max(self.mean, 0.0)), 1.0)
        return (observed - self.mean) / denom


# ---------------------------------------------------------------------------
# Per-entity state
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _CardState:
    w30: SlidingAggregate = field(
        default_factory=lambda: SlidingAggregate(W_30S)
    )
    w5m: SlidingAggregate = field(
        default_factory=lambda: SlidingAggregate(W_5M, ("merchant",))
    )
    w1h: SlidingAggregate = field(
        default_factory=lambda: SlidingAggregate(W_1H, ("merchant", "device", "city"))
    )
    amount: RunningStats = field(default_factory=RunningStats)
    last_ts: float = 0.0
    last_lat: float = 0.0
    last_lon: float = 0.0
    seen: bool = False


@dataclass(slots=True)
class _DeviceState:
    w30: SlidingAggregate = field(default_factory=lambda: SlidingAggregate(W_30S))
    w5m: SlidingAggregate = field(
        default_factory=lambda: SlidingAggregate(W_5M, ("card", "merchant"))
    )
    w1h: SlidingAggregate = field(default_factory=lambda: SlidingAggregate(W_1H, ("card",)))
    last_ts: float = 0.0


@dataclass(slots=True)
class _IpState:
    w5m: SlidingAggregate = field(default_factory=lambda: SlidingAggregate(W_5M))
    w1h: SlidingAggregate = field(default_factory=lambda: SlidingAggregate(W_1H, ("card",)))
    last_ts: float = 0.0


@dataclass(slots=True)
class _MerchantState:
    w1m: SlidingAggregate = field(default_factory=lambda: SlidingAggregate(W_1M))
    w5m: SlidingAggregate = field(default_factory=lambda: SlidingAggregate(W_5M, ("card",)))
    rate: EwmaRate = field(default_factory=EwmaRate)
    last_ts: float = 0.0


#: Canonical feature order. Models and the audit trail both key off this, so it
#: must stay stable; append new features at the end.
FEATURE_NAMES: tuple[str, ...] = (
    # --- card behaviour ---
    "card_cnt_30s",
    "card_cnt_5m",
    "card_cnt_1h",
    "card_distinct_merchants_5m",
    "card_distinct_merchants_1h",
    "card_distinct_devices_1h",
    "card_distinct_cities_1h",
    "card_fail_ratio_5m",
    "card_tiny_ratio_5m",
    "card_mean_amount_inr_5m",
    "card_amount_z",
    "card_gap_s",
    "card_geo_speed_kmph",
    # --- device behaviour ---
    "device_cnt_30s",
    "device_cnt_5m",
    "device_cnt_1h",
    "device_distinct_cards_5m",
    "device_distinct_cards_1h",
    "device_distinct_merchants_5m",
    "device_fail_ratio_5m",
    # --- network ---
    "ip_cnt_5m",
    "ip_distinct_cards_1h",
    # --- merchant-side ---
    "merchant_cnt_1m",
    "merchant_cnt_5m",
    "merchant_rate_z",
    "merchant_fail_ratio_5m",
    "merchant_tiny_ratio_5m",
    "merchant_distinct_cards_5m",
    # --- this payment ---
    "amount_inr",
    "log_amount",
    "is_tiny",
    "hour_of_day",
)


@dataclass(slots=True)
class FeatureRow:
    """A payment plus its feature vector.

    ``txn`` carries the ground-truth labels, so detectors are handed
    ``row.values`` and never the row itself.
    """

    txn: Transaction
    values: dict[str, float]

    def vector(self) -> list[float]:
        return [self.values[name] for name in FEATURE_NAMES]


class StreamingFeaturizer:
    """Turns an ordered payment stream into feature rows, one pass, O(1)/event."""

    def __init__(self, *, state_ttl_s: float = STATE_TTL_S) -> None:
        self.cards: dict[str, _CardState] = {}
        self.devices: dict[str, _DeviceState] = {}
        self.ips: dict[str, _IpState] = {}
        self.merchants: dict[str, _MerchantState] = {}
        self.state_ttl_s = state_ttl_s
        self.n_processed = 0
        self._last_ts = float("-inf")

    # ---------------------------------------------------------------- memory
    def _sweep(self, now: float) -> None:
        """Drop entities that have gone cold, so memory tracks active traffic."""
        cutoff = now - self.state_ttl_s
        for store in (self.cards, self.devices, self.ips, self.merchants):
            stale = [key for key, state in store.items() if state.last_ts < cutoff]
            for key in stale:
                del store[key]

    def state_size(self) -> dict[str, int]:
        return {
            "cards": len(self.cards),
            "devices": len(self.devices),
            "ips": len(self.ips),
            "merchants": len(self.merchants),
        }

    # --------------------------------------------------------------- process
    def process(self, txn: Transaction) -> FeatureRow:
        if txn.created_at < self._last_ts:
            raise ValueError(
                f"stream out of order: {txn.payment_id} at {txn.created_at} "
                f"follows {self._last_ts}"
            )
        self._last_ts = txn.created_at

        ts = txn.created_at
        card = self.cards.get(txn.card_id)
        if card is None:
            card = self.cards[txn.card_id] = _CardState()
        device = self.devices.get(txn.device_id)
        if device is None:
            device = self.devices[txn.device_id] = _DeviceState()
        ip = self.ips.get(txn.ip)
        if ip is None:
            ip = self.ips[txn.ip] = _IpState()
        merchant = self.merchants.get(txn.merchant_id)
        if merchant is None:
            merchant = self.merchants[txn.merchant_id] = _MerchantState()

        # 1. Age every window out to the current instant.
        for agg in (card.w30, card.w5m, card.w1h, device.w30, device.w5m,
                    device.w1h, ip.w5m, ip.w1h, merchant.w1m, merchant.w5m):
            agg.advance(ts)

        # 2. History-only signals, read before the current payment is inserted.
        #    This is what keeps failure ratios usable pre-authorisation.
        card_fail = _history_ratio(card.w5m.n_failed, card.w5m.count, PRIOR_FAIL_RATE)
        card_tiny = _history_ratio(card.w5m.n_tiny, card.w5m.count, PRIOR_TINY_RATE)
        card_mean_amt = (card.w5m.sum_amount / card.w5m.count / 100.0) if card.w5m.count else 0.0
        card_amount_z = card.amount.zscore(txn.amount_inr)
        device_fail = _history_ratio(device.w5m.n_failed, device.w5m.count, PRIOR_FAIL_RATE)
        merchant_fail = _history_ratio(merchant.w5m.n_failed, merchant.w5m.count, PRIOR_FAIL_RATE)
        merchant_tiny = _history_ratio(merchant.w5m.n_tiny, merchant.w5m.count, PRIOR_TINY_RATE)

        if card.seen:
            gap_s = min(ts - card.last_ts, W_1H)
            distance = haversine_km(card.last_lat, card.last_lon, txn.lat, txn.lon)
            if distance < MIN_GEO_DISTANCE_KM:
                geo_speed = 0.0
            else:
                geo_speed = distance / (max(ts - card.last_ts, MIN_GEO_DT_S) / 3600.0)
        else:
            gap_s = W_1H
            geo_speed = 0.0

        # 3. Merchant rate baseline. observe() first so the EWMA has zero-filled
        #    any quiet stretch before we ask it for a z-score.
        merchant.rate.observe(ts)
        merchant_rate_z = merchant.rate.zscore(merchant.w1m.count + 1)

        is_tiny = txn.amount <= TINY_AMOUNT_PAISE

        values: dict[str, float] = {
            # Counts include the current payment: an alert fires *on* the Nth
            # payment of a burst, not one payment late.
            "card_cnt_30s": float(card.w30.count + 1),
            "card_cnt_5m": float(card.w5m.count + 1),
            "card_cnt_1h": float(card.w1h.count + 1),
            "card_distinct_merchants_5m": float(
                card.w5m.distinct_including("merchant", txn.merchant_id)
            ),
            "card_distinct_merchants_1h": float(
                card.w1h.distinct_including("merchant", txn.merchant_id)
            ),
            "card_distinct_devices_1h": float(
                card.w1h.distinct_including("device", txn.device_id)
            ),
            "card_distinct_cities_1h": float(card.w1h.distinct_including("city", txn.city)),
            "card_fail_ratio_5m": card_fail,
            "card_tiny_ratio_5m": card_tiny,
            "card_mean_amount_inr_5m": card_mean_amt,
            "card_amount_z": card_amount_z,
            "card_gap_s": gap_s,
            "card_geo_speed_kmph": geo_speed,
            "device_cnt_30s": float(device.w30.count + 1),
            "device_cnt_5m": float(device.w5m.count + 1),
            "device_cnt_1h": float(device.w1h.count + 1),
            "device_distinct_cards_5m": float(
                device.w5m.distinct_including("card", txn.card_id)
            ),
            "device_distinct_cards_1h": float(
                device.w1h.distinct_including("card", txn.card_id)
            ),
            "device_distinct_merchants_5m": float(
                device.w5m.distinct_including("merchant", txn.merchant_id)
            ),
            "device_fail_ratio_5m": device_fail,
            "ip_cnt_5m": float(ip.w5m.count + 1),
            "ip_distinct_cards_1h": float(ip.w1h.distinct_including("card", txn.card_id)),
            "merchant_cnt_1m": float(merchant.w1m.count + 1),
            "merchant_cnt_5m": float(merchant.w5m.count + 1),
            "merchant_rate_z": merchant_rate_z,
            "merchant_fail_ratio_5m": merchant_fail,
            "merchant_tiny_ratio_5m": merchant_tiny,
            "merchant_distinct_cards_5m": float(
                merchant.w5m.distinct_including("card", txn.card_id)
            ),
            "amount_inr": txn.amount_inr,
            "log_amount": math.log1p(txn.amount_inr),
            "is_tiny": 1.0 if is_tiny else 0.0,
            "hour_of_day": float(int(((ts + IST_OFFSET_S) // 3600) % 24)),
        }

        # 4. Now fold the current payment into the state for future events.
        failed = txn.failed
        card.w30.add(ts, txn.amount, failed, is_tiny, ())
        card.w5m.add(ts, txn.amount, failed, is_tiny, (txn.merchant_id,))
        card.w1h.add(ts, txn.amount, failed, is_tiny, (txn.merchant_id, txn.device_id, txn.city))
        card.amount.update(txn.amount_inr)
        card.last_ts = ts
        card.last_lat = txn.lat
        card.last_lon = txn.lon
        card.seen = True

        device.w30.add(ts, txn.amount, failed, is_tiny, ())
        device.w5m.add(ts, txn.amount, failed, is_tiny, (txn.card_id, txn.merchant_id))
        device.w1h.add(ts, txn.amount, failed, is_tiny, (txn.card_id,))
        device.last_ts = ts

        ip.w5m.add(ts, txn.amount, failed, is_tiny, ())
        ip.w1h.add(ts, txn.amount, failed, is_tiny, (txn.card_id,))
        ip.last_ts = ts

        merchant.w1m.add(ts, txn.amount, failed, is_tiny, ())
        merchant.w5m.add(ts, txn.amount, failed, is_tiny, (txn.card_id,))
        merchant.last_ts = ts

        self.n_processed += 1
        if self.n_processed % GC_EVERY_N_EVENTS == 0:
            self._sweep(ts)

        return FeatureRow(txn, values)


def extract_all(transactions: list[Transaction]) -> list[FeatureRow]:
    """Convenience one-pass extraction over an already-ordered stream."""
    featurizer = StreamingFeaturizer()
    return [featurizer.process(t) for t in transactions]
