"""Synthetic payment stream with labelled attacks *and* labelled hard negatives.

Design notes that matter for the evaluation being honest:

1. Diurnal baseline. Every merchant has its own daily traffic curve, so a
   global "more than N payments per minute" threshold is useless -- 3 a.m.
   traffic and 8 p.m. traffic differ by roughly 6x. This forces the detector to
   baseline each entity against itself (see the merchant_rate_z feature).

2. Hard negatives. Six legitimate patterns are injected specifically to break
   naive rules -- each one targets a different rule:

     flash_sale           merchant traffic jumps ~12x for 20 min, many distinct
                          cards -- breaks merchant-count rules
     subscription_batch   hundreds of recurring debits in ~2 min from distinct
                          cards -- breaks merchant-burst rules
     retry_storm          one customer retries a failing payment 3-6 times --
                          looks almost exactly like a small card-testing burst
     shared_nat_ip        an office egress IP with dozens of cards behind it --
                          breaks IP-card-cardinality rules
     pos_terminal         a busy shop counter: one device, hundreds of
                          cards, all day -- breaks device-fan-out rules
     air_travel           a customer who flew, then paid at the destination;
                          35% are tight connections implying 500-700 km/h --
                          breaks geo-velocity rules

   Without these, precision is a fiction: nothing in the data would resemble an
   attack without being one, so any threshold would look perfect.

3. Time-ordered split. split_ts cuts the timeline, not the rows. Tuning on a
   random row split would leak future information into past decisions, which is
   exactly the mistake a streaming detector must not be allowed to make.
   Episodes are kept clear of the boundary so that none straddles it.

4. geo_impossible is deliberately low-volume (2-4 payments). It is invisible to
   every counting rule, which is the point: it shows up in the results as the
   pattern that drags recall down, instead of being quietly omitted.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

from .schema import Dataset, Episode, TINY_AMOUNT_PAISE, Transaction

# ---------------------------------------------------------------------------
# Geography: real Indian cities, so haversine distances are realistic.
# ---------------------------------------------------------------------------
CITIES: dict[str, tuple[float, float]] = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Kochi": (9.9312, 76.2673),
    "Guwahati": (26.1445, 91.7362),
    "Chandigarh": (30.7333, 76.7794),
}
CITY_NAMES = list(CITIES)

# Merchant archetypes: (kind, mean payments/min at peak, median ticket in INR,
# lognormal sigma of the ticket, baseline failure rate).
MERCHANT_KINDS = [
    ("food_delivery", 1.60, 320.0, 0.55, 0.055),
    ("ecommerce", 1.20, 1450.0, 0.95, 0.070),
    ("ride_hailing", 1.10, 210.0, 0.60, 0.060),
    ("utility_bills", 0.45, 1900.0, 0.70, 0.045),
    ("edtech", 0.30, 4800.0, 0.80, 0.090),
    ("gaming_topup", 0.70, 150.0, 0.75, 0.085),
    ("saas_subscription", 0.25, 2400.0, 0.50, 0.050),
    ("travel_booking", 0.35, 8600.0, 0.90, 0.105),
]

#: Payments are Indian, so local time is IST (UTC+05:30) and every "hour of
#: day" in this project means an IST hour. Indexing the curve by UTC hour --
#: which is what the first version did -- put the evening peak at 00:30 IST.
IST_OFFSET_S = 19_800

#: Multiplier on the base rate for each IST hour of the day.
DIURNAL = [
    0.18, 0.10, 0.07, 0.06, 0.07, 0.12,  # 00-05
    0.28, 0.52, 0.78, 0.95, 1.05, 1.15,  # 06-11
    1.30, 1.22, 1.00, 0.92, 1.00, 1.18,  # 12-17
    1.45, 1.60, 1.55, 1.20, 0.75, 0.38,  # 18-23
]


def ist_hour(ts: float) -> int:
    """The IST hour of day for a unix timestamp."""
    return int(((ts + IST_OFFSET_S) // 3600) % 24)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r_lat1, r_lat2 = math.radians(lat1), math.radians(lat2)
    d_lat = r_lat2 - r_lat1
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(r_lat1) * math.cos(r_lat2) * math.sin(d_lon / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(a)))


def _lognormal_paise(rng: random.Random, median_inr: float, sigma: float) -> int:
    """Ticket size: lognormal around a median, clamped to a sane payment range."""
    value = median_inr * math.exp(rng.gauss(0.0, sigma))
    return max(100, min(int(round(value * 100)), 5_000_000))


@dataclass(slots=True)
class _Merchant:
    merchant_id: str
    kind: str
    base_rate: float
    median_inr: float
    sigma: float
    fail_rate: float


@dataclass(slots=True)
class _Card:
    card_id: str
    city: str
    devices: list[str]
    ip: str
    weight: float


class SimulatorConfig:
    """Knobs for the generator.

    Defaults produce ~185k payments over 3 days with a 1.04% fraud rate and
    ~5 payments per card per day. The fraud rate is the number that matters:
    at the 2.8% an earlier draft produced, precision is flattered badly,
    because the positive class is easy to hit by accident. Raise
    ``rate_scale`` (more legitimate traffic per merchant) to push it lower
    and make the problem harder.
    """

    def __init__(
        self,
        *,
        seed: int = 7,
        days: float = 3.0,
        n_merchants: int = 24,
        n_cards: int = 12_000,
        start_ts: float = 1_756_000_000.0,
        rate_scale: float = 2.5,
        n_card_testing: int | None = None,
        n_velocity: int | None = None,
        n_geo: int | None = None,
        n_flash_sale: int | None = None,
        n_subscription: int | None = None,
        n_retry_storm: int | None = None,
        n_shared_ip: int | None = None,
        n_air_travel: int | None = None,
        n_pos_terminal: int | None = None,
        test_fraction: float = 0.30,
        boundary_guard_s: float = 1800.0,
    ) -> None:
        self.seed = seed
        self.days = days
        self.n_merchants = n_merchants
        self.n_cards = n_cards
        self.start_ts = start_ts
        self.rate_scale = rate_scale
        self.test_fraction = test_fraction
        self.boundary_guard_s = boundary_guard_s

        # Episode counts are densities, not absolutes. Holding them fixed while
        # the horizon shrinks is how a short run ends up reporting a 38% fraud
        # rate: the attacks stay put while the legitimate traffic underneath
        # them disappears. Scaling with the horizon keeps the fraud rate --
        # the number that governs how hard the problem is -- roughly constant
        # at any --days setting. Explicit values still win.
        def scaled(value: int | None, per_3_days: int, *, floor: int = 1) -> int:
            if value is not None:
                return value
            return max(floor, round(per_3_days * days / 3.0))

        self.n_card_testing = scaled(n_card_testing, 26)
        self.n_velocity = scaled(n_velocity, 18)
        self.n_geo = scaled(n_geo, 22)
        self.n_flash_sale = scaled(n_flash_sale, 10)
        self.n_subscription = scaled(n_subscription, 12)
        self.n_retry_storm = scaled(n_retry_storm, 90)
        self.n_shared_ip = scaled(n_shared_ip, 10)
        self.n_air_travel = scaled(n_air_travel, 70)
        self.n_pos_terminal = scaled(n_pos_terminal, 8)

    @property
    def horizon_s(self) -> float:
        return self.days * 86_400.0

    @property
    def end_ts(self) -> float:
        return self.start_ts + self.horizon_s

    @property
    def split_ts(self) -> float:
        return self.start_ts + self.horizon_s * (1.0 - self.test_fraction)


class Simulator:
    """Generates a labelled Dataset."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.cfg = config or SimulatorConfig()
        self.rng = random.Random(self.cfg.seed)
        self._seq = 0
        # Cards whose location history has been rewritten by a trip, and
        # payments removed because they fell inside a flight. Episodes must
        # not overlap on the same card or the labels become ambiguous.
        self._relocated_cards: set[str] = set()
        self._geo_victim_cards: set[str] = set()
        self._dropped_ids: set[str] = set()
        self.merchants = self._make_merchants()
        self.cards = self._make_cards()
        # Cumulative weights, not raw weights: random.choices rebuilds the
        # cumulative distribution on every call when given weights=, which
        # is O(n_cards) per draw and dominates generation once the card
        # population grows. cum_weights= bisects instead.
        self._cum_weights = list(itertools.accumulate(c.weight for c in self.cards))
        self._cards_by_id = {c.card_id: c for c in self.cards}
        self._cards_by_city: dict[str, list[_Card]] = {}
        for card in self.cards:
            self._cards_by_city.setdefault(card.city, []).append(card)

    # ------------------------------------------------------------------ ids
    def _pid(self) -> str:
        self._seq += 1
        return f"pay_{self._seq:012d}"

    def _new_attacker_ip(self) -> str:
        r = self.rng
        return f"{r.randint(11, 223)}.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}"

    # -------------------------------------------------------------- fixtures
    def _make_merchants(self) -> list[_Merchant]:
        out: list[_Merchant] = []
        for i in range(self.cfg.n_merchants):
            kind, rate, median, sigma, fail = MERCHANT_KINDS[i % len(MERCHANT_KINDS)]
            # Per-merchant scale so same-kind merchants are not clones.
            scale = math.exp(self.rng.gauss(0.0, 0.45))
            out.append(
                _Merchant(
                    merchant_id=f"acc_{i:04d}{kind[:3].upper()}",
                    kind=kind,
                    base_rate=rate * scale * self.cfg.rate_scale,
                    median_inr=median * math.exp(self.rng.gauss(0.0, 0.15)),
                    sigma=sigma,
                    fail_rate=min(0.25, max(0.01, self.rng.gauss(fail, 0.015))),
                )
            )
        return out

    def _make_cards(self) -> list[_Card]:
        r = self.rng
        out: list[_Card] = []
        for i in range(self.cfg.n_cards):
            city = r.choice(CITY_NAMES)
            n_dev = 1 if r.random() < 0.75 else 2
            devices = [f"dev_{i:06d}_{d}" for d in range(n_dev)]
            ip = f"49.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}"
            # Zipf-like popularity: a few cards transact often, most rarely.
            weight = 1.0 / (1.0 + r.random() * 9.0) ** 1.4
            out.append(_Card(f"card_{i:08d}", city, devices, ip, weight))
        return out

    # ------------------------------------------------------------- utilities
    def _jitter_geo(self, city: str) -> tuple[float, float]:
        """Scatter within roughly 15 km of the city centre."""
        lat, lon = CITIES[city]
        return lat + self.rng.gauss(0.0, 0.10), lon + self.rng.gauss(0.0, 0.10)

    def _pick_card(self) -> _Card:
        return self.rng.choices(self.cards, cum_weights=self._cum_weights, k=1)[0]

    def _legit_txn(
        self,
        ts: float,
        merchant: _Merchant,
        card: _Card | None = None,
        *,
        amount: int | None = None,
        force_status: str | None = None,
        pattern: str | None = None,
        episode_id: str | None = None,
    ) -> Transaction:
        card = card or self._pick_card()
        lat, lon = self._jitter_geo(card.city)
        if amount is None:
            amount = _lognormal_paise(self.rng, merchant.median_inr, merchant.sigma)
        if force_status is not None:
            status = force_status
        else:
            status = "failed" if self.rng.random() < merchant.fail_rate else "captured"
        method = self.rng.choices(
            ("card", "upi", "netbanking", "wallet"), weights=(0.42, 0.44, 0.08, 0.06), k=1
        )[0]
        return Transaction(
            payment_id=self._pid(),
            created_at=ts,
            amount=amount,
            method=method,
            merchant_id=merchant.merchant_id,
            card_id=card.card_id,
            device_id=self.rng.choice(card.devices),
            ip=card.ip,
            city=card.city,
            lat=lat,
            lon=lon,
            status=status,
            is_fraud=False,
            pattern=pattern,
            episode_id=episode_id,
        )

    def _sample_start(self, duration: float) -> float:
        """Uniform start time that keeps an episode clear of the dev/test cut."""
        cfg = self.cfg
        for _ in range(200):
            ts = self.rng.uniform(cfg.start_ts + 3600.0, cfg.end_ts - duration - 3600.0)
            near_cut = abs(ts - cfg.split_ts) <= cfg.boundary_guard_s
            near_cut_end = abs(ts + duration - cfg.split_ts) <= cfg.boundary_guard_s
            straddles = ts < cfg.split_ts <= ts + duration
            if not (near_cut or near_cut_end or straddles):
                return ts
        return cfg.start_ts + 3600.0

    # -------------------------------------------------------------- baseline
    def _generate_baseline(self) -> list[Transaction]:
        """Poisson arrivals per merchant per minute, shaped by the diurnal curve."""
        txns: list[Transaction] = []
        cfg = self.cfg
        n_minutes = int(cfg.horizon_s // 60)
        for merchant in self.merchants:
            for minute in range(n_minutes):
                ts0 = cfg.start_ts + minute * 60.0
                hour = ist_hour(ts0)
                lam = merchant.base_rate * DIURNAL[hour]
                # Knuth's Poisson sampler; lam is small so this is cheap.
                count = 0
                target = math.exp(-lam)
                p = self.rng.random()
                while p > target:
                    count += 1
                    p *= self.rng.random()
                for _ in range(count):
                    txns.append(self._legit_txn(ts0 + self.rng.random() * 60.0, merchant))
        return txns

    # -------------------------------------------------------- hard negatives
    def _flash_sale(self, eid: str):
        """Legitimate ~12x traffic spike: many distinct cards, normal tickets."""
        merchant = self.rng.choice(self.merchants)
        duration = self.rng.uniform(900.0, 1800.0)
        start = self._sample_start(duration)
        multiplier = self.rng.uniform(8.0, 16.0)
        hour = ist_hour(start)
        lam_per_s = merchant.base_rate * DIURNAL[hour] * multiplier / 60.0
        n = max(20, int(lam_per_s * duration))
        txns = [
            self._legit_txn(
                start + self.rng.random() * duration,
                merchant,
                pattern="flash_sale",
                episode_id=eid,
            )
            for _ in range(n)
        ]
        txns.sort(key=lambda t: t.created_at)
        episode = Episode(
            eid, "flash_sale", False, start, start + duration,
            [t.payment_id for t in txns],
            {"merchant_id": merchant.merchant_id, "multiplier": round(multiplier, 1)},
        )
        return txns, episode

    def _subscription_batch(self, eid: str):
        """A recurring-mandate run: hundreds of debits, distinct cards, ~2 min."""
        preferred = [
            m for m in self.merchants
            if m.kind in ("saas_subscription", "utility_bills", "edtech")
        ]
        merchant = self.rng.choice(preferred or self.merchants)
        duration = self.rng.uniform(60.0, 180.0)
        start = self._sample_start(duration)
        n = self.rng.randint(60, 220)
        # A mandate run debits the same plan amount from every subscriber.
        plan_amount = _lognormal_paise(self.rng, merchant.median_inr, 0.2)
        txns = []
        for _ in range(n):
            t = self._legit_txn(
                start + self.rng.random() * duration,
                merchant,
                amount=plan_amount,
                pattern="subscription_batch",
                episode_id=eid,
            )
            t.method = "card"
            txns.append(t)
        txns.sort(key=lambda t: t.created_at)
        episode = Episode(
            eid, "subscription_batch", False, start, start + duration,
            [t.payment_id for t in txns],
            {"merchant_id": merchant.merchant_id, "plan_amount": plan_amount},
        )
        return txns, episode

    def _retry_storm(self, eid: str):
        """One frustrated customer retrying a failing payment 3-6 times.

        This is the nastiest hard negative: same card, same device, seconds
        apart, mostly failures. Only the amount separates it from card testing
        -- a retry storm repeats one normal-sized ticket, while a card tester
        walks tiny amounts.
        """
        merchant = self.rng.choice(self.merchants)
        card = self._pick_card()
        n = self.rng.randint(3, 6)
        duration = self.rng.uniform(25.0, 150.0)
        start = self._sample_start(duration)
        amount = _lognormal_paise(self.rng, merchant.median_inr, merchant.sigma)
        device = self.rng.choice(card.devices)
        txns = []
        ts = start
        for i in range(n):
            succeeded = i == n - 1 and self.rng.random() < 0.6
            t = self._legit_txn(
                ts,
                merchant,
                card,
                amount=amount,
                force_status="captured" if succeeded else "failed",
                pattern="retry_storm",
                episode_id=eid,
            )
            t.device_id = device
            txns.append(t)
            ts += duration / n * self.rng.uniform(0.6, 1.4)
        episode = Episode(
            eid, "retry_storm", False, start, txns[-1].created_at,
            [t.payment_id for t in txns],
            {"card_id": card.card_id, "amount": amount},
        )
        return txns, episode

    def _shared_nat_ip(self, eid: str):
        """An office egress IP: many cards on one IP, but one card per device.

        The staff are drawn from cards whose home city *is* the office city.
        Drawing them at random instead would silently make every one of these
        payments look like a teleport (card last seen in its home city, now
        transacting 1500 km away), which would hand the geo rule a fake false
        positive that has nothing to do with NAT. This hard negative is meant
        to stress IP-cardinality rules only -- travel is stressed by
        ``_air_travel``.
        """
        duration = self.rng.uniform(2400.0, 5400.0)
        start = self._sample_start(duration)
        r = self.rng
        office_ip = f"14.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}"
        city = r.choice(CITY_NAMES)
        local = self._cards_by_city.get(city, [])
        if len(local) < 10:
            return None
        n_people = min(len(local), r.randint(25, 60))
        txns = []
        for card in r.sample(local, k=n_people):
            merchant = r.choice(self.merchants)
            for _ in range(r.randint(1, 2)):
                t = self._legit_txn(
                    start + r.random() * duration,
                    merchant,
                    card,
                    pattern="shared_nat_ip",
                    episode_id=eid,
                )
                t.ip = office_ip
                txns.append(t)
        txns.sort(key=lambda t: t.created_at)
        episode = Episode(
            eid, "shared_nat_ip", False, start, start + duration,
            [t.payment_id for t in txns],
            {"ip": office_ip, "n_people": n_people, "city": city},
        )
        return txns, episode

    def _pos_terminal(self, eid: str):
        """A busy in-store card terminal: one device, hundreds of cards, fast.

        This is the hard negative that attacks the *best* rule in the system.
        DEVICE_ENUMERATION keys on one device touching many cards in five
        minutes -- and a queue at a shop counter looks exactly like that. A
        terminal at a busy outlet will run 20-60 distinct cards through one
        device fingerprint in five minutes, all day, entirely legitimately.

        What actually separates it from a card dump being walked:

        * a terminal belongs to **one merchant**; an enumerator sprays many
        * a terminal's decline rate is normal (~6%); enumeration runs 30-60%
        * a terminal's amounts are normal retail tickets

        Without this episode in the data, ``device_cnt_30s`` looks like a
        near-perfect fraud oracle -- which is what the first version of this
        simulator produced, and it was an artefact, not a finding.
        """
        r = self.rng
        merchant = r.choice(self.merchants)
        city = r.choice(CITY_NAMES)
        local = self._cards_by_city.get(city, [])
        if len(local) < 30:
            return None
        duration = r.uniform(2.0 * 3600.0, 5.0 * 3600.0)
        start = self._sample_start(duration)
        device = f"dev_pos_{r.randrange(10 ** 6):06d}"
        store_ip = f"27.{r.randint(0, 255)}.{r.randint(0, 255)}.{r.randint(1, 254)}"
        gap_mean = r.uniform(6.0, 20.0)

        txns = []
        ts = start
        while ts < start + duration and len(txns) < 600:
            card = r.choice(local)
            t = self._legit_txn(
                ts, merchant, card, pattern="pos_terminal", episode_id=eid
            )
            t.device_id = device
            t.ip = store_ip
            t.method = "card"
            txns.append(t)
            # Exponential inter-arrivals: customers queue in clumps, so the
            # terminal genuinely does produce short high-rate bursts.
            ts += r.expovariate(1.0 / gap_mean)
        if not txns:
            return None
        episode = Episode(
            eid, "pos_terminal", False, start, txns[-1].created_at,
            [t.payment_id for t in txns],
            {"merchant_id": merchant.merchant_id, "device_id": device, "city": city},
        )
        return txns, episode

    def _air_travel(self, eid: str, baseline_by_card: dict[str, list[Transaction]]):
        """A customer who legitimately flew, then paid at the destination.

        This is the hard negative that gives the geo-velocity rule an honest
        price. A domestic flight covers 900-2000 km, and the implied
        point-to-point speed once you include getting to and from airports is
        roughly 200-450 km/h -- comfortably under any sane threshold. But a
        fraction of these are *tight* cases (short airport dwell, quick
        connection) that imply 500-700 km/h and start to graze the ramp.
        Those are the false positives a geo rule really produces in
        production, and they are in the test set on purpose.
        """
        r = self.rng
        cfg = self.cfg
        candidates = [
            cid for cid, lst in baseline_by_card.items()
            if len(lst) >= 2 and cid not in self._relocated_cards
        ]
        if not candidates:
            return None
        for _ in range(60):
            card_id = r.choice(candidates)
            history = baseline_by_card[card_id]
            anchor = r.choice(history)
            # Relocating this card would overwrite the labels of any other
            # episode it takes part in after the departure, so only travel
            # cards whose later payments are plain baseline are eligible.
            if any(
                t.created_at > anchor.created_at and t.pattern is not None
                for t in history
            ):
                continue
            far = [
                (c, haversine_km(anchor.lat, anchor.lon, *CITIES[c]))
                for c in CITY_NAMES
                if haversine_km(anchor.lat, anchor.lon, *CITIES[c]) > 800.0
            ]
            if not far:
                continue
            city, distance_km = r.choice(far)
            tight = r.random() < 0.35
            # Cruise ~750 km/h, plus airport overhead at either end.
            overhead_h = r.uniform(0.7, 1.1) if tight else r.uniform(1.8, 3.5)
            gap = (distance_km / 750.0 + overhead_h) * 3600.0
            start = anchor.created_at + gap
            if start > cfg.end_ts - 3600.0 or abs(start - cfg.split_ts) <= cfg.boundary_guard_s:
                continue

            card = self._cards_by_id.get(card_id)
            devices = card.devices if card else [f"dev_trav_{r.randrange(10 ** 6):06d}"]
            txns = []
            ts = start
            for _ in range(r.randint(1, 4)):
                merchant = r.choice(self.merchants)
                lat, lon = self._jitter_geo(city)
                t = Transaction(
                    payment_id=self._pid(),
                    created_at=ts,
                    amount=_lognormal_paise(r, merchant.median_inr, merchant.sigma),
                    method=r.choices(("card", "upi"), weights=(0.6, 0.4), k=1)[0],
                    merchant_id=merchant.merchant_id,
                    card_id=card_id,
                    device_id=r.choice(devices),
                    ip=card.ip if card else self._new_attacker_ip(),
                    city=city,
                    lat=lat,
                    lon=lon,
                    status="failed" if r.random() < 0.07 else "captured",
                    is_fraud=False,
                    pattern="air_travel",
                    episode_id=eid,
                )
                txns.append(t)
                ts += r.uniform(600.0, 3600.0)

            # The arrival window must not span the dev/test cut, or this
            # episode would be tuned on one side and scored on the other.
            if (txns[0].created_at < cfg.split_ts) != (txns[-1].created_at < cfg.split_ts):
                for t in txns:
                    self._dropped_ids.add(t.payment_id)
                continue

            # The customer physically moves, so their other payments move too.
            # Without this the card keeps paying in its home city from
            # mid-flight onwards and manufactures teleports far faster than the
            # flight being modelled. Mid-flight payments are dropped (you do
            # not tap your card at 35,000 feet). One-way trip, so there is no
            # return leg to teleport back on.
            #
            # Payments after arrival are relocated but deliberately left
            # *unlabelled*: once the customer has landed, buying coffee in the
            # destination city is ordinary baseline traffic, not part of a
            # travel event. Labelling the whole tail would stretch this episode
            # to the end of the timeline and across the dev/test split.
            for other in history:
                if anchor.created_at < other.created_at < start:
                    self._dropped_ids.add(other.payment_id)
                elif other.created_at >= start:
                    other.city = city
                    other.lat, other.lon = self._jitter_geo(city)
            self._relocated_cards.add(card_id)
            episode = Episode(
                eid, "air_travel", False, start, txns[-1].created_at,
                [t.payment_id for t in txns],
                {
                    "card_id": card_id,
                    "from_city": anchor.city,
                    "to_city": city,
                    "distance_km": round(distance_km, 1),
                    "implied_kmph": round(distance_km / (gap / 3600.0), 1),
                    "tight_connection": tight,
                },
            )
            return txns, episode
        return None

    # ----------------------------------------------------------------- fraud
    def _card_testing(self, eid: str):
        """Stolen card probed with a burst of tiny payments, mostly declining."""
        r = self.rng
        n = r.randint(12, 55)
        gap_lo, gap_hi = 0.4, 3.0
        merchants = r.sample(self.merchants, k=r.randint(1, 3))
        card_id = f"card_ct_{r.randrange(10 ** 8):08d}"
        device = f"dev_ct_{r.randrange(10 ** 6):06d}"
        ip = self._new_attacker_ip()
        city = r.choice(CITY_NAMES)
        start = self._sample_start(n * gap_hi)
        fail_p = r.uniform(0.6, 0.88)
        txns = []
        ts = start
        for _ in range(n):
            lat, lon = self._jitter_geo(city)
            txns.append(
                Transaction(
                    payment_id=self._pid(),
                    created_at=ts,
                    amount=r.randint(100, TINY_AMOUNT_PAISE),
                    method="card",
                    merchant_id=r.choice(merchants).merchant_id,
                    card_id=card_id,
                    device_id=device,
                    ip=ip,
                    city=city,
                    lat=lat,
                    lon=lon,
                    status="failed" if r.random() < fail_p else "captured",
                    is_fraud=True,
                    pattern="card_testing",
                    episode_id=eid,
                )
            )
            ts += r.uniform(gap_lo, gap_hi)
        episode = Episode(
            eid, "card_testing", True, start, txns[-1].created_at,
            [t.payment_id for t in txns],
            {"card_id": card_id, "n": n, "fail_p": round(fail_p, 2)},
        )
        return txns, episode

    def _velocity_enumeration(self, eid: str):
        """One device/IP cycling a dump of stolen cards across many merchants."""
        r = self.rng
        n_cards = r.randint(20, 70)
        device = f"dev_vel_{r.randrange(10 ** 6):06d}"
        ip = self._new_attacker_ip()
        city = r.choice(CITY_NAMES)
        merchants = r.sample(self.merchants, k=min(len(self.merchants), r.randint(4, 12)))
        fail_p = r.uniform(0.3, 0.6)
        n = int(n_cards * r.uniform(1.0, 1.8))
        start = self._sample_start(n * 5.0)
        stolen = [f"card_vl_{r.randrange(10 ** 8):08d}" for _ in range(n_cards)]
        txns = []
        ts = start
        for _ in range(n):
            merchant = r.choice(merchants)
            lat, lon = self._jitter_geo(city)
            txns.append(
                Transaction(
                    payment_id=self._pid(),
                    created_at=ts,
                    amount=_lognormal_paise(r, min(merchant.median_inr, 1500.0), 0.7),
                    method="card",
                    merchant_id=merchant.merchant_id,
                    card_id=r.choice(stolen),
                    device_id=device,
                    ip=ip,
                    city=city,
                    lat=lat,
                    lon=lon,
                    status="failed" if r.random() < fail_p else "captured",
                    is_fraud=True,
                    pattern="velocity_enumeration",
                    episode_id=eid,
                )
            )
            ts += r.uniform(0.8, 5.0)
        episode = Episode(
            eid, "velocity_enumeration", True, start, txns[-1].created_at,
            [t.payment_id for t in txns],
            {"device_id": device, "n_cards": n_cards},
        )
        return txns, episode

    def _geo_impossible(self, eid: str, baseline_by_card: dict[str, list[Transaction]]):
        """A legit card suddenly transacting a flight away, minutes later.

        Only 2-4 payments, normal-looking in isolation. Nothing about the volume
        is anomalous -- only the implied travel speed is.
        """
        r = self.rng
        cfg = self.cfg
        candidates = [
            cid for cid, lst in baseline_by_card.items()
            if len(lst) >= 3
            and cid not in self._relocated_cards
            and cid not in self._geo_victim_cards
        ]
        if not candidates:
            return None
        for _ in range(60):
            card_id = r.choice(candidates)
            anchor = r.choice(baseline_by_card[card_id])
            gap = r.uniform(240.0, 2100.0)  # 4-35 minutes later
            start = anchor.created_at + gap
            if start > cfg.end_ts - 3600.0 or abs(start - cfg.split_ts) <= cfg.boundary_guard_s:
                continue
            far = [
                c for c in CITY_NAMES
                if haversine_km(anchor.lat, anchor.lon, *CITIES[c]) > 900.0
            ]
            if not far:
                continue
            city = r.choice(far)
            self._geo_victim_cards.add(card_id)
            device = f"dev_geo_{r.randrange(10 ** 6):06d}"
            ip = self._new_attacker_ip()
            txns = []
            ts = start
            for _ in range(r.randint(2, 4)):
                merchant = r.choice(self.merchants)
                lat, lon = self._jitter_geo(city)
                txns.append(
                    Transaction(
                        payment_id=self._pid(),
                        created_at=ts,
                        # Cash-out attempt: larger than this card's usual ticket.
                        amount=_lognormal_paise(r, merchant.median_inr * 3.5, 0.5),
                        method="card",
                        merchant_id=merchant.merchant_id,
                        card_id=card_id,
                        device_id=device,
                        ip=ip,
                        city=city,
                        lat=lat,
                        lon=lon,
                        status="failed" if r.random() < 0.3 else "captured",
                        is_fraud=True,
                        pattern="geo_impossible",
                        episode_id=eid,
                    )
                )
                ts += r.uniform(45.0, 420.0)
            episode = Episode(
                eid, "geo_impossible", True, start, txns[-1].created_at,
                [t.payment_id for t in txns],
                {
                    "card_id": card_id,
                    "from_city": anchor.city,
                    "to_city": city,
                    "gap_s": round(gap, 1),
                },
            )
            return txns, episode
        return None

    # ------------------------------------------------------------------ main
    def generate(self) -> Dataset:
        cfg = self.cfg
        txns = self._generate_baseline()
        episodes: list[Episode] = []

        def add(result) -> None:
            if result is None:
                return
            new_txns, episode = result
            txns.extend(new_txns)
            episodes.append(episode)

        def index_by_card() -> dict[str, list[Transaction]]:
            out: dict[str, list[Transaction]] = {}
            for t in txns:
                if t.payment_id not in self._dropped_ids:
                    out.setdefault(t.card_id, []).append(t)
            for lst in out.values():
                lst.sort(key=lambda t: t.created_at)
            return out

        # Volume-based hard negatives first: they mint extra payments on
        # existing cards, and the location-based episodes below have to see
        # those payments to reason about a card's movement correctly.
        for i in range(cfg.n_flash_sale):
            add(self._flash_sale(f"neg_flash_{i:03d}"))
        for i in range(cfg.n_subscription):
            add(self._subscription_batch(f"neg_sub_{i:03d}"))
        for i in range(cfg.n_retry_storm):
            add(self._retry_storm(f"neg_retry_{i:03d}"))
        for i in range(cfg.n_shared_ip):
            add(self._shared_nat_ip(f"neg_nat_{i:03d}"))
        for i in range(cfg.n_pos_terminal):
            add(self._pos_terminal(f"neg_pos_{i:03d}"))

        # Location episodes: re-index first so a relocated card cannot be left
        # with a stray payment back in its home city.
        by_card = index_by_card()
        for i in range(cfg.n_air_travel):
            add(self._air_travel(f"neg_travel_{i:03d}", by_card))

        for i in range(cfg.n_card_testing):
            add(self._card_testing(f"atk_ct_{i:03d}"))
        for i in range(cfg.n_velocity):
            add(self._velocity_enumeration(f"atk_vel_{i:03d}"))

        by_card = index_by_card()
        for i in range(cfg.n_geo):
            add(self._geo_impossible(f"atk_geo_{i:03d}", by_card))

        if self._dropped_ids:
            txns = [t for t in txns if t.payment_id not in self._dropped_ids]
        txns.sort(key=lambda t: (t.created_at, t.payment_id))

        # Rebuild episode membership from the final stream. Episodes that
        # relocate existing payments (air_travel) claim rows they did not
        # create, so deriving membership from the labels is the only way to
        # keep episode-level scoring consistent with per-payment scoring.
        members: dict[str, list[str]] = {}
        for t in txns:
            if t.episode_id:
                members.setdefault(t.episode_id, []).append(t.payment_id)
        for episode in episodes:
            episode.payment_ids = members.get(episode.episode_id, [])
        episodes = [e for e in episodes if e.payment_ids]

        n_fraud = sum(1 for t in txns if t.is_fraud)
        return Dataset(
            transactions=txns,
            episodes=episodes,
            split_ts=cfg.split_ts,
            meta={
                "seed": cfg.seed,
                "days": cfg.days,
                "n_transactions": len(txns),
                "n_fraud": n_fraud,
                "fraud_rate": n_fraud / max(1, len(txns)),
                "n_merchants": len(self.merchants),
                "n_cards": len(self.cards),
                "start_ts": cfg.start_ts,
                "end_ts": cfg.end_ts,
                "split_ts": cfg.split_ts,
                "n_episodes": len(episodes),
            },
        )
