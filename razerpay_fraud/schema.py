"""Transaction schema.

Field names and units are anchored to the Razorpay Payments API payment entity
(https://razorpay.com/docs/api/payments/) so that swapping the simulator for a
real test-mode webhook feed is a parsing change, not a redesign:

  * ``amount`` is an integer in **paise** (Razorpay never uses floats for money)
  * ``created_at`` is a unix epoch timestamp
  * ids carry Razorpay's prefixes (``pay_``, ``acc_``, ``card_``)

Two deliberate deviations, both documented rather than hidden:

  * ``created_at`` is a float here. Razorpay reports integer seconds, which is
    too coarse to order a card-testing burst that fires 3 payments per second.
    :meth:`Transaction.to_razorpay` truncates it back to an int on export.
  * ``device_id`` / ``ip`` / ``lat`` / ``lon`` are not on the payment entity.
    In production they come from the checkout SDK's fingerprint and the
    authorization request, and would be joined onto the payment by ``payment_id``.

The ``is_fraud`` / ``episode_id`` / ``pattern`` fields are **ground truth**.
They exist only to score the detector and are never visible to feature
extraction -- see :mod:`razerpay_fraud.features`, which only ever reads the
:class:`Transaction` fields listed in ``OBSERVABLE_FIELDS``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

INR = "INR"

#: Methods we simulate, using Razorpay's own vocabulary.
METHODS = ("card", "upi", "netbanking", "wallet")

#: The only fields a detector is allowed to read. Anything outside this set is
#: either a ground-truth label or metadata that would not exist at scoring time.
OBSERVABLE_FIELDS = (
    "payment_id",
    "created_at",
    "amount",
    "currency",
    "method",
    "merchant_id",
    "card_id",
    "device_id",
    "ip",
    "city",
    "lat",
    "lon",
)

#: Anything at or below this value (in paise) is a "tiny" payment. Card testers
#: probe with amounts small enough that the victim ignores the statement line.
TINY_AMOUNT_PAISE = 5_000  # Rs 50


@dataclass(slots=True)
class Transaction:
    """One payment attempt as it arrives on the stream."""

    payment_id: str
    created_at: float
    amount: int
    method: str
    merchant_id: str
    card_id: str
    device_id: str
    ip: str
    city: str
    lat: float
    lon: float
    status: str = "captured"  # captured | failed
    currency: str = INR

    # ---------------- ground truth: never fed to a detector ----------------
    is_fraud: bool = False
    episode_id: Optional[str] = None
    pattern: Optional[str] = None

    @property
    def amount_inr(self) -> float:
        return self.amount / 100.0

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    def to_razorpay(self) -> dict:
        """Render as a Razorpay-shaped payment entity (labels stripped)."""
        return {
            "id": self.payment_id,
            "entity": "payment",
            "amount": self.amount,
            "currency": self.currency,
            "status": "captured" if self.status == "captured" else "failed",
            "method": self.method,
            "card_id": self.card_id if self.method == "card" else None,
            "created_at": int(self.created_at),
            "notes": {"merchant_id": self.merchant_id},
        }


@dataclass(slots=True)
class Episode:
    """A labelled stretch of the stream: an attack, or a hard negative.

    Hard negatives are the point of this class. A detector that only ever sees
    "quiet baseline vs. obvious attack" scores a fake 99% precision, because
    nothing in the data looks like an attack without being one. Each
    :class:`Episode` with ``is_fraud=False`` is a legitimate burst engineered to
    trip a naive rule (flash sale, subscription run, retry storm, office NAT).
    """

    episode_id: str
    pattern: str
    is_fraud: bool
    start_ts: float
    end_ts: float
    payment_ids: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def n_payments(self) -> int:
        return len(self.payment_ids)


#: Attack patterns (``is_fraud=True``).
FRAUD_PATTERNS = ("card_testing", "velocity_enumeration", "geo_impossible")

#: Legitimate patterns that superficially resemble attacks (``is_fraud=False``).
HARD_NEGATIVE_PATTERNS = (
    "flash_sale",
    "subscription_batch",
    "retry_storm",
    "shared_nat_ip",
    "air_travel",
    "pos_terminal",
)


@dataclass(slots=True)
class Dataset:
    """Simulator output: an ordered stream plus its labels and its time split."""

    transactions: list[Transaction]
    episodes: list[Episode]
    split_ts: float
    meta: dict = field(default_factory=dict)

    def dev(self) -> list[Transaction]:
        return [t for t in self.transactions if t.created_at < self.split_ts]

    def test(self) -> list[Transaction]:
        return [t for t in self.transactions if t.created_at >= self.split_ts]

    def episodes_in(self, *, test: bool) -> list[Episode]:
        if test:
            return [e for e in self.episodes if e.start_ts >= self.split_ts]
        return [e for e in self.episodes if e.start_ts < self.split_ts]
