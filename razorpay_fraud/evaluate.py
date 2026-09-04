"""Metrics: per-payment, per-episode, and in rupees.

Three views of the same predictions, because per-payment precision/recall on
its own is misleading for burst attacks:

1. **Per payment.** The standard confusion matrix, precision/recall/F1,
   average precision. This is the pessimistic view: a card-testing burst is 40
   payments, and missing 30 of them scores as 30 misses even if the card was
   blocked on payment 10.

2. **Per episode.** An attack is caught if *any* of its payments is flagged,
   together with the detection latency (seconds from the attack's first payment
   to the first alert). This is the view that matches what a fraud team
   experiences -- you do not need to flag every payment in a burst, you need to
   flag one of them fast enough to block the card. Reported per pattern, so a
   pattern that is never caught cannot hide inside an aggregate.

3. **In rupees.** A false positive and a false negative are not
   interchangeable, and F1 silently assumes they are. A declined legitimate
   payment costs the merchant its margin plus support and goodwill; a missed
   fraud costs the full amount plus a chargeback fee -- on the order of 50x
   different. The operating threshold is chosen by minimising expected cost on
   the dev split, then applied unchanged to the held-out test split.

The cost model is computed two ways, and the gap between them is itself a
result:

* ``strict`` -- every fraudulent payment that scores below the threshold is a
  full loss. Assumes alerts do nothing.
* ``contained`` -- once the first payment of an attack is flagged, the card or
  device is blocked and the rest of that attack is prevented. This is how a
  real system behaves, and it is why detection *latency* is a headline number
  rather than a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .schema import Episode, FRAUD_PATTERNS, HARD_NEGATIVE_PATTERNS, Transaction


# ---------------------------------------------------------------------------
# Per-payment metrics
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PointMetrics:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def alert_rate(self) -> float:
        """Share of all payments that raise an alert -- the review workload."""
        return (self.tp + self.fp) / self.n if self.n else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    def as_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 6),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "alert_rate": round(self.alert_rate, 6),
            "false_positive_rate": round(self.false_positive_rate, 6),
        }


def confusion_at(scores: Sequence[float], labels: Sequence[bool], threshold: float) -> PointMetrics:
    tp = fp = fn = tn = 0
    for s, y in zip(scores, labels):
        flagged = s >= threshold
        if y and flagged:
            tp += 1
        elif y:
            fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    return PointMetrics(threshold, tp, fp, fn, tn)


def pr_curve(
    scores: Sequence[float], labels: Sequence[bool]
) -> tuple[list[float], list[float], list[float]]:
    """Precision/recall/threshold arrays, ties grouped.

    Grouping ties matters here: the rule detector's score is a max over ramps,
    so large blocks of payments share a score of exactly 0.0. Emitting a curve
    point mid-tie would report a precision no threshold can actually achieve.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n_pos = sum(1 for y in labels if y)
    precisions: list[float] = []
    recalls: list[float] = []
    thresholds: list[float] = []
    if n_pos == 0:
        return precisions, recalls, thresholds

    tp = fp = 0
    i = 0
    while i < len(order):
        current = scores[order[i]]
        while i < len(order) and scores[order[i]] == current:
            if labels[order[i]]:
                tp += 1
            else:
                fp += 1
            i += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
        thresholds.append(current)
    return precisions, recalls, thresholds


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Step-wise AP: sum of (recall increment) x precision. No interpolation."""
    precisions, recalls, _ = pr_curve(scores, labels)
    ap = 0.0
    prev_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def candidate_thresholds(scores: Sequence[float], *, max_points: int = 300) -> list[float]:
    """A threshold grid: every distinct score if there are few, else quantiles."""
    unique = sorted(set(scores))
    if len(unique) <= max_points:
        grid = unique
    else:
        step = (len(unique) - 1) / (max_points - 1)
        grid = [unique[min(len(unique) - 1, int(round(k * step)))] for k in range(max_points)]
    # A threshold above every score (alert on nothing) is a legitimate and
    # sometimes cost-optimal operating point, so keep it available.
    top = unique[-1] if unique else 0.0
    return sorted(set(grid) | {math.nextafter(top, math.inf)})


# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class EpisodeOutcome:
    episode_id: str
    pattern: str
    is_fraud: bool
    n_payments: int
    n_flagged: int
    detected: bool
    latency_s: float | None  # first alert, seconds after the episode started
    payments_before_detection: int


def episode_outcomes(
    transactions: Sequence[Transaction],
    scores: Sequence[float],
    episodes: Sequence[Episode],
    threshold: float,
) -> list[EpisodeOutcome]:
    """Resolve every episode to caught/missed plus how long it took."""
    flagged_at: dict[str, float] = {}
    score_by_id: dict[str, float] = {}
    for txn, score in zip(transactions, scores):
        score_by_id[txn.payment_id] = score
        if score >= threshold:
            flagged_at[txn.payment_id] = txn.created_at
    ts_by_id = {t.payment_id: t.created_at for t in transactions}

    out: list[EpisodeOutcome] = []
    for episode in episodes:
        members = [pid for pid in episode.payment_ids if pid in score_by_id]
        if not members:
            continue
        members.sort(key=lambda pid: ts_by_id[pid])
        hits = [pid for pid in members if pid in flagged_at]
        if hits:
            first = min(hits, key=lambda pid: ts_by_id[pid])
            latency = ts_by_id[first] - episode.start_ts
            before = sum(1 for pid in members if ts_by_id[pid] < ts_by_id[first])
        else:
            latency = None
            before = len(members)
        out.append(
            EpisodeOutcome(
                episode_id=episode.episode_id,
                pattern=episode.pattern,
                is_fraud=episode.is_fraud,
                n_payments=len(members),
                n_flagged=len(hits),
                detected=bool(hits),
                latency_s=latency,
                payments_before_detection=before,
            )
        )
    return out


def _median(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


@dataclass(slots=True)
class PatternBreakdown:
    """Per-pattern behaviour: recall for attacks, alert rate for hard negatives."""

    pattern: str
    is_fraud: bool
    n_payments: int
    n_flagged_payments: int
    n_episodes: int
    n_detected_episodes: int
    median_latency_s: float | None
    median_payments_before_detection: float | None

    @property
    def payment_rate(self) -> float:
        return self.n_flagged_payments / self.n_payments if self.n_payments else 0.0

    @property
    def episode_rate(self) -> float:
        return self.n_detected_episodes / self.n_episodes if self.n_episodes else 0.0

    def as_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "is_fraud": self.is_fraud,
            "n_payments": self.n_payments,
            "n_flagged_payments": self.n_flagged_payments,
            "payment_flag_rate": round(self.payment_rate, 4),
            "n_episodes": self.n_episodes,
            "n_detected_episodes": self.n_detected_episodes,
            "episode_detection_rate": round(self.episode_rate, 4),
            "median_latency_s": (
                round(self.median_latency_s, 1) if self.median_latency_s is not None else None
            ),
            "median_payments_before_detection": self.median_payments_before_detection,
        }


def pattern_breakdown(
    transactions: Sequence[Transaction],
    scores: Sequence[float],
    episodes: Sequence[Episode],
    threshold: float,
) -> list[PatternBreakdown]:
    """One row per labelled pattern, plus a row for unlabelled baseline traffic."""
    outcomes = episode_outcomes(transactions, scores, episodes, threshold)
    by_pattern: dict[str, list[EpisodeOutcome]] = {}
    for outcome in outcomes:
        by_pattern.setdefault(outcome.pattern, []).append(outcome)

    payments: dict[str, list[int]] = {}
    for txn, score in zip(transactions, scores):
        key = txn.pattern or "baseline_legit"
        bucket = payments.setdefault(key, [0, 0])
        bucket[0] += 1
        if score >= threshold:
            bucket[1] += 1

    rows: list[PatternBreakdown] = []
    known = list(FRAUD_PATTERNS) + list(HARD_NEGATIVE_PATTERNS) + ["baseline_legit"]
    for pattern in known:
        if pattern not in payments:
            continue
        n_payments, n_flagged = payments[pattern]
        eps = by_pattern.get(pattern, [])
        detected = [e for e in eps if e.detected]
        rows.append(
            PatternBreakdown(
                pattern=pattern,
                is_fraud=pattern in FRAUD_PATTERNS,
                n_payments=n_payments,
                n_flagged_payments=n_flagged,
                n_episodes=len(eps),
                n_detected_episodes=len(detected),
                median_latency_s=_median([e.latency_s for e in detected if e.latency_s is not None]),
                median_payments_before_detection=_median(
                    [float(e.payments_before_detection) for e in detected]
                ),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CostModel:
    """Rupee cost of each decision.

    Defaults are order-of-magnitude figures for Indian card payments, stated
    explicitly so they can be argued with -- which is the point. They are
    inputs to the threshold decision, not results, and
    :func:`cost_sensitivity` re-derives the threshold under alternative values.

    take_rate            merchant margin lost when a good payment is declined
    fp_goodwill_inr      support contact + churn allowance per false decline
    chargeback_fee_inr   scheme + bank admin fee on a disputed payment
    review_cost_inr      analyst time to clear one alert
    recovery_rate        share of value saved when fraud is blocked in time
    """

    take_rate: float = 0.02
    fp_goodwill_inr: float = 40.0
    chargeback_fee_inr: float = 1500.0
    review_cost_inr: float = 12.0
    recovery_rate: float = 1.0

    def false_positive(self, amount_inr: float) -> float:
        return self.take_rate * amount_inr + self.fp_goodwill_inr + self.review_cost_inr

    def false_negative(self, amount_inr: float) -> float:
        return amount_inr + self.chargeback_fee_inr

    def true_positive(self, amount_inr: float) -> float:
        return self.review_cost_inr + (1.0 - self.recovery_rate) * amount_inr

    def as_dict(self) -> dict:
        return {
            "take_rate": self.take_rate,
            "fp_goodwill_inr": self.fp_goodwill_inr,
            "chargeback_fee_inr": self.chargeback_fee_inr,
            "review_cost_inr": self.review_cost_inr,
            "recovery_rate": self.recovery_rate,
        }


@dataclass(slots=True)
class CostBreakdown:
    threshold: float
    fp_cost: float = 0.0
    fn_cost: float = 0.0
    tp_cost: float = 0.0
    n_fp: int = 0
    n_fn: int = 0
    n_tp: int = 0
    n_prevented: int = 0  # contained mode: payments blocked by an earlier alert
    mode: str = "strict"

    @property
    def total(self) -> float:
        return self.fp_cost + self.fn_cost + self.tp_cost

    def as_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 6),
            "mode": self.mode,
            "total_inr": round(self.total, 2),
            "fp_cost_inr": round(self.fp_cost, 2),
            "fn_cost_inr": round(self.fn_cost, 2),
            "tp_cost_inr": round(self.tp_cost, 2),
            "n_fp": self.n_fp,
            "n_fn": self.n_fn,
            "n_tp": self.n_tp,
            "n_prevented": self.n_prevented,
        }


def do_nothing_cost(transactions: Sequence[Transaction], model: CostModel) -> float:
    """Baseline: no detector at all. Every fraud is a loss."""
    return sum(model.false_negative(t.amount_inr) for t in transactions if t.is_fraud)


def block_everything_cost(transactions: Sequence[Transaction], model: CostModel) -> float:
    """The other extreme: decline all traffic. Bounds the useful range."""
    total = 0.0
    for t in transactions:
        total += model.true_positive(t.amount_inr) if t.is_fraud else model.false_positive(
            t.amount_inr
        )
    return total


def evaluate_cost(
    transactions: Sequence[Transaction],
    scores: Sequence[float],
    threshold: float,
    model: CostModel,
    *,
    episodes: Sequence[Episode] | None = None,
    mode: str = "strict",
) -> CostBreakdown:
    """Rupee cost of operating at ``threshold``.

    ``mode="contained"`` credits the detector for intervention: within a fraud
    episode, every payment at or after the first alert is treated as prevented
    (cost = one review), because in production the card or device would already
    be blocked. Payments before the first alert are still full losses. This
    rewards fast detection instead of exhaustive flagging, which is the actual
    operational objective.
    """
    breakdown = CostBreakdown(threshold=threshold, mode=mode)

    first_alert_ts: dict[str, float] = {}
    episode_of: dict[str, str] = {}
    if mode == "contained":
        if episodes is None:
            raise ValueError("contained mode requires episodes")
        fraud_ids = {e.episode_id for e in episodes if e.is_fraud}
        for txn, score in zip(transactions, scores):
            eid = txn.episode_id
            if eid is None or eid not in fraud_ids:
                continue
            episode_of[txn.payment_id] = eid
            if score >= threshold:
                previous = first_alert_ts.get(eid)
                if previous is None or txn.created_at < previous:
                    first_alert_ts[eid] = txn.created_at

    for txn, score in zip(transactions, scores):
        flagged = score >= threshold
        if txn.is_fraud:
            if flagged:
                breakdown.n_tp += 1
                breakdown.tp_cost += model.true_positive(txn.amount_inr)
                continue
            contained_by = first_alert_ts.get(episode_of.get(txn.payment_id, ""))
            if contained_by is not None and txn.created_at >= contained_by:
                # An earlier alert in this attack already blocked the entity.
                breakdown.n_prevented += 1
                breakdown.tp_cost += model.true_positive(txn.amount_inr)
            else:
                breakdown.n_fn += 1
                breakdown.fn_cost += model.false_negative(txn.amount_inr)
        elif flagged:
            breakdown.n_fp += 1
            breakdown.fp_cost += model.false_positive(txn.amount_inr)
    return breakdown


@dataclass(slots=True)
class ThresholdChoice:
    threshold: float
    cost: CostBreakdown
    metrics: PointMetrics
    do_nothing_inr: float
    mode: str

    @property
    def savings_inr(self) -> float:
        return self.do_nothing_inr - self.cost.total

    @property
    def savings_pct(self) -> float:
        return self.savings_inr / self.do_nothing_inr if self.do_nothing_inr else 0.0

    def as_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 6),
            "mode": self.mode,
            "cost": self.cost.as_dict(),
            "metrics": self.metrics.as_dict(),
            "do_nothing_inr": round(self.do_nothing_inr, 2),
            "savings_inr": round(self.savings_inr, 2),
            "savings_pct": round(self.savings_pct, 4),
        }


def choose_threshold(
    transactions: Sequence[Transaction],
    scores: Sequence[float],
    episodes: Sequence[Episode],
    model: CostModel,
    *,
    mode: str = "contained",
    thresholds: Sequence[float] | None = None,
) -> ThresholdChoice:
    """Pick the cost-minimising threshold. Call this on the dev split only."""
    labels = [t.is_fraud for t in transactions]
    grid = list(thresholds) if thresholds is not None else candidate_thresholds(scores)
    best: ThresholdChoice | None = None
    baseline = do_nothing_cost(transactions, model)
    for threshold in grid:
        cost = evaluate_cost(
            transactions, scores, threshold, model, episodes=episodes, mode=mode
        )
        if best is None or cost.total < best.cost.total:
            best = ThresholdChoice(
                threshold=threshold,
                cost=cost,
                metrics=confusion_at(scores, labels, threshold),
                do_nothing_inr=baseline,
                mode=mode,
            )
    assert best is not None
    return best


def cost_curve(
    transactions: Sequence[Transaction],
    scores: Sequence[float],
    episodes: Sequence[Episode],
    model: CostModel,
    *,
    mode: str = "contained",
    thresholds: Sequence[float] | None = None,
) -> list[CostBreakdown]:
    grid = list(thresholds) if thresholds is not None else candidate_thresholds(scores)
    return [
        evaluate_cost(transactions, scores, t, model, episodes=episodes, mode=mode)
        for t in grid
    ]


def cost_sensitivity(
    transactions: Sequence[Transaction],
    scores: Sequence[float],
    episodes: Sequence[Episode],
    *,
    mode: str = "contained",
) -> list[dict]:
    """Re-derive the threshold under different cost assumptions.

    The chosen threshold is only as trustworthy as the cost ratio behind it, so
    this reports how much it moves when that ratio is wrong by an order of
    magnitude in either direction. A threshold that is stable across these is a
    real finding; one that swings wildly means the cost model, not the
    detector, is driving the results.
    """
    variants = {
        "default": CostModel(),
        "fp_10x_costlier": CostModel(fp_goodwill_inr=400.0, take_rate=0.05),
        "fp_cheap": CostModel(fp_goodwill_inr=5.0, take_rate=0.005),
        "no_chargeback_fee": CostModel(chargeback_fee_inr=0.0),
        "partial_recovery": CostModel(recovery_rate=0.6),
    }
    out = []
    for name, model in variants.items():
        choice = choose_threshold(transactions, scores, episodes, model, mode=mode)
        out.append(
            {
                "variant": name,
                "cost_model": model.as_dict(),
                "chosen_threshold": round(choice.threshold, 6),
                "precision": round(choice.metrics.precision, 4),
                "recall": round(choice.metrics.recall, 4),
                "total_inr": round(choice.cost.total, 2),
                "savings_pct": round(choice.savings_pct, 4),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Assembled report for one detector on one split
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SplitReport:
    detector: str
    split: str
    n_payments: int
    n_fraud: int
    threshold: float
    average_precision: float
    metrics: PointMetrics
    cost_strict: CostBreakdown
    cost_contained: CostBreakdown
    do_nothing_inr: float
    block_everything_inr: float
    patterns: list[PatternBreakdown] = field(default_factory=list)
    operating_points: list[PointMetrics] = field(default_factory=list)

    @property
    def savings_strict_pct(self) -> float:
        return (
            (self.do_nothing_inr - self.cost_strict.total) / self.do_nothing_inr
            if self.do_nothing_inr
            else 0.0
        )

    @property
    def savings_contained_pct(self) -> float:
        return (
            (self.do_nothing_inr - self.cost_contained.total) / self.do_nothing_inr
            if self.do_nothing_inr
            else 0.0
        )

    def as_dict(self) -> dict:
        return {
            "detector": self.detector,
            "split": self.split,
            "n_payments": self.n_payments,
            "n_fraud": self.n_fraud,
            "fraud_rate": round(self.n_fraud / self.n_payments, 6) if self.n_payments else 0.0,
            "threshold": round(self.threshold, 6),
            "average_precision": round(self.average_precision, 4),
            "metrics": self.metrics.as_dict(),
            "cost_strict": self.cost_strict.as_dict(),
            "cost_contained": self.cost_contained.as_dict(),
            "do_nothing_inr": round(self.do_nothing_inr, 2),
            "block_everything_inr": round(self.block_everything_inr, 2),
            "savings_strict_pct": round(self.savings_strict_pct, 4),
            "savings_contained_pct": round(self.savings_contained_pct, 4),
            "patterns": [p.as_dict() for p in self.patterns],
            "operating_points": [p.as_dict() for p in self.operating_points],
        }


def build_report(
    detector_name: str,
    split_name: str,
    transactions: Sequence[Transaction],
    scores: Sequence[float],
    episodes: Sequence[Episode],
    threshold: float,
    model: CostModel,
    *,
    extra_operating_points: Sequence[float] = (),
) -> SplitReport:
    labels = [t.is_fraud for t in transactions]
    operating = [confusion_at(scores, labels, t) for t in sorted(set(extra_operating_points))]
    return SplitReport(
        detector=detector_name,
        split=split_name,
        n_payments=len(transactions),
        n_fraud=sum(labels),
        threshold=threshold,
        average_precision=average_precision(scores, labels),
        metrics=confusion_at(scores, labels, threshold),
        cost_strict=evaluate_cost(
            transactions, scores, threshold, model, episodes=episodes, mode="strict"
        ),
        cost_contained=evaluate_cost(
            transactions, scores, threshold, model, episodes=episodes, mode="contained"
        ),
        do_nothing_inr=do_nothing_cost(transactions, model),
        block_everything_inr=block_everything_cost(transactions, model),
        patterns=pattern_breakdown(transactions, scores, episodes, threshold),
        operating_points=operating,
    )
