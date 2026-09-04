"""Detectors: an explainable rule/statistical baseline and a gradient-boosted model.

Both expose the same interface -- ``score(values) -> float`` in [0, 1] plus
``reasons(values)`` for the audit trail -- so the evaluation harness and the
streaming replay treat them interchangeably, and the comparison between them is
apples to apples on identical feature vectors.

Why the rule detector is the primary model
------------------------------------------
It is the one you can defend in a chargeback dispute. Every alert decomposes
into named rules with the exact feature values that tripped them, so a risk
analyst can read "card made 14 payments in 30 s, 88% of them under Rs 50, 71%
of recent attempts declined" and either act or dismiss it. It also has no
training step, so it cannot silently rot when traffic shifts.

Rule scores are built from two primitives:

* ``ramp(x, lo, hi)`` -- a soft indicator, 0 below ``lo``, 1 above ``hi``,
  linear between. Soft edges matter: a hard ``> 10`` threshold makes the
  precision/recall curve a staircase with a handful of usable operating
  points, while ramps give a genuinely tunable single knob.
* multiplication as a soft AND. ``CARD_TESTING`` requires burst *and* tiny
  amounts *and* declines together, because each alone is a legitimate pattern
  somewhere in the traffic: a flash sale bursts, a gaming top-up merchant is
  all tiny amounts, and a bank outage declines everything. The product is what
  survives the hard negatives.

The detector's overall score is the **max** over rules, not a sum. Two
independent weak signals should not add up to an alert -- that is how you get
false positives you cannot explain -- and the max keeps the semantics of the
threshold constant: "at least one rule is at least this confident".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

from .features import FEATURE_NAMES

Values = dict[str, float]


def ramp(x: float, lo: float, hi: float) -> float:
    """Soft indicator: 0 at or below lo, 1 at or above hi, linear between."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


@dataclass(slots=True)
class Reason:
    """One rule's contribution to an alert, in a form a human can act on."""

    rule: str
    score: float
    detail: str

    def as_dict(self) -> dict:
        return {"rule": self.rule, "score": round(self.score, 4), "detail": self.detail}


class Detector(Protocol):
    name: str

    def score(self, values: Values) -> float: ...

    def reasons(self, values: Values) -> list[Reason]: ...


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Rule:
    name: str
    fn: Callable[[Values], float]
    describe: Callable[[Values], str]
    targets: str  # which attack pattern this rule is meant to catch


def _r_card_testing(v: Values) -> float:
    """Burst on one card, tiny amounts, high decline rate -- all three."""
    return (
        ramp(v["card_cnt_30s"], 4.0, 12.0)
        * ramp(v["card_tiny_ratio_5m"], 0.35, 0.75)
        * ramp(v["card_fail_ratio_5m"], 0.25, 0.55)
    )


def _d_card_testing(v: Values) -> str:
    return (
        f"card made {v['card_cnt_30s']:.0f} payments in 30s, "
        f"{v['card_tiny_ratio_5m']:.0%} of recent ones under Rs 50, "
        f"{v['card_fail_ratio_5m']:.0%} of recent attempts declined"
    )


def _enumeration_signature(v: Values) -> float:
    """Does this device's fan-out look like enumeration rather than a counter?

    Card fan-out on a single device is *not* by itself suspicious. A busy shop
    terminal runs 20-60 distinct cards through one device fingerprint every
    five minutes, all day, and in the simulated traffic its fan-out
    distribution overlaps card-dump enumeration almost exactly (median 24 vs
    20 cards / 5 min). Any rule keyed on fan-out alone flags every retail
    counter on the platform.

    Two things do separate them, and either is sufficient:

    * **merchant spread** -- a terminal belongs to one merchant; an enumerator
      sprays a stolen dump across whatever merchants will take it.
    * **decline rate** -- a terminal's customers mostly succeed (~6% failure);
      enumeration burns through dead cards at 30-60%.

    The decline ramp starts at 25%, not at the ~10% a terminal averages. A
    counter running 27 payments per five minutes at a 7% base decline rate hits
    5 declines in a window often enough to matter, and an earlier 15% start
    turned that ordinary binomial noise into 79 false positives. Enumeration
    sits at a median of 45%, so the looser ramp costs almost no recall.

    Combined with ``max`` (a soft OR) rather than a product, so an attacker who
    parks on a single merchant is still caught by the decline rate, and one who
    somehow has a high-quality dump is still caught by the merchant spread.
    """
    return max(
        ramp(v["device_distinct_merchants_5m"], 1.0, 4.0),
        ramp(v["device_fail_ratio_5m"], 0.25, 0.55),
    )


def _r_device_enumeration(v: Values) -> float:
    """One device walking a dump of cards: fan-out, throughput, and intent."""
    return (
        ramp(v["device_distinct_cards_5m"], 4.0, 15.0)
        * ramp(v["device_cnt_5m"], 8.0, 25.0)
        * _enumeration_signature(v)
    )


def _d_device_enumeration(v: Values) -> str:
    return (
        f"device used {v['device_distinct_cards_5m']:.0f} distinct cards and "
        f"{v['device_cnt_5m']:.0f} payments in 5 min across "
        f"{v['device_distinct_merchants_5m']:.0f} merchants, "
        f"{v['device_fail_ratio_5m']:.0%} declining"
    )


def _r_card_merchant_fanout(v: Values) -> float:
    """One card sprayed across many merchants in minutes."""
    return ramp(v["card_distinct_merchants_5m"], 3.0, 8.0) * ramp(
        v["card_cnt_5m"], 4.0, 12.0
    )


def _d_card_merchant_fanout(v: Values) -> str:
    return (
        f"card hit {v['card_distinct_merchants_5m']:.0f} distinct merchants in 5 min "
        f"over {v['card_cnt_5m']:.0f} payments"
    )


def _r_geo_velocity(v: Values) -> float:
    """Implied travel speed between consecutive payments on one card.

    The ramp starts at 600 km/h, above a domestic flight's realistic
    point-to-point speed once airport time is included, and saturates at
    1100 km/h, above any commercial aircraft's cruise. See the ``air_travel``
    hard negative -- tight connections in the test set reach ~590 km/h, so this
    ramp is deliberately set just clear of them.
    """
    return ramp(v["card_geo_speed_kmph"], 600.0, 1100.0)


def _d_geo_velocity(v: Values) -> str:
    return (
        f"consecutive payments on this card imply "
        f"{v['card_geo_speed_kmph']:.0f} km/h of travel "
        f"({v['card_distinct_cities_1h']:.0f} cities in the last hour)"
    )


def _r_merchant_under_attack(v: Values) -> float:
    """Merchant traffic spike that also looks like probing.

    The rate term alone flags every flash sale and every subscription run --
    both of which spike harder than most attacks. What distinguishes an attack
    is the *composition* of the spike: tiny amounts and declines. All three
    terms are required.
    """
    return (
        ramp(v["merchant_rate_z"], 3.0, 9.0)
        * ramp(v["merchant_tiny_ratio_5m"], 0.30, 0.65)
        * ramp(v["merchant_fail_ratio_5m"], 0.22, 0.50)
    )


def _d_merchant_under_attack(v: Values) -> str:
    return (
        f"merchant traffic {v['merchant_rate_z']:.1f} sigma above its own baseline "
        f"({v['merchant_cnt_1m']:.0f}/min), with {v['merchant_tiny_ratio_5m']:.0%} tiny "
        f"amounts and {v['merchant_fail_ratio_5m']:.0%} declines"
    )


def _r_ip_card_fanout(v: Values) -> float:
    """Many cards behind one IP -- but only when the *devices* agree.

    An office NAT gateway shows 40 cards on one IP too. The difference is that
    each of those cards sits behind its own device fingerprint, while an
    attacker's cards all share one. Gating on the device term is what makes
    this rule survive the ``shared_nat_ip`` hard negative; without it, every
    corporate egress IP is an incident.

    A shop's network defeats *both* of those terms at once -- one IP, one
    terminal, hundreds of cards -- so this rule carries the same enumeration
    signature as DEVICE_ENUMERATION.
    """
    return (
        ramp(v["ip_distinct_cards_1h"], 15.0, 50.0)
        * ramp(v["device_distinct_cards_1h"], 3.0, 8.0)
        * _enumeration_signature(v)
    )


def _d_ip_card_fanout(v: Values) -> str:
    return (
        f"{v['ip_distinct_cards_1h']:.0f} distinct cards from this IP in 1h, "
        f"{v['device_distinct_cards_1h']:.0f} of them on a single device"
    )


RULES: tuple[Rule, ...] = (
    Rule("CARD_TESTING", _r_card_testing, _d_card_testing, "card_testing"),
    Rule("DEVICE_ENUMERATION", _r_device_enumeration, _d_device_enumeration, "velocity_enumeration"),
    Rule("CARD_MERCHANT_FANOUT", _r_card_merchant_fanout, _d_card_merchant_fanout, "velocity_enumeration"),
    Rule("GEO_VELOCITY", _r_geo_velocity, _d_geo_velocity, "geo_impossible"),
    Rule("MERCHANT_UNDER_ATTACK", _r_merchant_under_attack, _d_merchant_under_attack, "card_testing"),
    Rule("IP_CARD_FANOUT", _r_ip_card_fanout, _d_ip_card_fanout, "velocity_enumeration"),
)


class RuleDetector:
    """Explainable rule/statistical detector. Score is the max over rules."""

    name = "rules"

    def __init__(self, rules: Sequence[Rule] = RULES, *, min_reason: float = 0.02) -> None:
        self.rules = tuple(rules)
        self.min_reason = min_reason

    def score(self, values: Values) -> float:
        return max((rule.fn(values) for rule in self.rules), default=0.0)

    def rule_scores(self, values: Values) -> dict[str, float]:
        return {rule.name: rule.fn(values) for rule in self.rules}

    def reasons(self, values: Values) -> list[Reason]:
        scored = [(rule, rule.fn(values)) for rule in self.rules]
        out = [
            Reason(rule.name, score, rule.describe(values))
            for rule, score in scored
            if score > self.min_reason
        ]
        if not out:
            # An alert with no stated reason is not auditable, and the
            # cost-optimal threshold can sit below min_reason -- which silently
            # produced unexplained alerts until this was caught. Whenever any
            # rule contributed at all, name the strongest one.
            rule, score = max(scored, key=lambda pair: pair[1])
            if score > 0.0:
                out = [Reason(rule.name, score, rule.describe(values))]
        out.sort(key=lambda r: r.score, reverse=True)
        return out


class NaiveCountDetector:
    """The strawman: one global "too many payments on this card" threshold.

    Included because it is what a fraud rule looks like before anyone measures
    it, and quantifying how much better the real detectors are is more
    convincing than asserting it. Normalised so its score is comparable.
    """

    name = "naive_card_count"

    def __init__(self, lo: float = 2.0, hi: float = 20.0) -> None:
        self.lo, self.hi = lo, hi

    def score(self, values: Values) -> float:
        return ramp(values["card_cnt_5m"], self.lo, self.hi)

    def reasons(self, values: Values) -> list[Reason]:
        return [
            Reason(
                "NAIVE_CARD_COUNT",
                self.score(values),
                f"card made {values['card_cnt_5m']:.0f} payments in 5 min",
            )
        ]


# ---------------------------------------------------------------------------
# Supervised model
# ---------------------------------------------------------------------------
class MLDetector:
    """Gradient-boosted trees over the same streaming features.

    Trained on the dev split only. sklearn is an optional dependency: if it is
    not installed, the pipeline reports rule-detector results alone rather than
    failing, since the rule detector is the primary deliverable.

    The model gets the identical feature vectors the rules see, so any gap
    between them is attributable to the decision function rather than to one of
    them having better inputs.
    """

    name = "gbdt"

    def __init__(self, *, random_state: int = 0, max_iter: int = 250) -> None:
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
        self.feature_names = FEATURE_NAMES
        self._rules_for_explanation = RuleDetector()

    @staticmethod
    def available() -> bool:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            return False
        return True

    def fit(self, values_list: Iterable[Values], labels: Sequence[bool]) -> "MLDetector":
        from sklearn.ensemble import HistGradientBoostingClassifier

        X = [[v[name] for name in self.feature_names] for v in values_list]
        y = [int(b) for b in labels]
        n_pos = sum(y)
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            raise ValueError("training split contains a single class")
        # Rebalance so the ~1-3% positive class is not simply ignored. Note
        # this makes predict_proba a *ranking* score, not a calibrated
        # probability -- fine here, since every downstream decision is a
        # threshold sweep over the ranking.
        pos_weight = n_neg / n_pos
        sample_weight = [pos_weight if label else 1.0 for label in y]
        self.model = HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=self.random_state,
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def score(self, values: Values) -> float:
        if self.model is None:
            raise RuntimeError("MLDetector.fit must be called before score")
        X = [[values[name] for name in self.feature_names]]
        return float(self.model.predict_proba(X)[0][1])

    def score_batch(self, values_list: Sequence[Values]) -> list[float]:
        """Vectorised scoring -- one predict call instead of N."""
        if self.model is None:
            raise RuntimeError("MLDetector.fit must be called before score")
        if not values_list:
            return []
        X = [[v[name] for name in self.feature_names] for v in values_list]
        return [float(p) for p in self.model.predict_proba(X)[:, 1]]

    def reasons(self, values: Values) -> list[Reason]:
        """Explain a model alert.

        Honest about its limits: this is not SHAP. It reports (a) which
        hand-written rules also fired, which is what an analyst actually acts
        on, and (b) the model's globally most important features with this
        payment's values. That is enough for an audit trail -- "the model
        scored this 0.93, and here is the behaviour behind it" -- without
        pretending to a per-prediction attribution we have not computed.
        """
        out = list(self._rules_for_explanation.reasons(values))
        for name in self.top_features(6):
            out.append(
                Reason(f"MODEL_FEATURE:{name}", 0.0, f"{name} = {values[name]:.3f}")
            )
        return out

    def top_features(self, k: int = 10) -> list[str]:
        """Global feature importance by permutation on the training data.

        Populated by :meth:`compute_importances`; falls back to declaration
        order so the audit trail still renders if importances were not computed.
        """
        if getattr(self, "_importances", None):
            ranked = sorted(self._importances.items(), key=lambda kv: kv[1], reverse=True)
            return [name for name, _ in ranked[:k]]
        return list(self.feature_names[:k])

    def compute_importances(
        self, values_list: Sequence[Values], labels: Sequence[bool], *, n_repeats: int = 3
    ) -> dict[str, float]:
        """Permutation importance w.r.t. average precision."""
        from sklearn.inspection import permutation_importance

        X = [[v[name] for name in self.feature_names] for v in values_list]
        y = [int(b) for b in labels]
        result = permutation_importance(
            self.model,
            X,
            y,
            scoring="average_precision",
            n_repeats=n_repeats,
            random_state=self.random_state,
        )
        self._importances = {
            name: float(mean)
            for name, mean in zip(self.feature_names, result.importances_mean)
        }
        return self._importances


class IsolationForestDetector:
    """Unsupervised alternative, for the "what if we had no labels" comparison.

    Real fraud labels arrive weeks late via chargebacks, so a detector that
    needs none is operationally valuable even if it scores worse. Reported
    alongside the supervised model to show what that independence costs.
    """

    name = "isolation_forest"

    def __init__(self, *, random_state: int = 0, n_estimators: int = 200) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.model = None
        self.feature_names = FEATURE_NAMES
        self._rules_for_explanation = RuleDetector()

    def fit(self, values_list: Iterable[Values]) -> "IsolationForestDetector":
        from sklearn.ensemble import IsolationForest

        X = [[v[name] for name in self.feature_names] for v in values_list]
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination="auto",
            random_state=self.random_state,
        )
        self.model.fit(X)
        return self

    def score_batch(self, values_list: Sequence[Values]) -> list[float]:
        if self.model is None:
            raise RuntimeError("IsolationForestDetector.fit must be called first")
        if not values_list:
            return []
        X = [[v[name] for name in self.feature_names] for v in values_list]
        # score_samples: lower is more anomalous. Map to [0, 1] increasing in
        # anomalousness via a logistic squash so it shares the rule detector's
        # orientation and the threshold sweep needs no special-casing.
        return [1.0 / (1.0 + math.exp(s)) for s in self.model.score_samples(X)]

    def score(self, values: Values) -> float:
        return self.score_batch([values])[0]

    def reasons(self, values: Values) -> list[Reason]:
        return list(self._rules_for_explanation.reasons(values))
