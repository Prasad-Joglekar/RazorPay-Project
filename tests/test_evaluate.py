"""Tests for the metrics. Hand-computed expected values throughout.

A metrics bug is the worst kind of bug in this project: it does not crash, it
just reports the wrong number, and the whole deliverable is a number.
"""

from __future__ import annotations

import unittest

from razerpay_fraud.evaluate import (
    CostModel,
    average_precision,
    candidate_thresholds,
    choose_threshold,
    confusion_at,
    do_nothing_cost,
    episode_outcomes,
    evaluate_cost,
    pattern_breakdown,
    pr_curve,
)
from razerpay_fraud.schema import Episode, Transaction


def txn(pid: str, ts: float, amount_inr: float, *, fraud=False, pattern=None, episode=None):
    return Transaction(
        payment_id=pid,
        created_at=ts,
        amount=int(amount_inr * 100),
        method="card",
        merchant_id="acc_1",
        card_id="card_1",
        device_id="dev_1",
        ip="1.1.1.1",
        city="Mumbai",
        lat=19.0,
        lon=72.8,
        is_fraud=fraud,
        pattern=pattern,
        episode_id=episode,
    )


class TestConfusion(unittest.TestCase):
    def test_hand_computed(self):
        scores = [0.9, 0.8, 0.4, 0.2, 0.95]
        labels = [True, False, True, False, True]
        m = confusion_at(scores, labels, 0.5)
        self.assertEqual((m.tp, m.fp, m.fn, m.tn), (2, 1, 1, 1))
        self.assertAlmostEqual(m.precision, 2 / 3)
        self.assertAlmostEqual(m.recall, 2 / 3)
        self.assertAlmostEqual(m.f1, 2 / 3)
        self.assertAlmostEqual(m.alert_rate, 3 / 5)
        self.assertAlmostEqual(m.false_positive_rate, 1 / 2)

    def test_threshold_is_inclusive(self):
        m = confusion_at([0.5], [True], 0.5)
        self.assertEqual(m.tp, 1)

    def test_empty_positive_class(self):
        m = confusion_at([0.1, 0.2], [False, False], 0.5)
        self.assertEqual(m.recall, 0.0)
        self.assertEqual(m.precision, 0.0)


class TestPRCurve(unittest.TestCase):
    def test_perfect_ranking(self):
        scores = [0.9, 0.8, 0.3, 0.1]
        labels = [True, True, False, False]
        self.assertAlmostEqual(average_precision(scores, labels), 1.0)

    def test_worst_ranking(self):
        scores = [0.9, 0.8, 0.3, 0.1]
        labels = [False, False, True, True]
        # Precision is 1/3 at the first hit and 1/2 at the second.
        expected = 0.5 * (1 / 3) + 0.5 * (2 / 4)
        self.assertAlmostEqual(average_precision(scores, labels), expected)

    def test_ties_grouped_into_one_point(self):
        """Tied scores must produce a single, achievable curve point."""
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [True, False, True, False]
        precisions, recalls, thresholds = pr_curve(scores, labels)
        self.assertEqual(len(precisions), 1)
        self.assertAlmostEqual(precisions[0], 0.5)
        self.assertAlmostEqual(recalls[0], 1.0)
        self.assertEqual(thresholds[0], 0.5)

    def test_no_positives_returns_empty(self):
        self.assertEqual(pr_curve([0.1, 0.2], [False, False]), ([], [], []))
        self.assertEqual(average_precision([0.1], [False]), 0.0)

    def test_candidate_thresholds_include_alert_on_nothing(self):
        grid = candidate_thresholds([0.0, 0.5, 1.0])
        self.assertTrue(max(grid) > 1.0, "must offer a threshold above every score")
        m = confusion_at([0.0, 0.5, 1.0], [False, False, True], max(grid))
        self.assertEqual(m.tp + m.fp, 0)


class TestEpisodes(unittest.TestCase):
    def setUp(self):
        self.txns = [
            txn("p1", 100.0, 10.0, fraud=True, pattern="card_testing", episode="e1"),
            txn("p2", 105.0, 10.0, fraud=True, pattern="card_testing", episode="e1"),
            txn("p3", 110.0, 10.0, fraud=True, pattern="card_testing", episode="e1"),
            txn("p4", 200.0, 500.0),
        ]
        self.episodes = [
            Episode("e1", "card_testing", True, 100.0, 110.0, ["p1", "p2", "p3"])
        ]

    def test_latency_measured_from_episode_start(self):
        outcomes = episode_outcomes(self.txns, [0.0, 0.9, 0.9, 0.0], self.episodes, 0.5)
        self.assertEqual(len(outcomes), 1)
        outcome = outcomes[0]
        self.assertTrue(outcome.detected)
        self.assertAlmostEqual(outcome.latency_s, 5.0)
        self.assertEqual(outcome.payments_before_detection, 1)
        self.assertEqual(outcome.n_flagged, 2)

    def test_missed_episode(self):
        outcomes = episode_outcomes(self.txns, [0.0, 0.0, 0.0, 0.0], self.episodes, 0.5)
        self.assertFalse(outcomes[0].detected)
        self.assertIsNone(outcomes[0].latency_s)
        self.assertEqual(outcomes[0].payments_before_detection, 3)

    def test_one_hit_detects_the_episode(self):
        outcomes = episode_outcomes(self.txns, [0.0, 0.0, 0.9, 0.0], self.episodes, 0.5)
        self.assertTrue(outcomes[0].detected)
        self.assertAlmostEqual(outcomes[0].latency_s, 10.0)

    def test_pattern_breakdown_separates_baseline(self):
        rows = pattern_breakdown(self.txns, [0.9, 0.0, 0.0, 0.9], self.episodes, 0.5)
        by_name = {r.pattern: r for r in rows}
        self.assertEqual(by_name["card_testing"].n_payments, 3)
        self.assertEqual(by_name["card_testing"].n_flagged_payments, 1)
        self.assertEqual(by_name["baseline_legit"].n_payments, 1)
        self.assertEqual(by_name["baseline_legit"].n_flagged_payments, 1)


class TestCostModel(unittest.TestCase):
    def setUp(self):
        self.model = CostModel(
            take_rate=0.02,
            fp_goodwill_inr=40.0,
            chargeback_fee_inr=1500.0,
            review_cost_inr=12.0,
            recovery_rate=1.0,
        )

    def test_component_costs(self):
        self.assertAlmostEqual(self.model.false_positive(1000.0), 20.0 + 40.0 + 12.0)
        self.assertAlmostEqual(self.model.false_negative(1000.0), 2500.0)
        self.assertAlmostEqual(self.model.true_positive(1000.0), 12.0)

    def test_partial_recovery_charges_the_remainder(self):
        model = CostModel(review_cost_inr=10.0, recovery_rate=0.6)
        self.assertAlmostEqual(model.true_positive(1000.0), 10.0 + 400.0)

    def test_strict_mode_charges_every_miss(self):
        txns = [
            txn("p1", 1.0, 100.0, fraud=True, episode="e1"),
            txn("p2", 2.0, 100.0, fraud=True, episode="e1"),
        ]
        episodes = [Episode("e1", "card_testing", True, 1.0, 2.0, ["p1", "p2"])]
        cost = evaluate_cost(txns, [0.9, 0.0], 0.5, self.model, episodes=episodes, mode="strict")
        self.assertEqual(cost.n_tp, 1)
        self.assertEqual(cost.n_fn, 1)
        self.assertAlmostEqual(cost.fn_cost, 1600.0)

    def test_contained_mode_credits_the_block(self):
        """After the first alert in an attack, later payments are prevented."""
        txns = [
            txn("p1", 1.0, 100.0, fraud=True, episode="e1"),
            txn("p2", 2.0, 100.0, fraud=True, episode="e1"),
        ]
        episodes = [Episode("e1", "card_testing", True, 1.0, 2.0, ["p1", "p2"])]
        cost = evaluate_cost(
            txns, [0.9, 0.0], 0.5, self.model, episodes=episodes, mode="contained"
        )
        self.assertEqual(cost.n_tp, 1)
        self.assertEqual(cost.n_fn, 0)
        self.assertEqual(cost.n_prevented, 1)
        self.assertAlmostEqual(cost.fn_cost, 0.0)

    def test_contained_does_not_credit_payments_before_the_alert(self):
        txns = [
            txn("p1", 1.0, 100.0, fraud=True, episode="e1"),
            txn("p2", 2.0, 100.0, fraud=True, episode="e1"),
        ]
        episodes = [Episode("e1", "card_testing", True, 1.0, 2.0, ["p1", "p2"])]
        cost = evaluate_cost(
            txns, [0.0, 0.9], 0.5, self.model, episodes=episodes, mode="contained"
        )
        self.assertEqual(cost.n_fn, 1)  # p1 happened before we knew
        self.assertEqual(cost.n_tp, 1)

    def test_do_nothing_is_the_no_detector_baseline(self):
        txns = [txn("p1", 1.0, 100.0, fraud=True), txn("p2", 2.0, 5000.0)]
        self.assertAlmostEqual(do_nothing_cost(txns, self.model), 1600.0)

    def test_choose_threshold_prefers_lower_cost(self):
        """With FN 50x costlier than FP, the sweep should favour recall."""
        txns = [txn(f"p{i}", float(i), 1000.0, fraud=(i % 20 == 0)) for i in range(200)]
        for t in txns:
            if t.is_fraud:
                t.episode_id = "e1"
        episodes = [
            Episode("e1", "card_testing", True, 0.0, 200.0,
                    [t.payment_id for t in txns if t.is_fraud])
        ]
        scores = [0.9 if t.is_fraud else 0.1 for t in txns]
        choice = choose_threshold(txns, scores, episodes, self.model, mode="strict")
        self.assertGreater(choice.metrics.recall, 0.99)
        self.assertGreater(choice.savings_inr, 0.0)


if __name__ == "__main__":
    unittest.main()
