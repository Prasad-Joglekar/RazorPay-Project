"""Tests for the streaming feature layer.

The causality test is the important one. Everything else in this project rests
on the claim that a payment's features depend only on its predecessors; if that
breaks, the reported precision and recall are fiction and nothing else here
means anything.
"""

from __future__ import annotations

import math
import unittest

from razorpay_fraud.features import (
    EwmaRate,
    RunningStats,
    SlidingAggregate,
    StreamingFeaturizer,
    haversine_km,
)
from razorpay_fraud.schema import Transaction
from razorpay_fraud.simulator import Simulator, SimulatorConfig


def make_txn(
    payment_id: str,
    ts: float,
    *,
    amount: int = 50_000,
    merchant: str = "acc_0001",
    card: str = "card_1",
    device: str = "dev_1",
    ip: str = "1.2.3.4",
    city: str = "Mumbai",
    lat: float = 19.076,
    lon: float = 72.8777,
    status: str = "captured",
) -> Transaction:
    return Transaction(
        payment_id=payment_id,
        created_at=ts,
        amount=amount,
        method="card",
        merchant_id=merchant,
        card_id=card,
        device_id=device,
        ip=ip,
        city=city,
        lat=lat,
        lon=lon,
        status=status,
    )


class TestSlidingAggregate(unittest.TestCase):
    def test_evicts_expired_events(self):
        agg = SlidingAggregate(10.0)
        for i in range(5):
            agg.advance(float(i))
            agg.add(float(i), 100, False, False, ())
        self.assertEqual(agg.count, 5)
        agg.advance(12.0)  # window is (2, 12]: drops ts=0,1,2
        self.assertEqual(agg.count, 2)

    def test_window_is_half_open(self):
        """An event exactly ``window`` seconds old is outside the window."""
        agg = SlidingAggregate(10.0)
        agg.add(0.0, 100, False, False, ())
        agg.advance(10.0)
        self.assertEqual(agg.count, 0)
        agg2 = SlidingAggregate(10.0)
        agg2.add(0.0, 100, False, False, ())
        agg2.advance(9.999)
        self.assertEqual(agg2.count, 1)

    def test_distinct_is_reference_counted(self):
        """Cardinality must drop only when the *last* holder of a key expires."""
        agg = SlidingAggregate(10.0, ("merchant",))
        agg.add(0.0, 100, False, False, ("m1",))
        agg.add(1.0, 100, False, False, ("m1",))
        agg.add(2.0, 100, False, False, ("m2",))
        self.assertEqual(agg.distinct("merchant"), 2)
        agg.advance(10.5)  # drops the ts=0 m1 event only
        self.assertEqual(agg.distinct("merchant"), 2)
        agg.advance(11.5)  # drops the ts=1 m1 event; m1 is now gone
        self.assertEqual(agg.distinct("merchant"), 1)

    def test_distinct_including_does_not_mutate(self):
        agg = SlidingAggregate(10.0, ("merchant",))
        agg.add(0.0, 100, False, False, ("m1",))
        self.assertEqual(agg.distinct_including("merchant", "m2"), 2)
        self.assertEqual(agg.distinct_including("merchant", "m1"), 1)
        self.assertEqual(agg.distinct("merchant"), 1)

    def test_aggregates_track_sums(self):
        agg = SlidingAggregate(100.0)
        agg.add(0.0, 500, True, True, ())
        agg.add(1.0, 1500, False, False, ())
        self.assertEqual(agg.sum_amount, 2000)
        self.assertEqual(agg.n_failed, 1)
        self.assertEqual(agg.n_tiny, 1)
        agg.advance(100.5)  # cutoff 0.5: drops the ts=0.0 event only
        self.assertEqual(agg.sum_amount, 1500)
        self.assertEqual(agg.n_failed, 0)
        self.assertEqual(agg.n_tiny, 0)


class TestEwmaRate(unittest.TestCase):
    def test_not_ready_before_warmup(self):
        rate = EwmaRate(warmup=5)
        for i in range(3):
            rate.observe(i * 60.0)
        self.assertFalse(rate.ready)
        self.assertEqual(rate.zscore(50.0), 0.0)

    def test_zero_fills_quiet_buckets(self):
        """A gap must decay the baseline, not preserve it."""
        rate = EwmaRate(alpha=0.3, warmup=1)
        for i in range(10):  # 10 busy minutes, 5 events each
            for _ in range(5):
                rate.observe(i * 60.0 + 1.0)
        busy_mean = rate.mean
        self.assertGreater(busy_mean, 2.0)
        rate.observe(10 * 60.0 + 60 * 60.0)  # one event an hour later
        self.assertLess(rate.mean, busy_mean)
        self.assertLess(rate.mean, 1.0)

    def test_huge_gap_resets_instead_of_looping(self):
        rate = EwmaRate(alpha=0.3, warmup=1)
        rate.observe(0.0)
        rate.observe(60.0)
        rate.observe(60.0 * (EwmaRate.MAX_ZERO_FILL + 500))
        self.assertEqual(rate.mean, 0.0)
        self.assertEqual(rate.var, 0.0)

    def test_steady_traffic_is_not_anomalous(self):
        """Poisson noise around a stable mean must not produce big z-scores."""
        rate = EwmaRate(alpha=0.05, warmup=10)
        for i in range(200):
            for _ in range(4):
                rate.observe(i * 60.0 + 1.0)
        self.assertLess(abs(rate.zscore(4.0)), 1.0)
        self.assertGreater(rate.zscore(30.0), 5.0)

    def test_only_closed_buckets_count(self):
        """The in-progress bucket must not feed its own baseline."""
        rate = EwmaRate(alpha=0.5, warmup=1)
        rate.observe(0.0)
        self.assertEqual(rate.n_closed, 0)
        rate.observe(60.0)
        self.assertEqual(rate.n_closed, 1)


class TestRunningStats(unittest.TestCase):
    def test_matches_textbook_values(self):
        stats = RunningStats()
        for x in (2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0):
            stats.update(x)
        self.assertAlmostEqual(stats.mean, 5.0)
        self.assertAlmostEqual(stats.std, math.sqrt(32.0 / 7.0), places=9)

    def test_zscore_silent_until_enough_history(self):
        stats = RunningStats()
        stats.update(100.0)
        self.assertEqual(stats.zscore(10_000.0), 0.0)


class TestHaversine(unittest.TestCase):
    def test_known_distances(self):
        # Mumbai -> Delhi is ~1150 km by great circle.
        d = haversine_km(19.0760, 72.8777, 28.7041, 77.1025)
        self.assertTrue(1130 < d < 1170, d)

    def test_zero_for_identical_points(self):
        self.assertAlmostEqual(haversine_km(12.9, 77.5, 12.9, 77.5), 0.0)

    def test_symmetric(self):
        a = haversine_km(19.0, 72.0, 28.0, 77.0)
        b = haversine_km(28.0, 77.0, 19.0, 72.0)
        self.assertAlmostEqual(a, b, places=9)


class TestFeaturizer(unittest.TestCase):
    def test_prefix_causality(self):
        """Feature vectors must not change when later payments are appended.

        Replays a prefix of a real generated stream and a longer stream, and
        requires the overlapping rows to be bit-identical. If any feature ever
        peeked at the future, this fails.
        """
        dataset = Simulator(SimulatorConfig(days=0.35, n_cards=600)).generate()
        txns = dataset.transactions
        self.assertGreater(len(txns), 800)
        cut = len(txns) // 2

        prefix_rows = []
        featurizer = StreamingFeaturizer()
        for txn in txns[:cut]:
            prefix_rows.append(featurizer.process(txn))

        full_rows = []
        featurizer = StreamingFeaturizer()
        for txn in txns:
            full_rows.append(featurizer.process(txn))

        for a, b in zip(prefix_rows, full_rows):
            self.assertEqual(a.txn.payment_id, b.txn.payment_id)
            self.assertEqual(a.values, b.values)

    def test_rejects_out_of_order_stream(self):
        featurizer = StreamingFeaturizer()
        featurizer.process(make_txn("pay_1", 100.0))
        with self.assertRaises(ValueError):
            featurizer.process(make_txn("pay_2", 99.0))

    def test_current_payment_status_does_not_leak(self):
        """Failure ratios describe history, not the payment being scored.

        Two identical streams differing only in the final payment's status must
        produce identical features for that payment -- otherwise the detector
        is using an authorisation outcome it would not have at scoring time.
        """
        def run(final_status: str) -> dict:
            featurizer = StreamingFeaturizer()
            for i in range(4):
                featurizer.process(make_txn(f"pay_{i}", 100.0 + i))
            row = featurizer.process(make_txn("pay_final", 105.0, status=final_status))
            return row.values

        self.assertEqual(run("captured"), run("failed"))

    def test_counts_include_current_payment(self):
        featurizer = StreamingFeaturizer()
        row = featurizer.process(make_txn("pay_1", 0.0))
        self.assertEqual(row.values["card_cnt_30s"], 1.0)
        row = featurizer.process(make_txn("pay_2", 1.0))
        self.assertEqual(row.values["card_cnt_30s"], 2.0)

    def test_card_burst_raises_counts(self):
        featurizer = StreamingFeaturizer()
        row = None
        for i in range(12):
            row = featurizer.process(make_txn(f"pay_{i}", i * 1.5, amount=200))
        self.assertEqual(row.values["card_cnt_30s"], 12.0)
        self.assertGreater(row.values["card_tiny_ratio_5m"], 0.7)

    def test_geo_speed_ignores_intra_city_jitter(self):
        """Two payments seconds apart across town must not imply teleporting."""
        featurizer = StreamingFeaturizer()
        featurizer.process(make_txn("pay_1", 0.0, lat=19.00, lon=72.80))
        row = featurizer.process(make_txn("pay_2", 2.0, lat=19.10, lon=72.95))
        self.assertEqual(row.values["card_geo_speed_kmph"], 0.0)

    def test_geo_speed_flags_impossible_travel(self):
        featurizer = StreamingFeaturizer()
        featurizer.process(make_txn("pay_1", 0.0, city="Mumbai", lat=19.0760, lon=72.8777))
        row = featurizer.process(
            make_txn("pay_2", 600.0, city="Delhi", lat=28.7041, lon=77.1025)
        )
        # ~1150 km in 10 minutes.
        self.assertGreater(row.values["card_geo_speed_kmph"], 6000.0)

    def test_device_card_fanout(self):
        featurizer = StreamingFeaturizer()
        row = None
        for i in range(20):
            row = featurizer.process(
                make_txn(f"pay_{i}", i * 2.0, card=f"card_{i}", device="dev_shared")
            )
        self.assertEqual(row.values["device_distinct_cards_5m"], 20.0)
        self.assertEqual(row.values["card_cnt_5m"], 1.0)

    def test_cold_state_is_swept(self):
        featurizer = StreamingFeaturizer(state_ttl_s=60.0)
        featurizer.process(make_txn("pay_1", 0.0, card="card_old"))
        featurizer._sweep(10_000.0)
        self.assertNotIn("card_old", featurizer.cards)


if __name__ == "__main__":
    unittest.main()
