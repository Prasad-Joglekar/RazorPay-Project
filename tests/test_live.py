"""Tests for the live streaming console.

These exercise the engine headlessly -- no HTTP, no browser. What matters is
that the event plumbing is correct (sequence numbers never go backwards, a late
joiner can catch up without gaps) and that a live run produces the same kind of
answers the offline pipeline does. The HTTP layer on top is a thin adapter.
"""

from __future__ import annotations

import threading
import unittest

from razerpay_fraud.live import EventBus, LiveEngine
from razerpay_fraud.simulator import SimulatorConfig

# Small enough to run in a couple of seconds, large enough that attacks land.
TINY = SimulatorConfig(days=0.6, n_cards=900, rate_scale=1.0)


class TestEventBus(unittest.TestCase):
    def test_sequence_numbers_are_monotonic(self):
        bus = EventBus()
        for i in range(5):
            bus.publish({"type": "tick", "i": i})
        events = bus.since(0, timeout=0)
        self.assertEqual([e["seq"] for e in events], [1, 2, 3, 4, 5])

    def test_since_returns_only_newer_events(self):
        bus = EventBus()
        for i in range(5):
            bus.publish({"type": "tick", "i": i})
        events = bus.since(3, timeout=0)
        self.assertEqual([e["i"] for e in events], [3, 4])

    def test_since_blocks_then_returns_empty(self):
        """A quiet stream must time out rather than hang forever."""
        bus = EventBus()
        bus.publish({"type": "tick"})
        self.assertEqual(bus.since(1, timeout=0.05), [])

    def test_late_joiner_sees_everything_after_its_cursor(self):
        bus = EventBus()
        bus.publish({"type": "a"})
        cursor = bus.seq
        bus.publish({"type": "b"})
        bus.publish({"type": "c"})
        self.assertEqual([e["type"] for e in bus.since(cursor, timeout=0)], ["b", "c"])

    def test_bounded_buffer_drops_oldest(self):
        bus = EventBus(maxlen=10)
        for i in range(25):
            bus.publish({"type": "tick", "i": i})
        events = bus.since(0, timeout=0)
        self.assertEqual(len(events), 10)
        self.assertEqual(events[-1]["i"], 24)

    def test_waiter_is_woken_by_a_publish(self):
        bus = EventBus()
        got = []

        def wait():
            got.extend(bus.since(0, timeout=5.0))

        t = threading.Thread(target=wait)
        t.start()
        bus.publish({"type": "alert"})
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(got), 1)


class TestLiveEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # speed is enormous so the run finishes immediately; the pacing code is
        # still exercised, it just never sleeps.
        cls.engine = LiveEngine(config=TINY, speed=5_000_000.0, loop=False)
        cls.engine.run()

    def test_run_reaches_completion(self):
        self.assertEqual(self.engine.phase, "complete")

    def test_every_payment_in_the_split_was_scored(self):
        expected = len(
            [t for t in self.engine.dataset.transactions
             if t.created_at >= self.engine.dataset.split_ts]
        )
        self.assertEqual(self.engine.n_processed, expected)

    def test_it_finds_fraud_without_drowning_in_alerts(self):
        self.assertGreater(self.engine.n_alerts, 0)
        self.assertGreater(self.engine.n_true_alerts, 0)
        precision = self.engine.n_true_alerts / self.engine.n_alerts
        self.assertGreater(precision, 0.5, "live precision collapsed")
        self.assertLess(
            self.engine.n_alerts / self.engine.n_processed, 0.05,
            "alerting on more than 5% of traffic is not reviewable",
        )

    def test_counters_are_self_consistent(self):
        self.assertLessEqual(self.engine.n_true_alerts, self.engine.n_alerts)
        self.assertLessEqual(self.engine.n_true_alerts, self.engine.n_fraud_seen)
        tally = sum(self.engine.rule_tally.values())
        self.assertEqual(tally, self.engine.n_alerts)

    def test_timeline_bins_are_ordered_and_cover_the_traffic(self):
        minutes = [b["m"] for b in self.engine.timeline]
        self.assertEqual(minutes, sorted(minutes))
        self.assertEqual(len(minutes), len(set(minutes)), "duplicate minute bins")
        self.assertEqual(
            sum(b["n"] for b in self.engine.timeline), self.engine.n_processed
        )

    def test_alerts_carry_a_full_audit_record(self):
        self.assertTrue(self.engine.recent_alerts)
        for alert in self.engine.recent_alerts:
            self.assertTrue(alert["reasons"], "an alert with no stated reason is not auditable")
            self.assertTrue(alert["features"])
            for key in ("payment_id", "score", "amount_inr", "card_id", "top_rule"):
                self.assertIn(key, alert)
            self.assertGreaterEqual(alert["score"], self.engine.threshold)

    def test_snapshot_is_serialisable_and_complete(self):
        import json

        snap = self.engine.snapshot()
        for key in ("seq", "status", "meta", "timeline", "alerts", "rule_tally"):
            self.assertIn(key, snap)
        json.dumps(snap)  # must not raise

    def test_speed_is_clamped_to_a_sane_range(self):
        engine = LiveEngine(config=TINY, speed=300.0, loop=False)
        engine.set_speed(-5)
        self.assertGreaterEqual(engine.speed, 1.0)
        engine.set_speed(10**9)
        self.assertLessEqual(engine.speed, 20_000.0)

    def test_stop_ends_a_run(self):
        engine = LiveEngine(config=TINY, speed=1.0, loop=True)
        thread = threading.Thread(target=engine.run, daemon=True)
        thread.start()
        engine.stop()
        thread.join(timeout=20.0)
        self.assertFalse(thread.is_alive(), "engine did not stop when asked")


if __name__ == "__main__":
    unittest.main()


class TestSeedSweep(unittest.TestCase):
    """The sweep is what puts an error bar on the headline number."""

    def test_a_single_seed_produces_coherent_metrics(self):
        from razerpay_fraud.sweep import run_seed

        result = run_seed(3, days=0.6)
        self.assertGreater(result.n_test, 100)
        self.assertGreaterEqual(result.precision, 0.0)
        self.assertLessEqual(result.precision, 1.0)
        self.assertLessEqual(result.recall, 1.0)
        self.assertGreater(result.n_fraud_episodes, 0)
        import json

        json.dumps(result.as_dict())

    def test_different_seeds_give_different_splits(self):
        from razerpay_fraud.sweep import run_seed

        a = run_seed(1, days=0.6)
        b = run_seed(2, days=0.6)
        self.assertNotEqual(a.n_test, b.n_test)
