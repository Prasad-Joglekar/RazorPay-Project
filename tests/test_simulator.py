"""Tests for the data generator.

These guard the properties the evaluation depends on: the stream is ordered and
reproducible, the labels are internally consistent, and the dev/test cut is a
clean split in time with no episode straddling it. A generator bug here would
quietly invalidate every number in the report.
"""

from __future__ import annotations

import unittest

from razorpay_fraud.schema import FRAUD_PATTERNS, HARD_NEGATIVE_PATTERNS
from razorpay_fraud.simulator import CITIES, Simulator, SimulatorConfig, haversine_km

# A small but structurally complete dataset: every episode type still fires.
SMALL = SimulatorConfig(days=0.8, n_cards=1500, rate_scale=1.0)


class TestGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Simulator(SMALL).generate()

    def test_stream_is_ordered(self):
        times = [t.created_at for t in self.dataset.transactions]
        self.assertEqual(times, sorted(times))

    def test_payment_ids_unique(self):
        ids = [t.payment_id for t in self.dataset.transactions]
        self.assertEqual(len(ids), len(set(ids)))

    def test_deterministic_for_a_seed(self):
        a = Simulator(SimulatorConfig(days=0.4, n_cards=500, seed=99)).generate()
        b = Simulator(SimulatorConfig(days=0.4, n_cards=500, seed=99)).generate()
        self.assertEqual(len(a.transactions), len(b.transactions))
        self.assertEqual(
            [(t.payment_id, t.created_at, t.amount) for t in a.transactions],
            [(t.payment_id, t.created_at, t.amount) for t in b.transactions],
        )

    def test_different_seeds_differ(self):
        a = Simulator(SimulatorConfig(days=0.4, n_cards=500, seed=1)).generate()
        b = Simulator(SimulatorConfig(days=0.4, n_cards=500, seed=2)).generate()
        self.assertNotEqual(
            [t.amount for t in a.transactions][:200],
            [t.amount for t in b.transactions][:200],
        )

    def test_amounts_are_positive_integer_paise(self):
        for t in self.dataset.transactions:
            self.assertIsInstance(t.amount, int)
            self.assertGreater(t.amount, 0)

    def test_status_vocabulary(self):
        for t in self.dataset.transactions:
            self.assertIn(t.status, ("captured", "failed"))

    def test_fraud_rate_is_realistic(self):
        rate = self.dataset.meta["fraud_rate"]
        self.assertTrue(0.001 < rate < 0.08, f"implausible fraud rate {rate}")


class TestLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Simulator(SMALL).generate()

    def test_fraud_flag_matches_pattern(self):
        for t in self.dataset.transactions:
            if t.pattern in FRAUD_PATTERNS:
                self.assertTrue(t.is_fraud, t.payment_id)
            elif t.pattern in HARD_NEGATIVE_PATTERNS or t.pattern is None:
                self.assertFalse(t.is_fraud, t.payment_id)
            else:
                self.fail(f"unknown pattern {t.pattern!r}")

    def test_every_labelled_payment_has_an_episode(self):
        for t in self.dataset.transactions:
            if t.pattern is not None:
                self.assertIsNotNone(t.episode_id, t.payment_id)

    def test_episode_membership_matches_transactions(self):
        by_id = {t.payment_id: t for t in self.dataset.transactions}
        for episode in self.dataset.episodes:
            self.assertTrue(episode.payment_ids, episode.episode_id)
            for pid in episode.payment_ids:
                self.assertIn(pid, by_id)
                self.assertEqual(by_id[pid].episode_id, episode.episode_id)
                self.assertEqual(by_id[pid].is_fraud, episode.is_fraud)

    def test_all_episode_types_present(self):
        patterns = {e.pattern for e in self.dataset.episodes}
        for expected in FRAUD_PATTERNS:
            self.assertIn(expected, patterns)
        for expected in HARD_NEGATIVE_PATTERNS:
            self.assertIn(expected, patterns)

    def test_hard_negatives_are_not_fraud(self):
        for episode in self.dataset.episodes:
            if episode.pattern in HARD_NEGATIVE_PATTERNS:
                self.assertFalse(episode.is_fraud)


class TestSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = Simulator(SMALL).generate()

    def test_split_partitions_the_stream(self):
        dev, test = self.dataset.dev(), self.dataset.test()
        self.assertEqual(len(dev) + len(test), len(self.dataset.transactions))
        self.assertTrue(all(t.created_at < self.dataset.split_ts for t in dev))
        self.assertTrue(all(t.created_at >= self.dataset.split_ts for t in test))

    def test_both_sides_are_substantial(self):
        self.assertGreater(len(self.dataset.dev()), 100)
        self.assertGreater(len(self.dataset.test()), 100)

    def test_no_episode_straddles_the_split(self):
        """An episode spanning the cut would leak dev traffic into test."""
        by_id = {t.payment_id: t for t in self.dataset.transactions}
        split = self.dataset.split_ts
        for episode in self.dataset.episodes:
            times = [by_id[pid].created_at for pid in episode.payment_ids]
            self.assertTrue(
                all(t < split for t in times) or all(t >= split for t in times),
                f"episode {episode.episode_id} ({episode.pattern}) crosses the split",
            )

    def test_test_split_contains_every_attack_type(self):
        """A pattern absent from test cannot be evaluated at all.

        Checked at the default size rather than on ``SMALL``. At 0.8 days there
        are only ~6 geo_impossible episodes and ~30% of them land in the test
        half, so an empty draw is ordinary luck rather than a defect -- and a
        test that fails on luck teaches you to ignore it. At the size the
        project actually reports numbers on, every attack type must be present.
        """
        dataset = Simulator(SimulatorConfig()).generate()
        patterns = {e.pattern for e in dataset.episodes_in(test=True) if e.is_fraud}
        for expected in FRAUD_PATTERNS:
            self.assertIn(expected, patterns)


class TestAttackShapes(unittest.TestCase):
    """Attacks must actually look like the thing they are named after."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = Simulator(SMALL).generate()
        cls.by_id = {t.payment_id: t for t in cls.dataset.transactions}

    def _payments(self, pattern):
        return [t for t in self.dataset.transactions if t.pattern == pattern]

    def test_card_testing_is_tiny_and_bursty(self):
        payments = self._payments("card_testing")
        self.assertTrue(payments)
        self.assertTrue(all(t.amount <= 5_000 for t in payments))
        fail_rate = sum(t.failed for t in payments) / len(payments)
        self.assertGreater(fail_rate, 0.4)

    def test_card_testing_uses_one_card_per_episode(self):
        for episode in self.dataset.episodes:
            if episode.pattern != "card_testing":
                continue
            cards = {self.by_id[pid].card_id for pid in episode.payment_ids}
            self.assertEqual(len(cards), 1)

    def test_velocity_uses_one_device_and_many_cards(self):
        for episode in self.dataset.episodes:
            if episode.pattern != "velocity_enumeration":
                continue
            payments = [self.by_id[pid] for pid in episode.payment_ids]
            self.assertEqual(len({p.device_id for p in payments}), 1)
            self.assertGreater(len({p.card_id for p in payments}), 10)

    def test_geo_impossible_really_is_impossible(self):
        for episode in self.dataset.episodes:
            if episode.pattern != "geo_impossible":
                continue
            meta = episode.meta
            distance = haversine_km(*CITIES[meta["from_city"]], *CITIES[meta["to_city"]])
            implied = distance / (meta["gap_s"] / 3600.0)
            self.assertGreater(implied, 1200.0, episode.episode_id)

    def test_air_travel_is_physically_possible(self):
        """The legit-travel hard negative must stay under aircraft speed."""
        episodes = [e for e in self.dataset.episodes if e.pattern == "air_travel"]
        self.assertTrue(episodes)
        for episode in episodes:
            self.assertLess(episode.meta["implied_kmph"], 750.0, episode.episode_id)

    def test_shared_nat_ip_keeps_one_card_per_device(self):
        """Office staff share an IP, not a device -- that is the whole test."""
        for episode in self.dataset.episodes:
            if episode.pattern != "shared_nat_ip":
                continue
            payments = [self.by_id[pid] for pid in episode.payment_ids]
            self.assertEqual(len({p.ip for p in payments}), 1)
            by_device: dict[str, set[str]] = {}
            for p in payments:
                by_device.setdefault(p.device_id, set()).add(p.card_id)
            self.assertTrue(all(len(cards) == 1 for cards in by_device.values()))

    def test_pos_terminal_is_one_device_one_merchant_many_cards(self):
        """The hard negative aimed at the device-fan-out rules."""
        episodes = [e for e in self.dataset.episodes if e.pattern == "pos_terminal"]
        self.assertTrue(episodes)
        for episode in episodes:
            payments = [self.by_id[pid] for pid in episode.payment_ids]
            self.assertEqual(len({p.device_id for p in payments}), 1)
            self.assertEqual(len({p.merchant_id for p in payments}), 1)
            self.assertEqual(len({p.ip for p in payments}), 1)
            self.assertGreater(len({p.card_id for p in payments}), 15)
            fail_rate = sum(p.failed for p in payments) / len(payments)
            self.assertLess(fail_rate, 0.30, "a real terminal mostly succeeds")

    def test_retry_storm_repeats_one_normal_amount(self):
        for episode in self.dataset.episodes:
            if episode.pattern != "retry_storm":
                continue
            payments = [self.by_id[pid] for pid in episode.payment_ids]
            self.assertEqual(len({p.amount for p in payments}), 1)
            self.assertEqual(len({p.card_id for p in payments}), 1)

    def test_subscription_batch_uses_many_cards_one_amount(self):
        for episode in self.dataset.episodes:
            if episode.pattern != "subscription_batch":
                continue
            payments = [self.by_id[pid] for pid in episode.payment_ids]
            self.assertEqual(len({p.amount for p in payments}), 1)
            self.assertGreater(len({p.card_id for p in payments}), 20)


class TestRazorpayShape(unittest.TestCase):
    def test_export_matches_payment_entity(self):
        dataset = Simulator(SimulatorConfig(days=0.2, n_cards=200)).generate()
        payload = dataset.transactions[0].to_razorpay()
        for key in ("id", "entity", "amount", "currency", "status", "method", "created_at"):
            self.assertIn(key, payload)
        self.assertEqual(payload["entity"], "payment")
        self.assertEqual(payload["currency"], "INR")
        self.assertIsInstance(payload["amount"], int)
        self.assertIsInstance(payload["created_at"], int)
        self.assertTrue(payload["id"].startswith("pay_"))

    def test_export_strips_labels(self):
        """Ground truth must never ride along on an exported payment."""
        dataset = Simulator(SimulatorConfig(days=0.2, n_cards=200)).generate()
        fraud = next(t for t in dataset.transactions if t.is_fraud)
        payload = fraud.to_razorpay()
        flat = str(payload)
        self.assertNotIn("is_fraud", flat)
        self.assertNotIn("episode", flat)
        self.assertNotIn("card_testing", flat)


if __name__ == "__main__":
    unittest.main()
