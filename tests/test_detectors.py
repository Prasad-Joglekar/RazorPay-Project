"""Tests for the rule detector.

The valuable tests here are the hard negatives: each one asserts that a
legitimate pattern which superficially resembles an attack does *not* fire the
rule aimed at that attack. Those are the assertions that break if someone
loosens a threshold to chase recall.
"""

from __future__ import annotations

import unittest

from razerpay_fraud.detectors import (
    NaiveCountDetector,
    RuleDetector,
    ramp,
)
from razerpay_fraud.features import FEATURE_NAMES


def base_values(**overrides) -> dict[str, float]:
    """A quiet, unremarkable payment. Override only what a test is about."""
    values = {name: 0.0 for name in FEATURE_NAMES}
    values.update(
        {
            "card_cnt_30s": 1.0,
            "card_cnt_5m": 1.0,
            "card_cnt_1h": 1.0,
            "card_distinct_merchants_5m": 1.0,
            "card_distinct_merchants_1h": 1.0,
            "card_distinct_devices_1h": 1.0,
            "card_distinct_cities_1h": 1.0,
            "card_fail_ratio_5m": 0.07,
            "card_tiny_ratio_5m": 0.05,
            "card_mean_amount_inr_5m": 500.0,
            "card_gap_s": 3600.0,
            "device_cnt_30s": 1.0,
            "device_cnt_5m": 1.0,
            "device_cnt_1h": 1.0,
            "device_distinct_cards_5m": 1.0,
            "device_distinct_cards_1h": 1.0,
            "device_distinct_merchants_5m": 1.0,
            "device_fail_ratio_5m": 0.07,
            "ip_cnt_5m": 1.0,
            "ip_distinct_cards_1h": 1.0,
            "merchant_cnt_1m": 4.0,
            "merchant_cnt_5m": 20.0,
            "merchant_rate_z": 0.5,
            "merchant_fail_ratio_5m": 0.07,
            "merchant_tiny_ratio_5m": 0.05,
            "merchant_distinct_cards_5m": 20.0,
            "amount_inr": 500.0,
            "log_amount": 6.2,
            "is_tiny": 0.0,
            "hour_of_day": 14.0,
        }
    )
    values.update(overrides)
    return values


class TestRamp(unittest.TestCase):
    def test_endpoints_and_midpoint(self):
        self.assertEqual(ramp(0.0, 10.0, 20.0), 0.0)
        self.assertEqual(ramp(10.0, 10.0, 20.0), 0.0)
        self.assertEqual(ramp(15.0, 10.0, 20.0), 0.5)
        self.assertEqual(ramp(20.0, 10.0, 20.0), 1.0)
        self.assertEqual(ramp(999.0, 10.0, 20.0), 1.0)

    def test_degenerate_range_behaves_as_step(self):
        self.assertEqual(ramp(5.0, 10.0, 10.0), 0.0)
        self.assertEqual(ramp(10.0, 10.0, 10.0), 1.0)


class TestRuleDetector(unittest.TestCase):
    def setUp(self):
        self.detector = RuleDetector()

    # ---------------------------------------------------------- true positives
    def test_card_testing_fires(self):
        values = base_values(
            card_cnt_30s=16.0, card_tiny_ratio_5m=0.9, card_fail_ratio_5m=0.7, is_tiny=1.0
        )
        self.assertGreater(self.detector.score(values), 0.9)
        self.assertEqual(self.detector.reasons(values)[0].rule, "CARD_TESTING")

    def test_device_enumeration_fires(self):
        values = base_values(
            device_distinct_cards_5m=30.0,
            device_cnt_5m=40.0,
            device_distinct_merchants_5m=8.0,
            device_fail_ratio_5m=0.45,
        )
        self.assertGreater(self.detector.score(values), 0.9)
        self.assertEqual(self.detector.reasons(values)[0].rule, "DEVICE_ENUMERATION")

    def test_geo_velocity_fires(self):
        values = base_values(card_geo_speed_kmph=6000.0)
        self.assertEqual(self.detector.rule_scores(values)["GEO_VELOCITY"], 1.0)

    # ------------------------------------------------- hard negatives must not
    def test_retry_storm_does_not_fire(self):
        """Same card, seconds apart, mostly declines -- but normal amounts.

        This is the case that separates a real card-testing rule from a
        card-velocity rule. The tiny-amount term is what saves it.
        """
        values = base_values(
            card_cnt_30s=5.0,
            card_cnt_5m=5.0,
            card_fail_ratio_5m=0.8,
            card_tiny_ratio_5m=0.05,
            amount_inr=1450.0,
        )
        self.assertLess(self.detector.score(values), 0.05)

    def test_flash_sale_does_not_fire(self):
        """A legitimate merchant spike: high rate, but normal composition."""
        values = base_values(
            merchant_rate_z=12.0,
            merchant_cnt_1m=60.0,
            merchant_cnt_5m=300.0,
            merchant_distinct_cards_5m=290.0,
            merchant_fail_ratio_5m=0.07,
            merchant_tiny_ratio_5m=0.05,
        )
        self.assertEqual(self.detector.rule_scores(values)["MERCHANT_UNDER_ATTACK"], 0.0)

    def test_busy_pos_terminal_does_not_fire(self):
        """A shop counter: one device, 30 cards in 5 min, one merchant.

        Fan-out identical to enumeration. The merchant-spread and decline-rate
        terms are the only things standing between this rule and flagging
        every retail terminal on the platform.
        """
        values = base_values(
            device_distinct_cards_5m=30.0,
            device_cnt_5m=32.0,
            device_distinct_cards_1h=180.0,
            device_distinct_merchants_5m=1.0,
            device_fail_ratio_5m=0.06,
            ip_distinct_cards_1h=180.0,
        )
        self.assertEqual(self.detector.rule_scores(values)["DEVICE_ENUMERATION"], 0.0)
        self.assertEqual(self.detector.rule_scores(values)["IP_CARD_FANOUT"], 0.0)

    def test_enumeration_on_a_single_merchant_still_fires(self):
        """The decline-rate half of the signature must stand on its own."""
        values = base_values(
            device_distinct_cards_5m=30.0,
            device_cnt_5m=32.0,
            device_distinct_merchants_5m=1.0,
            device_fail_ratio_5m=0.55,
        )
        self.assertGreater(self.detector.rule_scores(values)["DEVICE_ENUMERATION"], 0.9)

    def test_enumeration_with_clean_cards_still_fires(self):
        """...and so must the merchant-spread half."""
        values = base_values(
            device_distinct_cards_5m=30.0,
            device_cnt_5m=32.0,
            device_distinct_merchants_5m=9.0,
            device_fail_ratio_5m=0.06,
        )
        self.assertGreater(self.detector.rule_scores(values)["DEVICE_ENUMERATION"], 0.9)

    def test_shared_nat_ip_does_not_fire(self):
        """40 cards on one office IP, but each on its own device."""
        values = base_values(ip_distinct_cards_1h=40.0, device_distinct_cards_1h=1.0)
        self.assertEqual(self.detector.rule_scores(values)["IP_CARD_FANOUT"], 0.0)

    def test_ip_fanout_needs_the_device_term(self):
        """The same IP fan-out *with* device concentration is an attack."""
        values = base_values(
            ip_distinct_cards_1h=40.0,
            device_distinct_cards_1h=40.0,
            device_distinct_merchants_5m=8.0,
            device_fail_ratio_5m=0.45,
        )
        self.assertGreater(self.detector.rule_scores(values)["IP_CARD_FANOUT"], 0.5)

    def test_legitimate_air_travel_does_not_fire(self):
        """A tight domestic connection implies ~590 km/h. Must stay silent."""
        values = base_values(card_geo_speed_kmph=590.0)
        self.assertEqual(self.detector.rule_scores(values)["GEO_VELOCITY"], 0.0)

    def test_quiet_payment_scores_zero(self):
        self.assertEqual(self.detector.score(base_values()), 0.0)

    # ------------------------------------------------------------- mechanics
    def test_score_is_max_not_sum(self):
        """Two half-confident rules must not add up to an alert."""
        values = base_values(
            card_cnt_30s=8.0,
            card_tiny_ratio_5m=1.0,
            card_fail_ratio_5m=1.0,
            card_geo_speed_kmph=850.0,
        )
        scores = self.detector.rule_scores(values)
        self.assertAlmostEqual(self.detector.score(values), max(scores.values()))
        self.assertLessEqual(self.detector.score(values), 1.0)

    def test_conjunctive_rule_needs_every_term(self):
        """Dropping any single term of CARD_TESTING must silence it."""
        full = base_values(card_cnt_30s=16.0, card_tiny_ratio_5m=0.9, card_fail_ratio_5m=0.7)
        self.assertGreater(self.detector.rule_scores(full)["CARD_TESTING"], 0.9)
        for missing in ("card_cnt_30s", "card_tiny_ratio_5m", "card_fail_ratio_5m"):
            values = dict(full)
            values[missing] = 0.0
            self.assertEqual(
                self.detector.rule_scores(values)["CARD_TESTING"], 0.0,
                f"rule still fired without {missing}",
            )

    def test_reasons_are_sorted_and_populated(self):
        values = base_values(
            card_cnt_30s=16.0, card_tiny_ratio_5m=0.9, card_fail_ratio_5m=0.7,
            card_geo_speed_kmph=9000.0,
        )
        reasons = self.detector.reasons(values)
        self.assertGreaterEqual(len(reasons), 2)
        self.assertEqual(reasons, sorted(reasons, key=lambda r: r.score, reverse=True))
        for reason in reasons:
            self.assertTrue(reason.detail.strip())
            self.assertIn("rule", reason.as_dict())

    def test_scores_stay_in_unit_interval(self):
        extreme = base_values(
            card_cnt_30s=1e6, card_tiny_ratio_5m=1.0, card_fail_ratio_5m=1.0,
            device_distinct_cards_5m=1e6, device_cnt_5m=1e6, card_geo_speed_kmph=1e9,
            merchant_rate_z=1e6, merchant_tiny_ratio_5m=1.0, merchant_fail_ratio_5m=1.0,
            ip_distinct_cards_1h=1e6, device_distinct_cards_1h=1e6,
        )
        self.assertLessEqual(self.detector.score(extreme), 1.0)
        self.assertGreaterEqual(self.detector.score(base_values()), 0.0)


class TestNaiveDetector(unittest.TestCase):
    def test_fires_on_volume_alone(self):
        detector = NaiveCountDetector()
        self.assertEqual(detector.score(base_values(card_cnt_5m=1.0)), 0.0)
        self.assertEqual(detector.score(base_values(card_cnt_5m=25.0)), 1.0)

    def test_cannot_tell_a_retry_storm_from_card_testing(self):
        """The whole point of the strawman: volume alone conflates them."""
        detector = NaiveCountDetector()
        retry = base_values(card_cnt_5m=6.0, card_tiny_ratio_5m=0.05, amount_inr=1450.0)
        testing = base_values(card_cnt_5m=6.0, card_tiny_ratio_5m=0.95, amount_inr=12.0)
        self.assertEqual(detector.score(retry), detector.score(testing))
        # The rule detector, given the same two, separates them.
        rules = RuleDetector()
        retry.update(card_cnt_30s=6.0, card_fail_ratio_5m=0.8)
        testing.update(card_cnt_30s=6.0, card_fail_ratio_5m=0.8)
        self.assertGreater(rules.score(testing), rules.score(retry))


if __name__ == "__main__":
    unittest.main()
