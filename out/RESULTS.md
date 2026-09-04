# Results

Generated from `187,233` simulated payments over 3 days (24 merchants, 12,000 cards), of which **2,170 (1.16%) are fraudulent**, across 266 labelled episodes.

Thresholds were chosen by minimising expected cost **on the dev split** and then applied unchanged to the held-out test split. Every number in the headline table is from the test split.

## Detector comparison (held-out test split)

| detector | AP | precision | recall | F1 | alert rate | cost (contained) | vs. no detector |
|---|---|---|---|---|---|---|---|
| `rules` | 0.796 | 0.931 | 0.815 | 0.869 | 0.58% | Rs 154,173 | **+85.3%** |
| `naive_card_count` | 0.326 | 0.413 | 0.388 | 0.400 | 0.62% | Rs 413,318 | **+60.5%** |
| `gbdt` | 0.980 | 0.653 | 0.974 | 0.782 | 0.98% | Rs 44,393 | **+95.8%** |
| `isolation_forest` | 0.605 | 0.144 | 0.839 | 0.246 | 3.85% | Rs 484,332 | **+53.7%** |

Doing nothing costs Rs 1,045,948 on this split; declining every payment costs Rs 5,420,465. Both bounds matter -- a detector that beats neither is not worth deploying.

## Dev vs. test (is the threshold overfit?)

| detector | threshold | dev P | dev R | test P | test R | ΔP | ΔR |
|---|---|---|---|---|---|---|---|
| `rules` | 0.0050 | 0.975 | 0.879 | 0.931 | 0.815 | -0.044 | -0.064 |
| `naive_card_count` | 0.0556 | 0.684 | 0.516 | 0.413 | 0.388 | -0.271 | -0.128 |
| `gbdt` | 0.0033 | 0.846 | 1.000 | 0.653 | 0.974 | -0.193 | -0.026 |
| `isolation_forest` | 0.6343 | 0.342 | 0.906 | 0.144 | 0.839 | -0.198 | -0.067 |

## Per-pattern breakdown (test split, rule detector)

Attack patterns are scored on detection; legitimate look-alike patterns are scored on how often they are wrongly flagged. Splitting them out is the whole point -- an aggregate precision figure hides which specific legitimate behaviour a detector cannot tell from fraud.

| pattern | kind | payments | flagged | episodes | detected | median latency |
|---|---|---|---|---|---|---|
| `card_testing` | **attack** | 112 | 85.71% | 4/4 | 100% | 8s |
| `velocity_enumeration` | **attack** | 245 | 83.67% | 5/5 | 100% | 23s |
| `geo_impossible` | **attack** | 22 | 36.36% | 7/7 | 100% | 0s |
| `flash_sale` | legit | 619 | 0.00% | 0/5 | 0% | - |
| `subscription_batch` | legit | 625 | 0.00% | 0/5 | 0% | - |
| `retry_storm` | legit | 142 | 0.00% | 0/30 | 0% | - |
| `shared_nat_ip` | legit | 331 | 0.00% | 0/5 | 0% | - |
| `air_travel` | legit | 70 | 0.00% | 0/26 | 0% | - |
| `pos_terminal` | legit | 1,800 | 0.61% | 1/3 | 33% | 3618s |
| `baseline_legit` | legit | 53,423 | 0.02% | - | - | - |

## Cost model

- false positive: 2.0% of the payment (lost merchant margin) + Rs 40 support/goodwill + Rs 12 review
- false negative: the full payment amount + Rs 1,500 chargeback fee
- true positive: Rs 12 of analyst time

`strict` charges every unflagged fraudulent payment as a full loss. `contained` credits the detector for intervention: once the first payment of an attack is flagged, the card or device is blocked and the rest of that attack is prevented. Production behaves like `contained`, which is why detection latency is a headline number.

| detector | strict cost | strict saving | contained cost | contained saving |
|---|---|---|---|---|
| `rules` | Rs 277,516 | +73.5% | Rs 154,173 | +85.3% |
| `naive_card_count` | Rs 778,319 | +25.6% | Rs 413,318 | +60.5% |
| `gbdt` | Rs 44,393 | +95.8% | Rs 44,393 | +95.8% |
| `isolation_forest` | Rs 517,289 | +50.5% | Rs 484,332 | +53.7% |

### Sensitivity to the cost assumptions

The chosen threshold is only as good as the cost ratio behind it. Re-deriving it under alternative assumptions shows how much of the result is the detector and how much is the accounting.

| assumption | threshold | precision | recall |
|---|---|---|---|
| default | 0.0050 | 0.975 | 0.879 |
| fp_10x_costlier | 0.0050 | 0.975 | 0.879 |
| fp_cheap | 0.0050 | 0.975 | 0.879 |
| no_chargeback_fee | 0.0160 | 0.978 | 0.878 |
| partial_recovery | 0.0050 | 0.975 | 0.879 |

## Streaming performance

- 57,389 payments replayed in 2.37s = **24,235 payments/sec** single-threaded
- per-payment processing latency: p50 17us, p95 37us, p99 80us
- 332 alerts (0.58% of traffic) with a full audit record each

Latency is feature extraction plus scoring, not end-to-end pipeline time -- a real deployment adds broker and network hops on top.

## Feature importance (gbdt, permutation on test)

| feature | importance (drop in AP) |
|---|---|
| `device_cnt_30s` | 0.2917 |
| `card_distinct_cities_1h` | 0.0342 |
| `device_distinct_merchants_5m` | 0.0256 |
| `amount_inr` | 0.0081 |
| `card_amount_z` | 0.0055 |
| `device_fail_ratio_5m` | 0.0042 |
| `card_cnt_1h` | 0.0031 |
| `merchant_cnt_1m` | 0.0021 |
| `is_tiny` | 0.0020 |
| `card_mean_amount_inr_5m` | 0.0019 |
| `device_distinct_cards_1h` | 0.0014 |
| `device_cnt_1h` | 0.0012 |

