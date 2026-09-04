# Near-real-time fraud spike detection for a payment stream

Detects sudden, anomalous spikes in fraud-like payment activity — card-testing
bursts, card-dump enumeration, and geographically impossible card use — as an
ordered event stream, one payment at a time, with an explainable alert for every
decision.

The headline is not the F1 score. It is that the evaluation is built to be
hard to fool: the test data contains six **legitimate** traffic patterns that
look like attacks, the operating threshold is chosen on a dev split and never
touched again, and the cost of a false positive is priced in rupees rather than
assumed equal to the cost of a miss.

## Results

The primary number is a **median with a range**, not a single run. Where attack
episodes fall relative to the dev/test cut moves precision several points, and
quoting the seed that looked best is the oldest trick in this genre.

**Rule detector across 10 seeds** (`python -m razorpay_fraud sweep`), threshold
re-chosen on dev for each seed and frozen before scoring test:

| metric | median | range |
|---|---|---|
| precision | **0.950** | 0.931 – 0.978 |
| recall | **0.854** | 0.810 – 0.884 |
| F1 | 0.901 | 0.869 – 0.927 |
| average precision | 0.847 | 0.796 – 0.883 |
| **attack episodes detected** | **100%** | **195 / 195, every seed** |

Every attack episode across all ten held-out splits was caught. Per-payment
recall is ~0.85 because the detector needs a few payments of a burst before it
fires; one flagged payment is enough to block the card.

### One seed in detail

Seed 7 — which the sweep shows is the **worst** of the ten for precision, so
these tables understate the median case. Held-out split: 57,389 payments.

| detector | AP | precision | recall | F1 | alert rate | vs. no detector |
|---|---|---|---|---|---|---|
| **`rules`** (primary) | 0.796 | **0.931** | 0.815 | 0.869 | 0.58% | **+85.3%** |
| `gbdt` | **0.980** | 0.653 | **0.974** | 0.782 | 0.98% | **+95.8%** |
| `isolation_forest` | 0.605 | 0.144 | 0.839 | 0.246 | 3.85% | +53.7% |
| `naive_card_count` (strawman) | 0.326 | 0.413 | 0.388 | 0.400 | 0.62% | +60.5% |

At 1.16% fraud prevalence, so the random-guessing baseline for precision is
0.012. Full tables, per-pattern breakdown and cost curves: **[`out/RESULTS.md`](out/RESULTS.md)**.

**Attacks caught** (an attack counts as caught if any one of its payments is
flagged — that is enough to block the card):

| attack | episodes caught | median detection latency |
|---|---|---|
| `card_testing` | 4 / 4 | 8 s |
| `velocity_enumeration` | 5 / 5 | 23 s |
| `geo_impossible` | 7 / 7 | 0 s |

**Legitimate traffic wrongly flagged** — the number that decides whether this is
deployable:

| legitimate pattern | payments | wrongly flagged |
|---|---|---|
| `flash_sale` | 619 | 0.00% |
| `subscription_batch` | 625 | 0.00% |
| `retry_storm` | 142 | 0.00% |
| `shared_nat_ip` | 331 | 0.00% |
| `air_travel` | 70 | 0.00% |
| `pos_terminal` | 1,800 | 0.61% |
| ordinary traffic | 53,423 | 0.02% |

23 false alarms in total, against 309 fraudulent payments caught. A reviewer
sees 332 alerts across ~21.6 hours of traffic, 93% of them real.

**Streaming:** 24,235 payments/sec single-threaded, p50 17 µs / p99 80 µs
per payment (feature extraction + scoring).

**All times are IST.** Payment traffic is Indian, so the simulator's diurnal
curve and the `hour_of_day` feature are both indexed by IST hour (UTC+05:30) —
peak 19:00, trough 03:00.

## Quickstart

```bash
python -m unittest discover -s tests -t .
```

117 tests, standard library `unittest`, no plugins. The suite includes a
**CLI smoke layer** (`tests/test_cli_smoke.py`) that byte-compiles and imports
every module in the package, builds the argument parser, renders `--help` for
every subcommand, and runs `python -m razorpay_fraud` in a real subprocess.

That layer exists because of a real failure: a batch edit left a syntax error in
`cli.py` and the whole suite still passed, because every other test imports the
library modules directly and nothing ever imported `cli`. Six commands were
broken and 107 green tests said otherwise. The smoke tests are verified against
that exact bug — reintroduce it and the suite goes red.

```bash
python -m razorpay_fraud demo --out out
```

`demo` generates the data, extracts features, trains the comparison models,
picks thresholds on dev, evaluates on test, and writes `out/RESULTS.md`,
`out/report.json`, `out/alerts.jsonl` and four charts. Runs in a few minutes.

The detector, simulator, streaming layer and metrics are **pure standard
library**. `scikit-learn` and `matplotlib` are optional — without them you lose
the comparison models and the charts, and everything else still runs.

### Watching it run — live console

```bash
python -m razorpay_fraud live --speed 300
```

Serves a browser console at `http://127.0.0.1:8800` with the detector **actually
running** in a background thread: it warms state on the dev split, then consumes
the held-out stream in wall-clock time and pushes each alert over Server-Sent
Events the moment its payment is scored. Live precision, recall, throughput, a
trailing traffic chart and a per-rule tally all update as it goes, and every
alert in the feed expands to its full audit record. Speed, pause and restart are
controlled from the page.

SSE rather than WebSockets: the traffic is entirely one-way, it is a plain HTTP
response the standard library serves with no dependency, and it reconnects on
its own. One engine thread owns all detector state and is the only writer, so
there is no lock on the hot path.

It binds to localhost, and ground-truth labels ride along in the payload so the
page can show live precision — both make it a demonstration tool, not something
to expose to a network.

### Deploying

**The console as a static page** — free, no account beyond Hugging Face, no card:

```bash
./deploy/huggingface-static/push-static-space.sh https://huggingface.co/spaces/<user>/<space>
```

The page is entirely self-contained: all 332 alerts, the timeline and the PR
curves are embedded, and it makes no network calls at all. Static hosting runs
it exactly as it runs anywhere else. See
**[deploy/huggingface-static/](deploy/huggingface-static/)**.

**The live SSE console** needs a Python process. Google Cloud Run's always-free
tier covers it (a card must be on file, nothing is charged inside the limits):

```bash
./deploy/cloudrun/deploy.sh
```

Google Cloud Run, whose always-free tier covers this comfortably — the service
scales to zero, so it costs nothing while nobody is watching. Full steps and the
reasoning behind each flag: **[deploy/cloudrun/DEPLOY.md](deploy/cloudrun/DEPLOY.md)**.

The image installs nothing: the detector, simulator, streaming layer and the
console's HTTP server are pure standard library, so it is ~150 MB and builds in
seconds.

A Hugging Face Docker Space works too and needs no card, but Docker Spaces now
require a paid plan — see [deploy/huggingface/](deploy/huggingface/DEPLOY.md).

### Watching it run — terminal

```bash
python -m razorpay_fraud replay --skip-hours 7.8 --hours 1.6 --speed 900
```

The same thing without a browser: replays the held-out stream at `--speed`×, printing a
traffic status line every `--tick` simulated minutes and a full alert the
moment one fires. This is the demo shot: you can watch a card-testing burst
build payment by payment as the score climbs 0.05 → 0.31 → 0.61 → 1.00 and the
explanation updates underneath it —

```
[0.17] pay_000000185249  Rs 33.10  card=card_ct_98540385 merchant=acc_0003UTI
    - CARD_TESTING (0.17): card made 6 payments in 30s, 64% of recent ones
      under Rs 50, 53% of recent attempts declined
[1.00] pay_000000185285  Rs 16.18  card=card_ct_98540385 merchant=acc_0003UTI
    - CARD_TESTING (1.00): card made 17 payments in 30s, 94% of recent ones
      under Rs 50, 62% of recent attempts declined
    - MERCHANT_UNDER_ATTACK (0.34): merchant traffic 6.4 sigma above its own
      baseline (20/min), with 58% tiny amounts and 43% declines
```

`--skip-hours` moves the window without changing the warm-up — state is still
built from every preceding payment, so detection is identical to a full replay.
Attacks are sparse by design, so a window starting at hour 0 may legitimately
contain none. Run `demo` first and read `out/timeline.png` to pick a busy window.

```bash
python -m razorpay_fraud explain --only-false-positives -n 5
```

```bash
python -m razorpay_fraud simulate --razorpay-shape --out out/payments.jsonl
```

## How it works

```
Simulator ──▶ StreamingFeaturizer ──▶ Detector ──▶ AlertRecord ──▶ evaluate
 labelled       O(1)/payment           score        audit trail     P/R/cost
 stream         sliding windows        + reasons    JSONL
```

### 1. Data — `simulator.py`

There is no production data, so the simulator has to be adversarial towards the
detector rather than flattering to it.

Baseline traffic is Poisson arrivals per merchant per minute, shaped by a
per-merchant diurnal curve (3 a.m. and 8 p.m. differ ~6×) with lognormal ticket
sizes and per-merchant decline rates. Cards have home cities, a small set of
devices, and Zipf-like activity.

Three attacks are injected: `card_testing` (one stolen card, 12–55 tiny
payments seconds apart, 60–88% declining), `velocity_enumeration` (one device
walking 20–70 stolen cards across many merchants), and `geo_impossible` (a card
transacting a flight away, minutes later — only 2–4 payments, invisible to any
counting rule).

**Six hard negatives** are injected alongside them, each aimed at a specific
rule:

| pattern | what it breaks |
|---|---|
| `flash_sale` | merchant-rate rules — a legitimate 12× spike |
| `subscription_batch` | merchant-burst rules — 200 mandate debits in 2 min |
| `retry_storm` | card-velocity rules — same card, seconds apart, mostly declining |
| `shared_nat_ip` | IP-cardinality rules — an office IP with 50 cards behind it |
| `pos_terminal` | device-fan-out rules — a shop counter, 27 cards / 5 min, all day |
| `air_travel` | geo-velocity rules — a real flight, 35% tight connections |

Without these, precision is fiction: nothing in the data resembles an attack
without being one, so every threshold looks perfect.

### 2. Features — `features.py`

Per-payment feature extraction is **incremental and causal**. Each entity
(card / device / IP / merchant) owns sliding windows at 30 s / 1 min / 5 min /
1 h, implemented as a deque plus running aggregates with two-pointer eviction —
counts, sums, decline counts, and reference-counted distinct-cardinality
counters, all O(1) amortised per payment. Recomputing a 1-hour distinct-card
count by rescanning would be O(window) per event, which is the difference
between a detector and a batch job.

Merchant traffic is baselined against **itself** with an EWMA over completed
one-minute buckets, zero-filled across quiet stretches, with the deviation
floored at the Poisson `sqrt(mean)` — a merchant averaging 4/min varies by ±2
for free, and calling that an anomaly is how you generate false positives all
night.

Three things are deliberately **excluded**, and this matters more than any
feature that is included:

- **The current payment's own `status`.** Decline ratios come from *prior*
  payments only, so the detector stays deployable pre-authorisation — at scoring
  time you do not yet know whether this attempt will decline.
- **"First time we have ever seen this card."** Every simulated card-testing
  attack mints a fresh card id, so this would be a near-perfect label here and
  would collapse in production, where much legitimate traffic is also
  first-sighting.
- **Instrument type.** 100% of simulated attacks are card payments vs. ~42% of
  legitimate ones. Some of that is real; the simulator exaggerates it into a
  giveaway.

Causality is enforced by a test, not by assertion: `test_prefix_causality`
replays a prefix of the stream and requires the feature vectors to be identical
to the full replay.

### 3. Detection — `detectors.py`

The primary detector is rules + statistics, because it is the one you can
defend in a chargeback dispute. Rules are built from `ramp(x, lo, hi)` soft
indicators combined multiplicatively as a soft AND, and the detector's score is
the **max** over rules — two weak signals should not add up to an alert.

Every conjunction exists because of a specific hard negative:

- `CARD_TESTING` = burst × tiny amounts × declines. Burst alone is a retry
  storm; tiny amounts alone are a gaming top-up merchant; declines alone are a
  bank outage.
- `DEVICE_ENUMERATION` = card fan-out × throughput × *enumeration signature*,
  where the signature is `max(merchant spread, decline rate)`. Fan-out alone
  flags every retail counter.
- `MERCHANT_UNDER_ATTACK` = rate z-score × tiny-amount share × decline share.
  The rate term alone flags every flash sale.
- `GEO_VELOCITY` ramps from 600 to 1100 km/h — above realistic point-to-point
  flight speed, below any aircraft's cruise.

A gradient-boosted model and an isolation forest run on the identical feature
vectors, so the gap between them is attributable to the decision function rather
than to better inputs.

### 4. Evaluation — `evaluate.py`

Three views, because per-payment precision/recall alone is misleading for burst
attacks:

1. **Per payment** — the pessimistic view. Missing 30 payments of a 40-payment
   burst scores as 30 misses even if the card was blocked on payment 10.
2. **Per episode** — an attack is caught if any payment is flagged, plus
   detection latency. This is what a fraud team actually experiences.
3. **In rupees** — a false positive costs the merchant margin plus support and
   goodwill; a miss costs the full amount plus a chargeback fee. Roughly 50×
   different, and F1 silently assumes they are equal.

The threshold is chosen by minimising expected cost **on dev**, then frozen.
Cost is reported two ways: `strict` (alerts do nothing) and `contained` (the
first alert in an attack blocks the entity and prevents the rest — how
production actually behaves, and why latency is a headline number).
`cost_sensitivity` re-derives the threshold under five different cost
assumptions to show how much of the result is the detector and how much is the
accounting.

## What the hard negatives actually caught

Two findings from building this, both of which made the reported numbers *worse*
and the system better.

**The POS terminal.** Before `pos_terminal` existed, the rule detector scored
0.959 precision and the gradient-boosted model showed permutation importance of
**0.62 on `device_cnt_30s`** — one feature carrying essentially all the signal,
with the next at 0.018. That is not a finding, it is an artefact: in the data as
it stood, no legitimate device ever ran many cards quickly. Adding a busy shop
counter (whose card fan-out distribution overlaps enumeration almost exactly —
median 24 vs 20 cards per 5 minutes) dropped rule precision to 0.767 and
collapsed the isolation forest from 0.888 AP to 0.635. Adding the merchant-spread
and decline-rate discriminators recovered precision to a 0.950 median across
seeds on the harder data, and `device_cnt_30s` importance fell to 0.053 with the
signal spread across features.

**Alerts with no reason.** The cost-optimal threshold — 0.0065 on the run that
surfaced this — landed *below*
the detector's `min_reason` floor (0.02), so 20 alerts were emitted with an
empty reasons list — an unexplainable alert, which defeats the point of the
whole design. `reasons()` now always names the strongest contributing rule.

## Honest limitations

- **The data is synthetic.** Every number here is conditional on the simulator
  being a fair model of payment traffic. The hard negatives are the main defence
  against self-congratulation, but real traffic will contain look-alike patterns
  nobody thought to inject.
- **`geo_impossible` per-payment recall is 39%**, against 100% episode
  detection. Only the payment *arriving* in the far city shows the velocity jump;
  the ones after it look local. The episode metric is the honest one here, and
  the per-payment number is reported rather than quietly dropped.
- **Every false alarm is accounted for**, which is reassuring about the
  detector and a caution about the data — there is no residual pile of
  unexplained errors, because the simulator contains only the confusions that
  were deliberately put in it:

  | # | rule | what it actually was |
  |---|---|---|
  | 13 | `MERCHANT_UNDER_ATTACK` | innocent bystanders — every one is a legitimate payment at a merchant that was genuinely under card-testing attack within 5 minutes |
  | 10 | `DEVICE_ENUMERATION` / `IP_CARD_FANOUT` | busy POS terminals in an unlucky run of declines |
  | 3 | `GEO_VELOCITY` | the victim's own legitimate payment, alternating with remote fraud on the same card |

- **Merchant-level rules cause collateral damage by construction.** All 13
  bystander alerts are arguably correct behaviour — during a live attack you
  would throttle the merchant and accept some friction — but they are counted
  as false positives here rather than excused, because the customer whose
  payment is held does not care why.
- **Geo-velocity flags the legitimate leg.** When a compromised card is used
  remotely while the victim keeps paying at home, the *victim's* next legitimate
  payment also shows an impossible speed. The real fix is to put the card under
  review rather than decline a single payment.
- **The cost model is assumptions, not measurements.** ₹1,500 chargeback fee,
  2% take rate, ₹40 goodwill. They are stated explicitly and swept in
  `cost_sensitivity` precisely so they can be argued with.
- **The GBDT beats the rules on AP (0.990 vs 0.851) and recall.** The rules are
  primary anyway, for explainability and because they need no labels — real
  fraud labels arrive weeks late via chargebacks. A production system would run
  both.
- **State partitioning is not solved.** Partitioning a Kafka topic by `card_id`
  keeps card windows correct but splits device- and IP-level state across
  workers, needing a second partitioning or a shared store.
- **Latency figures are processing time**, not end-to-end pipeline time.

## Mapping to production

| this repo | production |
|---|---|
| `replay()` over a sorted list | Kafka consumer, partitioned by `card_id` |
| `StreamingFeaturizer` dicts | Redis / RocksDB state store, same TTL sweep |
| `AlertRecord` → JSONL | alert topic + case-management queue |
| threshold from dev split | config value, re-tuned on a schedule |

The schema is anchored to the Razorpay Payments API payment entity — amounts as
integer paise, `pay_`/`acc_`/`card_` id prefixes, unix `created_at` — so
swapping the simulator for a real test-mode webhook feed is a parsing change,
not a redesign. `simulate --razorpay-shape` emits payment entities directly.

## Layout

```
razorpay_fraud/
  schema.py      Transaction / Episode / Dataset; Razorpay-shaped export
  simulator.py   labelled stream: 3 attacks + 6 hard negatives
  features.py    O(1)/payment causal sliding-window feature extraction
  detectors.py   rule + statistical detector, GBDT, isolation forest, strawman
  evaluate.py    per-payment, per-episode and rupee-cost metrics
  stream.py      streaming replay + alert audit trail
  viz.py         timeline / PR / cost / latency charts
  pipeline.py    the experimental protocol, end to end
  live.py        live console: engine thread + SSE server
  live_page.html the live console's page, served by live.py
  dashboard.py   exports one JSON bundle for the web dashboard
  cli.py         demo | live | simulate | replay | explain
tests/           117 tests, stdlib unittest (incl. CLI smoke tests)
out/             generated: RESULTS.md, report.json, alerts.jsonl, *.png
```

## Reproducing

Everything is seeded. `python -m razorpay_fraud demo --seed 7` reproduces every
number above. `--days`, `--cards` and `--rate-scale` vary the difficulty;
episode counts scale with the horizon so the fraud rate stays near 1% at any
setting.

## License

MIT — see [LICENSE](LICENSE).
