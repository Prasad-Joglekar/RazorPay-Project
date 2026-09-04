"""End-to-end run: generate, extract, tune on dev, evaluate on held-out test.

The ordering here is the experimental protocol, and it is deliberately rigid:

1. Generate a labelled stream and cut it by **time** at ``split_ts``.
2. Extract features in one continuous streaming pass over the whole timeline.
   State crosses the split; predictions do not. Restarting the featurizer at
   the split would hand every entity an empty history and inflate the test
   numbers.
3. Fit the supervised model on dev rows only.
4. Choose each detector's threshold by minimising expected cost **on dev**.
5. Apply those frozen thresholds to test, and report.

Step 4 is the one that is easy to get wrong and easy to hide: picking the
threshold that looks best on test is the single most common way a fraud demo
reports numbers it cannot reproduce in production. Both the dev and the test
figures are written to the report so the gap between them is visible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import dashboard, viz
from .detectors import (
    IsolationForestDetector,
    MLDetector,
    NaiveCountDetector,
    RuleDetector,
)
from .evaluate import (
    CostModel,
    SplitReport,
    average_precision,
    build_report,
    choose_threshold,
    confusion_at,
    cost_curve,
    cost_sensitivity,
    episode_outcomes,
)
from .features import FeatureRow, StreamingFeaturizer
from .schema import Dataset, Transaction
from .simulator import Simulator, SimulatorConfig
from .stream import replay, write_alerts_jsonl


@dataclass(slots=True)
class DetectorResult:
    name: str
    threshold: float
    dev: SplitReport
    test: SplitReport
    dev_scores: list[float]
    test_scores: list[float]

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "threshold": round(self.threshold, 6),
            "dev": self.dev.as_dict(),
            "test": self.test.as_dict(),
        }


@dataclass(slots=True)
class PipelineResult:
    dataset: Dataset
    dev_rows: list[FeatureRow]
    test_rows: list[FeatureRow]
    results: dict[str, DetectorResult]
    cost_model: CostModel
    sensitivity: list[dict]
    stream_stats: dict
    artifacts: dict[str, str]


def _split_rows(rows: Sequence[FeatureRow], split_ts: float):
    dev = [r for r in rows if r.txn.created_at < split_ts]
    test = [r for r in rows if r.txn.created_at >= split_ts]
    return dev, test


def run(
    *,
    config: SimulatorConfig | None = None,
    cost_model: CostModel | None = None,
    out_dir: str | Path = "out",
    make_plots: bool = True,
    write_alerts: bool = True,
    verbose: bool = True,
) -> PipelineResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cost_model = cost_model or CostModel()

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # ---------------------------------------------------------------- 1. data
    dataset = Simulator(config or SimulatorConfig()).generate()
    log(
        f"[1/6] generated {dataset.meta['n_transactions']:,} payments  "
        f"({dataset.meta['n_fraud']:,} fraudulent = "
        f"{dataset.meta['fraud_rate']:.2%}), {dataset.meta['n_episodes']} labelled episodes"
    )

    # ------------------------------------------------------------ 2. features
    featurizer = StreamingFeaturizer()
    rows = [featurizer.process(t) for t in dataset.transactions]
    dev_rows, test_rows = _split_rows(rows, dataset.split_ts)
    dev_txns = [r.txn for r in dev_rows]
    test_txns = [r.txn for r in test_rows]
    dev_eps = dataset.episodes_in(test=False)
    test_eps = dataset.episodes_in(test=True)
    log(
        f"[2/6] streamed features: {len(dev_rows):,} dev / {len(test_rows):,} test, "
        f"state at end = {featurizer.state_size()}"
    )

    # ----------------------------------------------------------- 3. detectors
    detectors: list[tuple[str, object, list[float], list[float]]] = []

    rule_detector = RuleDetector()
    detectors.append(
        (
            "rules",
            rule_detector,
            [rule_detector.score(r.values) for r in dev_rows],
            [rule_detector.score(r.values) for r in test_rows],
        )
    )

    naive = NaiveCountDetector()
    detectors.append(
        (
            "naive_card_count",
            naive,
            [naive.score(r.values) for r in dev_rows],
            [naive.score(r.values) for r in test_rows],
        )
    )

    importances: dict[str, float] = {}
    if MLDetector.available():
        ml = MLDetector()
        ml.fit([r.values for r in dev_rows], [r.txn.is_fraud for r in dev_rows])
        detectors.append(
            (
                "gbdt",
                ml,
                ml.score_batch([r.values for r in dev_rows]),
                ml.score_batch([r.values for r in test_rows]),
            )
        )
        try:
            # Permutation importance costs one full re-scoring per feature per
            # repeat, so it runs on a stratified subsample: every fraudulent
            # row (there are only ~1% of them, and dropping any would make the
            # average-precision drop it measures noisy) plus an evenly spaced
            # sample of the legitimate ones.
            imp_rows = [r for r in test_rows if r.txn.is_fraud]
            legit = [r for r in test_rows if not r.txn.is_fraud]
            stride = max(1, len(legit) // 12_000)
            imp_rows.extend(legit[::stride])
            importances = ml.compute_importances(
                [r.values for r in imp_rows], [r.txn.is_fraud for r in imp_rows]
            )
            log(f"      permutation importance on {len(imp_rows):,} sampled test rows")
        except Exception as exc:  # pragma: no cover - diagnostics only
            log(f"      (permutation importance skipped: {exc})")

        iso = IsolationForestDetector().fit([r.values for r in dev_rows])
        detectors.append(
            (
                "isolation_forest",
                iso,
                iso.score_batch([r.values for r in dev_rows]),
                iso.score_batch([r.values for r in test_rows]),
            )
        )
        log(f"[3/6] trained gbdt + isolation forest on {len(dev_rows):,} dev rows")
    else:
        log("[3/6] scikit-learn not installed - reporting rule detector only")

    # ------------------------------------- 4. tune on dev, freeze, test on test
    results: dict[str, DetectorResult] = {}
    for name, detector, dev_scores, test_scores in detectors:
        choice = choose_threshold(
            dev_txns, dev_scores, dev_eps, cost_model, mode="contained"
        )
        dev_report = build_report(
            name, "dev", dev_txns, dev_scores, dev_eps, choice.threshold, cost_model
        )
        test_report = build_report(
            name, "test", test_txns, test_scores, test_eps, choice.threshold, cost_model
        )
        results[name] = DetectorResult(
            name=name,
            threshold=choice.threshold,
            dev=dev_report,
            test=test_report,
            dev_scores=dev_scores,
            test_scores=test_scores,
        )
        m = test_report.metrics
        log(
            f"      {name:<18} thr={choice.threshold:.4f}  "
            f"AP={test_report.average_precision:.3f}  P={m.precision:.3f}  "
            f"R={m.recall:.3f}  F1={m.f1:.3f}"
        )
    log("[4/6] thresholds chosen on dev by minimum expected cost, then frozen")

    # --------------------------------------------- 5. streaming replay + audit
    primary = results["rules"]
    stream_result = replay(
        test_txns, rule_detector, primary.threshold, featurizer=StreamingFeaturizer()
    )
    stream_stats = stream_result.summary()
    artifacts: dict[str, str] = {}
    if write_alerts:
        alerts_path = out / "alerts.jsonl"
        n = write_alerts_jsonl(stream_result.alerts, alerts_path)
        artifacts["alerts"] = str(alerts_path)
        log(f"[5/6] replayed test stream: {n:,} alerts written to {alerts_path}")
    else:
        log("[5/6] replayed test stream")

    sensitivity = cost_sensitivity(
        dev_txns, primary.dev_scores, dev_eps, mode="contained"
    )

    # --------------------------------------------------------- 6. write it out
    if make_plots:
        outcomes = episode_outcomes(
            test_txns, primary.test_scores, test_eps, primary.threshold
        )
        made = viz.plot_timeline(
            test_txns, primary.test_scores, test_eps, primary.threshold,
            out / "timeline.png",
        )
        if made:
            artifacts["timeline"] = made
        made = viz.plot_pr_curves(
            {
                name: (r.test_scores, [t.is_fraud for t in test_txns])
                for name, r in results.items()
            },
            out / "pr_curves.png",
            operating_points={
                name: (r.test.metrics.recall, r.test.metrics.precision)
                for name, r in results.items()
            },
        )
        if made:
            artifacts["pr_curves"] = made
        made = viz.plot_cost_curve(
            cost_curve(dev_txns, primary.dev_scores, dev_eps, cost_model, mode="contained"),
            cost_curve(test_txns, primary.test_scores, test_eps, cost_model, mode="contained"),
            primary.threshold,
            primary.dev.do_nothing_inr,
            primary.test.do_nothing_inr,
            out / "cost_curve.png",
        )
        if made:
            artifacts["cost_curve"] = made
        made = viz.plot_detection_latency(outcomes, out / "detection_latency.png")
        if made:
            artifacts["detection_latency"] = made

    report = {
        "dataset": dataset.meta,
        "cost_model": cost_model.as_dict(),
        "detectors": {name: r.as_dict() for name, r in results.items()},
        "streaming": stream_stats,
        "cost_sensitivity": sensitivity,
        "feature_importance": dict(
            sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        ),
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    artifacts["report_json"] = str(report_path)

    result_for_export = PipelineResult(
        dataset=dataset,
        dev_rows=dev_rows,
        test_rows=test_rows,
        results=results,
        cost_model=cost_model,
        sensitivity=sensitivity,
        stream_stats=stream_stats,
        artifacts=artifacts,
    )
    artifacts["dashboard_data"] = dashboard.export(
        result_for_export, out / "dashboard_data.json"
    )

    summary_path = out / "RESULTS.md"
    summary_path.write_text(
        render_markdown(dataset, results, stream_stats, sensitivity, importances, cost_model),
        encoding="utf-8",
    )
    artifacts["results_md"] = str(summary_path)
    log(f"[6/6] wrote {report_path} and {summary_path}")

    return result_for_export


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _fmt_inr(x: float) -> str:
    return f"Rs {x:,.0f}"


def render_markdown(
    dataset: Dataset,
    results: dict[str, DetectorResult],
    stream_stats: dict,
    sensitivity: list[dict],
    importances: dict[str, float],
    cost_model: CostModel,
) -> str:
    meta = dataset.meta
    primary = results["rules"]
    lines: list[str] = []
    add = lines.append

    add("# Results")
    add("")
    add(
        f"Generated from `{meta['n_transactions']:,}` simulated payments over "
        f"{meta['days']:.0f} days ({meta['n_merchants']} merchants, "
        f"{meta['n_cards']:,} cards), of which **{meta['n_fraud']:,} "
        f"({meta['fraud_rate']:.2%}) are fraudulent**, across "
        f"{meta['n_episodes']} labelled episodes."
    )
    add("")
    add(
        "Thresholds were chosen by minimising expected cost **on the dev split** "
        "and then applied unchanged to the held-out test split. Every number in "
        "the headline table is from the test split."
    )
    add("")

    # ---- headline table
    add("## Detector comparison (held-out test split)")
    add("")
    add("| detector | AP | precision | recall | F1 | alert rate | cost (contained) | vs. no detector |")
    add("|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        m = r.test.metrics
        add(
            f"| `{name}` | {r.test.average_precision:.3f} | {m.precision:.3f} | "
            f"{m.recall:.3f} | {m.f1:.3f} | {m.alert_rate:.2%} | "
            f"{_fmt_inr(r.test.cost_contained.total)} | "
            f"**{r.test.savings_contained_pct:+.1%}** |"
        )
    add("")
    add(
        f"Doing nothing costs {_fmt_inr(primary.test.do_nothing_inr)} on this split; "
        f"declining every payment costs {_fmt_inr(primary.test.block_everything_inr)}. "
        "Both bounds matter -- a detector that beats neither is not worth deploying."
    )
    add("")

    # ---- dev vs test
    add("## Dev vs. test (is the threshold overfit?)")
    add("")
    add("| detector | threshold | dev P | dev R | test P | test R | ΔP | ΔR |")
    add("|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        d, t = r.dev.metrics, r.test.metrics
        add(
            f"| `{name}` | {r.threshold:.4f} | {d.precision:.3f} | {d.recall:.3f} | "
            f"{t.precision:.3f} | {t.recall:.3f} | {t.precision - d.precision:+.3f} | "
            f"{t.recall - d.recall:+.3f} |"
        )
    add("")

    # ---- per pattern
    add("## Per-pattern breakdown (test split, rule detector)")
    add("")
    add(
        "Attack patterns are scored on detection; legitimate look-alike patterns "
        "are scored on how often they are wrongly flagged. Splitting them out is "
        "the whole point -- an aggregate precision figure hides which specific "
        "legitimate behaviour a detector cannot tell from fraud."
    )
    add("")
    add("| pattern | kind | payments | flagged | episodes | detected | median latency |")
    add("|---|---|---|---|---|---|---|")
    for p in primary.test.patterns:
        kind = "**attack**" if p.is_fraud else "legit"
        latency = f"{p.median_latency_s:.0f}s" if p.median_latency_s is not None else "-"
        episodes = f"{p.n_detected_episodes}/{p.n_episodes}" if p.n_episodes else "-"
        rate = f"{p.episode_rate:.0%}" if p.n_episodes else "-"
        add(
            f"| `{p.pattern}` | {kind} | {p.n_payments:,} | {p.payment_rate:.2%} | "
            f"{episodes} | {rate} | {latency} |"
        )
    add("")

    # ---- cost
    add("## Cost model")
    add("")
    add(
        f"- false positive: {cost_model.take_rate:.1%} of the payment (lost merchant "
        f"margin) + {_fmt_inr(cost_model.fp_goodwill_inr)} support/goodwill + "
        f"{_fmt_inr(cost_model.review_cost_inr)} review"
    )
    add(
        f"- false negative: the full payment amount + "
        f"{_fmt_inr(cost_model.chargeback_fee_inr)} chargeback fee"
    )
    add(f"- true positive: {_fmt_inr(cost_model.review_cost_inr)} of analyst time")
    add("")
    add(
        "`strict` charges every unflagged fraudulent payment as a full loss. "
        "`contained` credits the detector for intervention: once the first "
        "payment of an attack is flagged, the card or device is blocked and the "
        "rest of that attack is prevented. Production behaves like `contained`, "
        "which is why detection latency is a headline number."
    )
    add("")
    add("| detector | strict cost | strict saving | contained cost | contained saving |")
    add("|---|---|---|---|---|")
    for name, r in results.items():
        add(
            f"| `{name}` | {_fmt_inr(r.test.cost_strict.total)} | "
            f"{r.test.savings_strict_pct:+.1%} | "
            f"{_fmt_inr(r.test.cost_contained.total)} | "
            f"{r.test.savings_contained_pct:+.1%} |"
        )
    add("")

    add("### Sensitivity to the cost assumptions")
    add("")
    add(
        "The chosen threshold is only as good as the cost ratio behind it. "
        "Re-deriving it under alternative assumptions shows how much of the "
        "result is the detector and how much is the accounting."
    )
    add("")
    add("| assumption | threshold | precision | recall |")
    add("|---|---|---|---|")
    for row in sensitivity:
        add(
            f"| {row['variant']} | {row['chosen_threshold']:.4f} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} |"
        )
    add("")

    # ---- streaming
    add("## Streaming performance")
    add("")
    lat = stream_stats["processing_latency_us"]
    add(
        f"- {stream_stats['n_processed']:,} payments replayed in "
        f"{stream_stats['elapsed_s']:.2f}s = "
        f"**{stream_stats['throughput_per_s']:,.0f} payments/sec** single-threaded"
    )
    add(
        f"- per-payment processing latency: p50 {lat['p50']:.0f}us, "
        f"p95 {lat['p95']:.0f}us, p99 {lat['p99']:.0f}us"
    )
    add(
        f"- {stream_stats['n_alerts']:,} alerts "
        f"({stream_stats['alert_rate']:.2%} of traffic) with a full audit record each"
    )
    add("")
    add(
        "Latency is feature extraction plus scoring, not end-to-end pipeline "
        "time -- a real deployment adds broker and network hops on top."
    )
    add("")

    if importances:
        add("## Feature importance (gbdt, permutation on test)")
        add("")
        add("| feature | importance (drop in AP) |")
        add("|---|---|")
        for name, value in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:12]:
            add(f"| `{name}` | {value:.4f} |")
        add("")

    return "\n".join(lines) + "\n"
