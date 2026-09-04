"""Command line entry points.

    python -m razerpay_fraud demo        full pipeline: generate, tune, evaluate, plot
    python -m razerpay_fraud simulate    write a labelled payment stream to JSONL
    python -m razerpay_fraud replay      live streaming demo, alerts printed as they fire
    python -m razerpay_fraud live        serve the live console in a browser
    python -m razerpay_fraud sweep       rerun across seeds for an error bar
    python -m razerpay_fraud explain     read back the audit trail for sample alerts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .detectors import RuleDetector
from .evaluate import CostModel
from .features import StreamingFeaturizer
from .pipeline import run
from .simulator import Simulator, SimulatorConfig
from .stream import paced_replay


def _config_from_args(args) -> SimulatorConfig:
    kwargs = {"seed": args.seed, "days": args.days}
    if getattr(args, "cards", None):
        kwargs["n_cards"] = args.cards
    if getattr(args, "rate_scale", None):
        kwargs["rate_scale"] = args.rate_scale
    return SimulatorConfig(**kwargs)


# ---------------------------------------------------------------------------
def cmd_demo(args) -> int:
    result = run(
        config=_config_from_args(args),
        cost_model=CostModel(),
        out_dir=args.out,
        make_plots=not args.no_plots,
        verbose=True,
    )

    primary = result.results["rules"]
    metrics = primary.test.metrics
    print()
    print("=" * 74)
    print("HELD-OUT TEST SPLIT - rule detector (threshold chosen on dev)")
    print("=" * 74)
    print(
        f"  precision {metrics.precision:.3f}   recall {metrics.recall:.3f}   "
        f"F1 {metrics.f1:.3f}   AP {primary.test.average_precision:.3f}"
    )
    print(
        f"  {metrics.tp:,} caught / {metrics.fn:,} missed / {metrics.fp:,} false alarms "
        f"out of {metrics.n:,} payments"
    )
    print(
        f"  alert rate {metrics.alert_rate:.2%}  "
        f"(a reviewer sees {metrics.tp + metrics.fp:,} alerts, "
        f"{metrics.precision:.0%} of them real)"
    )
    print()
    print("  attacks caught, by pattern:")
    for pattern in primary.test.patterns:
        if not pattern.is_fraud:
            continue
        latency = (
            f"{pattern.median_latency_s:.0f}s"
            if pattern.median_latency_s is not None
            else "n/a"
        )
        print(
            f"    {pattern.pattern:<24} "
            f"{pattern.n_detected_episodes}/{pattern.n_episodes} episodes  "
            f"median detection {latency}"
        )
    print()
    print("  legitimate look-alikes wrongly flagged:")
    for pattern in primary.test.patterns:
        if pattern.is_fraud:
            continue
        print(
            f"    {pattern.pattern:<24} {pattern.payment_rate:>7.2%} of "
            f"{pattern.n_payments:,} payments"
        )
    print()
    print(
        f"  cost: Rs {primary.test.cost_contained.total:,.0f} vs "
        f"Rs {primary.test.do_nothing_inr:,.0f} with no detector "
        f"({primary.test.savings_contained_pct:+.1%})"
    )
    print()
    for name, path in sorted(result.artifacts.items()):
        print(f"  {name:<18} {path}")
    return 0


def cmd_simulate(args) -> int:
    dataset = Simulator(_config_from_args(args)).generate()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for txn in dataset.transactions:
            record = txn.to_razorpay() if args.razorpay_shape else {
                "payment_id": txn.payment_id,
                "created_at": txn.created_at,
                "amount": txn.amount,
                "currency": txn.currency,
                "method": txn.method,
                "status": txn.status,
                "merchant_id": txn.merchant_id,
                "card_id": txn.card_id,
                "device_id": txn.device_id,
                "ip": txn.ip,
                "city": txn.city,
                "lat": round(txn.lat, 5),
                "lon": round(txn.lon, 5),
                "is_fraud": txn.is_fraud,
                "pattern": txn.pattern,
                "episode_id": txn.episode_id,
            }
            fh.write(json.dumps(record, separators=(",", ":")))
            fh.write("\n")
    print(
        f"wrote {len(dataset.transactions):,} payments to {out} "
        f"({dataset.meta['n_fraud']:,} fraudulent, {dataset.meta['fraud_rate']:.2%})"
    )
    if args.razorpay_shape:
        print("  (Razorpay payment-entity shape; ground-truth labels omitted)")
    return 0


def cmd_replay(args) -> int:
    """Watch the detector work in wall-clock time. This is the demo shot."""
    dataset = Simulator(_config_from_args(args)).generate()
    detector = RuleDetector()

    warmup = [t for t in dataset.transactions if t.created_at < dataset.split_ts]
    live = [t for t in dataset.transactions if t.created_at >= dataset.split_ts]
    # Attacks are sparse by construction, so a short window starting at the
    # beginning of the test split may legitimately contain none. --skip-hours
    # moves the window without changing the warm-up: state is still built from
    # every payment before it, so detection is identical to a full replay.
    if args.skip_hours:
        warm_until = live[0].created_at + args.skip_hours * 3600.0
        skipped = [t for t in live if t.created_at < warm_until]
        live = [t for t in live if t.created_at >= warm_until]
        warmup = warmup + skipped
    if args.hours and live:
        cutoff = live[0].created_at + args.hours * 3600.0
        live = [t for t in live if t.created_at <= cutoff]
    if not live:
        print("no payments in that window - try a smaller --skip-hours")
        return 1

    print(f"warming state on {len(warmup):,} dev payments...", flush=True)
    featurizer = StreamingFeaturizer()
    for txn in warmup:
        featurizer.process(txn)

    print(
        f"replaying {len(live):,} payments at {args.speed:.0f}x "
        f"(threshold {args.threshold:.4f})",
        flush=True,
    )
    print(
        f"{'sim clock':>11}  {'payments':>9}  {'rate/min':>9}  "
        f"{'alerts':>7}  {'real':>6}",
        flush=True,
    )
    print("-" * 74, flush=True)

    def tick(stats: dict) -> None:
        hours, rem = divmod(int(stats["sim_elapsed_s"]), 3600)
        print(
            f"  {hours:02d}:{rem // 60:02d} elapsed  {stats['n_processed']:>9,}  "
            f"{stats['payments_per_min']:>9.1f}  {stats['n_alerts']:>7,}  "
            f"{stats['n_true_alerts']:>6,}",
            flush=True,
        )

    caught = 0
    total = 0
    for alert in paced_replay(
        live,
        detector,
        args.threshold,
        speed=args.speed,
        featurizer=featurizer,
        on_progress=tick,
        progress_every_s=args.tick * 60.0,
    ):
        total += 1
        caught += alert.is_fraud
        marker = "FRAUD" if alert.is_fraud else "false alarm"
        print(f"\n{alert.explain()}\n    -> ground truth: {marker}\n", flush=True)
    print("-" * 74)
    if total:
        print(f"{total} alerts, {caught} genuine ({caught / total:.0%} precision)")
    else:
        print("no alerts in this window")
    return 0


def cmd_sweep(args) -> int:
    """Repeat the experiment across seeds so the headline carries an error bar."""
    import json

    from .sweep import sweep

    default_seeds = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else default_seeds
    result = sweep(seeds, days=args.days)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


def cmd_live(args) -> int:
    """Serve the live console: the detector running, pushing alerts as they fire."""
    from .live import serve

    serve(
        host=args.host,
        port=args.port,
        config=_config_from_args(args),
        threshold=args.threshold,
        speed=args.speed,
    )
    return 0


def cmd_explain(args) -> int:
    """Print audit records, to show alerts are defensible after the fact."""
    path = Path(args.alerts)
    if not path.exists():
        print(f"no alert file at {path} - run 'demo' first", file=sys.stderr)
        return 1

    shown = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if args.pattern and record["label"]["pattern"] != args.pattern:
                continue
            if args.only_false_positives and record["label"]["is_fraud"]:
                continue
            print("=" * 74)
            print(
                f"{record['payment_id']}  score {record['score']:.3f} "
                f"(threshold {record['threshold']:.3f})"
            )
            print(
                f"  Rs {record['amount_inr']:,.2f}  card={record['card_id']}  "
                f"device={record['device_id']}  merchant={record['merchant_id']}"
            )
            print("  why:")
            for reason in record["reasons"]:
                print(f"    - {reason['rule']} ({reason['score']:.2f}): {reason['detail']}")
            print("  features at decision time:")
            for key, value in record["features"].items():
                print(f"      {key:<28} {value:>12.3f}")
            label = record["label"]
            verdict = "FRAUD" if label["is_fraud"] else "legitimate"
            print(f"  ground truth: {verdict}  pattern={label['pattern']}")
            shown += 1
            if shown >= args.limit:
                break
    if shown == 0:
        print("no matching alerts")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="razerpay_fraud",
        description="Near-real-time detection of fraud spikes in a payment stream.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--seed", type=int, default=7, help="simulator seed")
        p.add_argument("--days", type=float, default=3.0, help="days of traffic")
        p.add_argument("--cards", type=int, default=None, help="card population")
        p.add_argument(
            "--rate-scale", dest="rate_scale", type=float, default=None,
            help="legitimate traffic multiplier; raise it to lower the fraud rate",
        )

    p_demo = sub.add_parser("demo", help="run the full pipeline and write reports")
    add_common(p_demo)
    p_demo.add_argument("--out", default="out", help="output directory")
    p_demo.add_argument("--no-plots", action="store_true", help="skip chart rendering")
    p_demo.set_defaults(func=cmd_demo)

    p_sim = sub.add_parser("simulate", help="write a labelled payment stream")
    add_common(p_sim)
    p_sim.add_argument("--out", default="out/payments.jsonl")
    p_sim.add_argument(
        "--razorpay-shape", action="store_true",
        help="emit Razorpay payment entities instead of the internal schema",
    )
    p_sim.set_defaults(func=cmd_simulate)

    p_replay = sub.add_parser("replay", help="paced streaming demo with live alerts")
    add_common(p_replay)
    p_replay.add_argument("--speed", type=float, default=600.0, help="replay speed multiplier")
    p_replay.add_argument("--hours", type=float, default=6.0, help="hours of test traffic")
    p_replay.add_argument(
        "--threshold", type=float, default=0.0065,
        help="alert threshold; the default is the cost-optimal value the demo "
             "pipeline selects on the dev split",
    )
    p_replay.add_argument(
        "--skip-hours", dest="skip_hours", type=float, default=0.0,
        help="hours of test traffic to warm up on before showing the stream",
    )
    p_replay.add_argument(
        "--tick", type=float, default=10.0,
        help="minutes of simulated time between traffic status lines",
    )
    p_replay.set_defaults(func=cmd_replay)

    p_sweep = sub.add_parser(
        "sweep", help="rerun the experiment across seeds and report the spread"
    )
    add_common(p_sweep)
    p_sweep.add_argument(
        "--seeds", default=None,
        help="comma-separated seeds (default: 10 fixed seeds)",
    )
    p_sweep.add_argument("--out", default="out/seed_sweep.json")
    p_sweep.set_defaults(func=cmd_sweep)

    p_live = sub.add_parser("live", help="serve the live streaming console in a browser")
    add_common(p_live)
    # Defaults stay local. A hosting platform assigns the port (and requires
    # binding all interfaces), so both are read from the environment when set
    # -- deploying should not need a code change, and running locally should
    # not expose the console to the network by accident.
    p_live.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p_live.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8800)))
    p_live.add_argument(
        "--speed", type=float, default=300.0,
        help="stream speed multiplier; 300x means one stream-hour every 12 seconds",
    )
    p_live.add_argument(
        "--threshold", type=float, default=0.0065,
        help="alert threshold; the default is the cost-optimal value chosen on dev",
    )
    p_live.set_defaults(func=cmd_live)

    p_explain = sub.add_parser("explain", help="read back alert audit records")
    p_explain.add_argument("--alerts", default="out/alerts.jsonl")
    p_explain.add_argument("--pattern", default=None, help="filter by labelled pattern")
    p_explain.add_argument(
        "--only-false-positives", action="store_true",
        help="show only alerts that were wrong - the useful ones to read",
    )
    p_explain.add_argument("-n", "--limit", type=int, default=3)
    p_explain.set_defaults(func=cmd_explain)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
