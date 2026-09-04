"""Repeat the whole experiment across seeds and report the spread.

A single held-out split is one sample. Where the attack episodes happen to fall
relative to the dev/test cut moves precision and recall by several points, and
quoting whichever seed looked best is the oldest trick in the book. This runs
the same protocol -- generate, stream features, tune the threshold on dev, score
the frozen threshold on test -- across several seeds and reports median and
range, so the headline number carries its own error bar.

Only the rule detector is swept: it needs no training, so a seed costs a few
seconds rather than a few minutes, and it is the primary model.
"""

from __future__ import annotations

from dataclasses import dataclass

from .detectors import RuleDetector
from .evaluate import (
    CostModel,
    average_precision,
    build_report,
    choose_threshold,
)
from .features import StreamingFeaturizer
from .simulator import Simulator, SimulatorConfig


@dataclass(slots=True)
class SeedResult:
    seed: int
    n_test: int
    n_test_fraud: int
    n_fraud_episodes: int
    threshold: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    alert_rate: float
    episode_detection: float
    savings_pct: float
    worst_hard_negative: float

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "n_test": self.n_test,
            "n_test_fraud": self.n_test_fraud,
            "n_fraud_episodes": self.n_fraud_episodes,
            "threshold": round(self.threshold, 6),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "average_precision": round(self.average_precision, 4),
            "alert_rate": round(self.alert_rate, 5),
            "episode_detection": round(self.episode_detection, 4),
            "savings_pct": round(self.savings_pct, 4),
            "worst_hard_negative_flag_rate": round(self.worst_hard_negative, 5),
        }


def run_seed(seed: int, *, days: float = 3.0, cost_model: CostModel | None = None) -> SeedResult:
    cost_model = cost_model or CostModel()
    dataset = Simulator(SimulatorConfig(seed=seed, days=days)).generate()

    featurizer = StreamingFeaturizer()
    rows = [featurizer.process(t) for t in dataset.transactions]
    dev = [r for r in rows if r.txn.created_at < dataset.split_ts]
    test = [r for r in rows if r.txn.created_at >= dataset.split_ts]
    dev_txns = [r.txn for r in dev]
    test_txns = [r.txn for r in test]
    dev_eps = dataset.episodes_in(test=False)
    test_eps = dataset.episodes_in(test=True)

    detector = RuleDetector()
    dev_scores = [detector.score(r.values) for r in dev]
    test_scores = [detector.score(r.values) for r in test]

    choice = choose_threshold(dev_txns, dev_scores, dev_eps, cost_model, mode="contained")
    report = build_report(
        "rules", "test", test_txns, test_scores, test_eps, choice.threshold, cost_model
    )
    metrics = report.metrics

    attacks = [p for p in report.patterns if p.is_fraud]
    n_eps = sum(p.n_episodes for p in attacks)
    n_det = sum(p.n_detected_episodes for p in attacks)
    hard_negatives = [
        p.payment_rate for p in report.patterns
        if not p.is_fraud and p.pattern != "baseline_legit"
    ]

    return SeedResult(
        seed=seed,
        n_test=len(test_txns),
        n_test_fraud=sum(1 for t in test_txns if t.is_fraud),
        n_fraud_episodes=n_eps,
        threshold=choice.threshold,
        precision=metrics.precision,
        recall=metrics.recall,
        f1=metrics.f1,
        average_precision=report.average_precision,
        alert_rate=metrics.alert_rate,
        episode_detection=n_det / n_eps if n_eps else 0.0,
        savings_pct=report.savings_contained_pct,
        worst_hard_negative=max(hard_negatives) if hard_negatives else 0.0,
    )


def _stats(values: list[float]) -> dict:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / n,
    }


def sweep(seeds: list[int], *, days: float = 3.0, verbose: bool = True) -> dict:
    results: list[SeedResult] = []
    if verbose:
        print(
            f"{'seed':>5}  {'test pmts':>10}  {'fraud':>6}  {'eps':>4}  "
            f"{'thr':>7}  {'prec':>6}  {'rec':>6}  {'F1':>6}  {'AP':>6}  {'eps det':>8}"
        )
        print("-" * 82)
    for seed in seeds:
        result = run_seed(seed, days=days)
        results.append(result)
        if verbose:
            print(
                f"{result.seed:>5}  {result.n_test:>10,}  {result.n_test_fraud:>6,}  "
                f"{result.n_fraud_episodes:>4}  {result.threshold:>7.4f}  "
                f"{result.precision:>6.3f}  {result.recall:>6.3f}  {result.f1:>6.3f}  "
                f"{result.average_precision:>6.3f}  {result.episode_detection:>7.0%}",
                flush=True,
            )

    summary = {
        field: _stats([getattr(r, field) for r in results])
        for field in (
            "precision", "recall", "f1", "average_precision",
            "alert_rate", "episode_detection", "savings_pct", "worst_hard_negative",
        )
    }
    if verbose:
        print("-" * 82)
        for field in ("precision", "recall", "f1", "average_precision", "episode_detection"):
            st = summary[field]
            print(
                f"  {field:<18} median {st['median']:.3f}   "
                f"range {st['min']:.3f} - {st['max']:.3f}"
            )
        worst = summary["worst_hard_negative"]
        print(
            f"  {'worst hard neg':<18} median {worst['median']:.2%}   "
            f"range {worst['min']:.2%} - {worst['max']:.2%}"
        )
    return {"seeds": [r.as_dict() for r in results], "summary": summary}
