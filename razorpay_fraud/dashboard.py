"""Export a single JSON bundle for the web dashboard.

Everything the dashboard needs is derived here rather than in the page, so the
page stays a renderer and the numbers on it are the same ones in
``report.json`` -- there is no second implementation of any metric.

Two things are downsampled, both for size rather than for looks:

* the per-payment score scatter keeps every alert and every fraudulent payment,
  and thins the quiet legitimate majority. Keeping all 61k points would add a
  megabyte of JSON to say "these are all near zero".
* precision/recall curves are reduced to at most ``PR_POINTS`` points by
  striding, after the curve itself has been computed on the full data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from .evaluate import average_precision, pr_curve
from .schema import Episode, Transaction

if TYPE_CHECKING:  # pragma: no cover
    from .pipeline import PipelineResult

PR_POINTS = 260
#: Keep 1 in N of the unremarkable legitimate payments in the scatter.
SCATTER_STRIDE = 9


def _timeline_bins(transactions: Sequence[Transaction], t0: float, bin_s: float = 60.0):
    """Payments per bin, plus how many of them were fraudulent."""
    if not transactions:
        return []
    span = transactions[-1].created_at - t0
    n_bins = max(1, int(span // bin_s) + 1)
    total = [0] * n_bins
    fraud = [0] * n_bins
    for txn in transactions:
        idx = min(n_bins - 1, int((txn.created_at - t0) // bin_s))
        total[idx] += 1
        fraud[idx] += txn.is_fraud
    return [
        {"h": round((i + 0.5) * bin_s / 3600.0, 4), "n": total[i], "f": fraud[i]}
        for i in range(n_bins)
    ]


def _thin_pr(precisions, recalls):
    if len(precisions) <= PR_POINTS:
        pairs = list(zip(recalls, precisions))
    else:
        stride = len(precisions) / PR_POINTS
        idx = sorted({int(i * stride) for i in range(PR_POINTS)} | {len(precisions) - 1})
        pairs = [(recalls[i], precisions[i]) for i in idx]
    return [{"r": round(r, 5), "p": round(p, 5)} for r, p in pairs]


def build(result: "PipelineResult") -> dict:
    dataset = result.dataset
    test_txns = [row.txn for row in result.test_rows]
    t0 = dataset.split_ts
    labels = [t.is_fraud for t in test_txns]
    primary = result.results["rules"]

    detectors = []
    for name, detector_result in result.results.items():
        precisions, recalls, _ = pr_curve(detector_result.test_scores, labels)
        metrics = detector_result.test.metrics
        detectors.append(
            {
                "name": name,
                "threshold": round(detector_result.threshold, 6),
                "ap": round(detector_result.test.average_precision, 4),
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1": round(metrics.f1, 4),
                "alert_rate": round(metrics.alert_rate, 6),
                "tp": metrics.tp,
                "fp": metrics.fp,
                "fn": metrics.fn,
                "tn": metrics.tn,
                "dev_precision": round(detector_result.dev.metrics.precision, 4),
                "dev_recall": round(detector_result.dev.metrics.recall, 4),
                "cost_strict": round(detector_result.test.cost_strict.total, 2),
                "cost_contained": round(detector_result.test.cost_contained.total, 2),
                "savings_pct": round(detector_result.test.savings_contained_pct, 4),
                "pr": _thin_pr(precisions, recalls),
            }
        )

    # Score scatter: every alert and every fraud kept, quiet traffic thinned.
    scatter = []
    for i, (txn, score) in enumerate(zip(test_txns, primary.test_scores)):
        keep = score >= primary.threshold or txn.is_fraud or (i % SCATTER_STRIDE == 0)
        if keep:
            scatter.append(
                {
                    "h": round((txn.created_at - t0) / 3600.0, 4),
                    "s": round(score, 4),
                    "f": 1 if txn.is_fraud else 0,
                }
            )

    episodes = [
        {
            "id": episode.episode_id,
            "pattern": episode.pattern,
            "fraud": episode.is_fraud,
            "start_h": round((episode.start_ts - t0) / 3600.0, 4),
            "end_h": round((episode.end_ts - t0) / 3600.0, 4),
            "n": episode.n_payments,
        }
        for episode in dataset.episodes_in(test=True)
    ]

    # Alerts carry their full audit record: the rules that fired, the
    # human-readable detail, and the feature values at decision time. This is
    # the same content as alerts.jsonl -- the dashboard's alert explorer is a
    # view onto the audit trail, not a separate summary of it.
    from .detectors import RuleDetector
    from .stream import AUDIT_FEATURES

    rule_detector = RuleDetector()
    alerts = []
    for row, score in zip(result.test_rows, primary.test_scores):
        if score < primary.threshold:
            continue
        txn = row.txn
        alerts.append(
            {
                "payment_id": txn.payment_id,
                "h": round((txn.created_at - t0) / 3600.0, 5),
                "score": round(score, 4),
                "amount_inr": round(txn.amount_inr, 2),
                "card_id": txn.card_id,
                "device_id": txn.device_id,
                "merchant_id": txn.merchant_id,
                "is_fraud": txn.is_fraud,
                "pattern": txn.pattern,
                "episode_id": txn.episode_id,
                "reasons": [r.as_dict() for r in rule_detector.reasons(row.values)],
                "features": {
                    name: round(row.values[name], 4) for name in AUDIT_FEATURES
                },
            }
        )

    return {
        "meta": dataset.meta,
        "cost_model": result.cost_model.as_dict(),
        "detectors": detectors,
        "primary": "rules",
        "patterns": [p.as_dict() for p in primary.test.patterns],
        "sensitivity": result.sensitivity,
        "streaming": result.stream_stats,
        "do_nothing_inr": round(primary.test.do_nothing_inr, 2),
        "block_everything_inr": round(primary.test.block_everything_inr, 2),
        "timeline": _timeline_bins(test_txns, t0),
        "episodes": episodes,
        "scatter": scatter,
        "alerts": alerts,
        "n_test": len(test_txns),
        "n_test_fraud": sum(labels),
    }


def export(result: "PipelineResult", path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(result), separators=(",", ":")), encoding="utf-8")
    return str(path)
