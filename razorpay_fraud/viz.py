"""Plots for the write-up and the demo video.

matplotlib is an optional dependency: if it is missing, every function here
returns ``None`` and the pipeline still produces its JSON and Markdown reports.
The numbers are the deliverable; the charts explain them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .evaluate import CostBreakdown, EpisodeOutcome, pr_curve
from .schema import Episode, Transaction

# A colour-blind-safe qualitative set (Okabe-Ito), so the charts survive both
# projector washout and the ~8% of reviewers with a colour vision deficiency.
PALETTE = {
    "rules": "#0072B2",
    "gbdt": "#D55E00",
    "isolation_forest": "#009E73",
    "naive_card_count": "#999999",
    "fraud": "#CC3311",
    "alert": "#D55E00",
    "ink": "#222222",
    "grid": "#DDDDDD",
}


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def plot_timeline(
    transactions: Sequence[Transaction],
    scores: Sequence[float],
    episodes: Sequence[Episode],
    threshold: float,
    path: str | Path,
    *,
    title: str = "Held-out test stream",
) -> str | None:
    """Traffic over time with attacks shaded and alerts marked.

    The chart that makes the case in a five-minute pitch: you can see the
    attacks land, see the alerts land on them, and see that the shaded
    legitimate spikes -- flash sales and subscription runs, which are *larger*
    than most attacks -- draw no alerts.
    """
    plt = _pyplot()
    if plt is None:
        return None

    t0 = min(t.created_at for t in transactions)
    hours = [(t.created_at - t0) / 3600.0 for t in transactions]

    # Payments per minute, binned.
    bin_h = 1.0 / 60.0
    n_bins = max(1, int(max(hours) / bin_h) + 1)
    counts = [0] * n_bins
    for h in hours:
        counts[min(n_bins - 1, int(h / bin_h))] += 1
    centres = [(i + 0.5) * bin_h for i in range(n_bins)]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax1.plot(centres, counts, color=PALETTE["ink"], linewidth=0.7, label="payments/min")

    # Shade labelled episodes. Attacks in red, legitimate look-alikes in grey:
    # the grey bands are the ones the detector is supposed to ignore.
    seen_fraud = seen_legit = False
    for ep in episodes:
        start_h = (ep.start_ts - t0) / 3600.0
        end_h = (ep.end_ts - t0) / 3600.0
        if end_h < 0 or start_h > max(hours):
            continue
        width = max(end_h - start_h, 0.02)
        if ep.is_fraud:
            ax1.axvspan(
                start_h, start_h + width, color=PALETTE["fraud"], alpha=0.22,
                label="attack" if not seen_fraud else None, zorder=0,
            )
            seen_fraud = True
        elif ep.pattern in ("flash_sale", "subscription_batch"):
            ax1.axvspan(
                start_h, start_h + width, color="#888888", alpha=0.18,
                label="legitimate spike" if not seen_legit else None, zorder=0,
            )
            seen_legit = True

    ax1.set_ylabel("payments / min")
    ax1.set_title(f"{title}: traffic, labelled episodes, and alerts")
    ax1.legend(loc="upper left", frameon=False, fontsize=9)
    _style(ax1)

    alert_h = [h for h, s in zip(hours, scores) if s >= threshold]
    alert_s = [s for s in scores if s >= threshold]
    quiet_h = [h for h, s in zip(hours, scores) if s < threshold]
    quiet_s = [s for s in scores if s < threshold]
    ax2.scatter(quiet_h, quiet_s, s=1.5, color="#BBBBBB", alpha=0.5, label="below threshold")
    ax2.scatter(alert_h, alert_s, s=7, color=PALETTE["alert"], alpha=0.85, label="alert")
    ax2.axhline(threshold, color=PALETTE["fraud"], linestyle="--", linewidth=1.2,
                label=f"threshold = {threshold:.3f}")
    ax2.set_ylabel("risk score")
    ax2.set_xlabel("hours into the test split")
    ax2.legend(loc="upper left", frameon=False, fontsize=9, ncol=3)
    _style(ax2)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_pr_curves(
    curves: dict[str, tuple[Sequence[float], Sequence[bool]]],
    path: str | Path,
    *,
    operating_points: dict[str, tuple[float, float]] | None = None,
    title: str = "Precision-recall, held-out test split",
) -> str | None:
    """PR curves for every detector, with the chosen operating point marked."""
    plt = _pyplot()
    if plt is None:
        return None

    from .evaluate import average_precision

    fig, ax = plt.subplots(figsize=(7.5, 6))
    baseline = 0.0
    for name, (scores, labels) in curves.items():
        precisions, recalls, _ = pr_curve(scores, labels)
        if not recalls:
            continue
        ax.step(
            recalls, precisions, where="post", linewidth=2,
            color=PALETTE.get(name, PALETTE["ink"]),
            label=f"{name} (AP = {average_precision(scores, labels):.3f})",
        )
        # Every curve is scored on the same split, so the prevalence line is
        # the same for all of them.
        baseline = sum(1 for y in labels if y) / len(labels) if labels else 0.0

    if baseline:
        ax.axhline(
            baseline, color="#999999", linestyle=":", linewidth=1.2,
            label=f"random baseline ({baseline:.4f})",
        )

    if operating_points:
        for name, (recall, precision) in operating_points.items():
            ax.plot(
                recall, precision, marker="o", markersize=9, markeredgewidth=2,
                markerfacecolor="white", color=PALETTE.get(name, PALETTE["ink"]), zorder=5,
            )

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{title}\n(circles = cost-optimal threshold chosen on dev)", fontsize=11)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_cost_curve(
    dev_curve: Sequence[CostBreakdown],
    test_curve: Sequence[CostBreakdown],
    chosen_threshold: float,
    do_nothing_dev: float,
    do_nothing_test: float,
    path: str | Path,
) -> str | None:
    """Expected rupee cost against threshold, on both splits.

    The point of the chart is the vertical line: the threshold is picked at the
    minimum of the dev curve, and lands where it lands on the test curve. If
    the test minimum sat somewhere else entirely, that would be visible here
    rather than buried.
    """
    plt = _pyplot()
    if plt is None:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, curve, baseline, label in (
        (ax1, dev_curve, do_nothing_dev, "dev (threshold chosen here)"),
        (ax2, test_curve, do_nothing_test, "held-out test"),
    ):
        xs = [c.threshold for c in curve]
        ys = [c.total for c in curve]
        ax.plot(xs, ys, color=PALETTE["rules"], linewidth=2, label="total cost")
        ax.plot(xs, [c.fp_cost for c in curve], color="#888888", linewidth=1.2,
                linestyle="--", label="false-positive cost")
        ax.plot(xs, [c.fn_cost for c in curve], color=PALETTE["fraud"], linewidth=1.2,
                linestyle="--", label="missed-fraud cost")
        ax.axhline(baseline, color="#444444", linestyle=":", linewidth=1.2,
                   label=f"no detector (Rs {baseline:,.0f})")
        ax.axvline(chosen_threshold, color=PALETTE["alert"], linewidth=1.6,
                   label=f"chosen threshold {chosen_threshold:.4f}")
        # symlog rather than log so threshold 0 ("alert on everything") stays on
        # the axis, but clamped at 0 -- a negative threshold is meaningless and
        # the default symlog range spends half the width drawing it.
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_xlim(0.0, max(xs) * 1.05 if xs else 1.0)
        ax.set_xlabel("threshold")
        ax.set_ylabel("expected cost (INR)")
        ax.set_title(label)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        _style(ax)

    fig.suptitle("Cost-based threshold selection (contained mode)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_detection_latency(
    outcomes: Sequence[EpisodeOutcome], path: str | Path
) -> str | None:
    """How long each attack ran before the first alert, by pattern."""
    plt = _pyplot()
    if plt is None:
        return None

    by_pattern: dict[str, list[float]] = {}
    for outcome in outcomes:
        if outcome.is_fraud and outcome.detected and outcome.latency_s is not None:
            by_pattern.setdefault(outcome.pattern, []).append(outcome.latency_s)
    if not by_pattern:
        return None

    patterns = sorted(by_pattern)
    data = [by_pattern[p] for p in patterns]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Tick labels are set separately rather than via boxplot's own keyword,
    # which was renamed from `labels` to `tick_labels` in matplotlib 3.9.
    parts = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True)
    ax.set_yticks(range(1, len(patterns) + 1))
    ax.set_yticklabels(patterns)
    for box in parts["boxes"]:
        box.set_facecolor("#CFE3F2")
        box.set_edgecolor(PALETTE["rules"])
    for median in parts["medians"]:
        median.set_color(PALETTE["fraud"])
        median.set_linewidth(2)

    for i, values in enumerate(data, start=1):
        ax.scatter(values, [i] * len(values), s=14, color=PALETTE["rules"], alpha=0.55, zorder=3)

    ax.set_xlabel("seconds from first payment of the attack to first alert")
    ax.set_title("Detection latency by attack pattern (test split)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)
