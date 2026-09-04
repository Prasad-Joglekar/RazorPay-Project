---
title: Fraud Spike Console
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Live streaming detector for payment fraud spikes
---

# Fraud Spike Console — live

A near-real-time detector for fraud spikes on a payment stream: card-testing
bursts, card-dump enumeration and impossible-travel card use.

The detector runs in a background thread and pushes each alert to your browser
over Server-Sent Events the moment its payment is scored. Live precision,
recall and throughput update as it goes, and every alert expands to the rules
that fired and the exact feature values at decision time.

**Payments are simulated.** Ground-truth labels travel with each alert so the
page can show live precision — a real deployment would not have them at
decision time.

The page takes a few seconds to warm up on start: the detector builds sliding
window state from earlier traffic before it scores anything, because a detector
deployed on Wednesday does not start with empty windows.

Source and full evaluation: see the project README.
