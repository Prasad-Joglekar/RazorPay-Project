---
title: Fraud Spike Console
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Detecting payment fraud spikes, with the false positives priced in
---

# Fraud Spike Console

Near-real-time detection of fraud spikes on a payment stream: card-testing
bursts, card-dump enumeration and geographically impossible card use.

Two views. **Overview** is the case in one page — results, the evaluation
design, and the decisions behind it. **Console** is the held-out stream: press
Replay to watch it run, and open any of the 332 alerts to see the rules that
fired and the exact feature values at the moment the decision was made.

The headline is not the F1 score. It is that the evaluation is built to be hard
to fool: the test data contains six *legitimate* traffic patterns that look
like attacks — flash sales, subscription runs, retry storms, office NAT, POS
terminals, real air travel — and each is scored separately. Precision median
0.950 across ten seeds, with every one of 195 attack episodes detected.

Payments are simulated. Source, full evaluation and a live streaming version
you can run locally: <https://github.com/Prasad-Joglekar/RazorPay-Project>
