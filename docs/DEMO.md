# Demo script

Three minutes. The whole case rests on one contrast, so everything else is
support: **individually unremarkable orders, a coordinated group.**

Do not narrate the architecture. Show the discovery.

---

## Before you start

```bash
make reproduce          # ~3 min, regenerates every number from seed 42
make api                # terminal 1
make ui                 # terminal 2 → localhost:5173
```

Have both terminals visible if you can. A judge seeing `202 passed` scroll by
before the demo starts is worth a paragraph of claims.

Open on the **Live demo** tab, not the measurement tab. Lead with the story.

---

## 0:00 — Set up the problem (20s)

> "A merchant is losing money to refund and coupon abuse. Every fraud tool
> scores transactions one at a time. Watch what that misses."

Press **Start simulation**. Say nothing for the first two phases.

---

## 0:20 — Let the boring part be boring (40s)

Phases 1–3 run. Orders scroll past scoring 12, 24, 41, 59. Let the audience
read them.

> "Nothing here crosses the threshold. A row-level model looks at each of these
> and correctly decides to let it through. The highest score so far is 59 — that
> is MONITOR, not even enough to ask the customer to verify."

Point at the running "highest single score" line under the feed. **This is the
most important sentence in the demo.** Do not rush it.

---

## 1:00 — The reveal (30s)

Phase 5 lands. The graph appears.

> "Seventeen accounts. Five devices. Three delivery addresses. One connected
> group scoring ninety-nine out of a hundred, with two lakh seventy thousand
> rupees at stake."

Click one of the large green nodes. Everything else dims.

> "That device is shared by six of them. None of those six orders was suspicious
> on its own."

---

## 1:30 — Evidence and bounds (40s)

Phases 6–8.

> "The evidence is weighted by how much each signal actually moved the decision,
> not by what reads well. And the policy engine is deterministic — above five
> thousand rupees nothing happens without a person, no matter how confident the
> model is."

Then the honesty beat:

> "The AI writes the summary. It does not decide anything."

Switch to **Investigations**, open any case, press **Turn off AI**.

> "Model's gone. Score unchanged, evidence unchanged, recommended action
> unchanged. The explanation falls back to a deterministic summary and says so."

---

## 2:10 — The money (30s)

Switch to **Measurement**.

> "Prevented loss, minus what our own false alarms cost the merchant. Net two
> lakh eleven thousand."

Then hand it over. Drag **lost margin** to 60%.

> "These are our assumptions, and they're yours to change. Push this far enough
> and RazorShield costs money — it says so rather than clamping at zero."

Reset the slider.

---

## 2:40 — Close on the measurement (20s)

Scroll to the cluster-level panel.

> "Fourteen of fifteen rings caught, one of eleven legitimate lookalikes wrongly
> accused. Without the graph: nine of fifteen. The graph is worth five rings,
> measured against a floor, not asserted."

> "And the held-out split contains families sharing an address, an office
> network, a hostel, a reseller returning a third of what they buy. Catching
> rings is easy. Not accusing those is the job."

---

## Questions you should want

**"Is this real data?"**
No, and every number says so on screen. It is synthetic, generated from a seed,
and the metrics measure recovery of injected patterns under assumptions
documented in `ml/config.py`. That is why we built legitimate lookalike clusters
— so the numbers mean something harder than "can you find what you planted."

**"Your PR-AUC is only 0.44."**
Against a random floor of 0.087, and deliberately so. Row-level detection is
supposed to be mediocre here — that is the premise. The result that matters is
at cluster level.

**"How do I know these numbers are real?"**
`make reproduce`. Three minutes, from seed to every figure on screen. The API
serves `artifacts/metrics.json` verbatim and a test asserts it does not alter
anything in transit.

**"Where does the shipping address come from? Razorpay doesn't have that."**
Correct. RazorShield sits on merchant commerce data with payment signals joined
in. The adapter maps what Razorpay actually holds and leaves the rest `None`
rather than guessing.

**"What didn't work?"**
Fusion. We built the four-signal stacker the brief asked for and it scored 0.424
against 0.436 for the transaction model alone. The sub-scores correlate at
0.74–0.79 — they are not independent views. It is in the README with the
correlation matrix. Two graph features we invented also failed; one was 1.0 by
construction.

Answering this one well is worth more than any feature.

---

## Submission checklist

- [x] `make reproduce` (or `python scripts/run_all.py` on Windows) runs clean from a fresh clone
- [x] All test suites and phase gates pass (206 unit and contract tests)
- [x] `.env` is gitignored; no live Razorpay key anywhere in the tree
- [x] Embedded high-resolution screenshots in `docs/screenshots/`
- [x] README opens with problem framing and comparison against no-graph floor
- [x] Limitations and negative results (e.g. Risk Fusion) transparently documented

## What to put above the fold in the submission

1. Graph 14/15 vs no-graph floor 9/15, on identical held-out clusters
2. Zero of eleven legitimate lookalikes wrongly accused by the floor, one by the graph
3. `make reproduce` regenerates every claim from seed 42
4. Fusion is reported as a failure, with the reason

The fourth item is not modesty. A submission that names its own failures is the
one a judge believes about everything else.
