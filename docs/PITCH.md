# 5-minute pitch script

For the Razorpay AI Buildathon submission video. Timestamps are cumulative —
read at a normal pace and it lands at 5:00. Record with `make reproduce` →
`make api` → `make ui` already running so the Live demo tab works live; if
anything stutters on the day, the screenshots in `docs/screenshots/` are the
fallback.

Bracketed lines are screen actions, not spoken words.

---

## 0:00 — Why now (25s)

> "Indian BFSI is getting hit by AI-enabled fraud, and returns and
> chargebacks are quietly eating merchant margin on top of that. Most fraud
> tools score one transaction at a time. We built RazorShield to show why
> that's not enough — and to prove it with numbers, not a demo trick."

[Open on the **Live demo** tab, not the measurement tab. Don't press start yet.]

## 0:25 — What we built (30s)

> "RazorShield is a risk graph for Razorpay merchants. It scores individual
> transactions like everyone else does — but it also builds a relationship
> map across devices, addresses, IPs and coupon codes, and finds the
> coordinated groups hiding inside data that looks fine row by row. Every
> number I'm about to show you is measured on a held-out test split the
> model never saw during training."

## 0:55 — Set up the problem (20s)

> "Watch what a row-level model misses."

[Press **Start simulation**. Say nothing for the first two phases.]

## 1:15 — Let the boring part be boring (30s)

Phases 1–3 run. Orders scroll past scoring 12, 24, 41, 59.

> "Nothing here crosses the threshold. Every one of these orders gets
> correctly allowed through. Watch the highest score reached, bottom left."

[Point at "Highest single score so far" under the feed.]

> "That's 59 at most. MONITOR. Not even enough to ask the customer to
> verify."

## 1:45 — The reveal (30s)

Phase 5 lands, the graph appears.

> "Seventeen accounts. Five devices. Three delivery addresses. One connected
> group scoring ninety-nine out of a hundred, two lakh seventy thousand
> rupees at stake — and every account in it individually looked fine."

[Click the large hub node. Everything else dims.]

> "That's the entity eight of these accounts share. None of those eight
> orders was suspicious alone."

## 2:15 — Evidence and bounds (35s)

Phases 6–8.

> "Each reason is ranked by how much it actually moved the score — not by
> what sounds convincing. And the rules are fixed: above five thousand
> rupees, nothing happens without a human, no matter how sure the model is."

[Switch to **Investigations**, open any case, press **Simulate model outage**.]

> "The AI only writes the summary — it never decides. Watch: AI's gone. Score
> unchanged, reasons unchanged, recommended action unchanged. It falls back
> to a fixed template and says so on screen."

## 2:50 — The money, honestly (40s)

[Switch to **Measurement**.]

> "This is the whole pitch in one panel. Prevented loss, minus what our own
> false alarms cost the merchant. Net protected: two lakh eleven thousand
> rupees. These three numbers on the right are assumptions, not facts — drag
> one."

[Drag **lost margin** toward 60%.]

> "Push it far enough and RazorShield costs the merchant money. It says so in
> red instead of hiding it. That's not a bug — it's the point. A number you
> can't disagree with isn't a number you should trust."

[Reset the slider.]

## 3:30 — The metric that matters (30s)

> "Fourteen of fifteen real fraud rings caught. Only one of eleven completely
> innocent groups wrongly flagged — and those innocent groups are a family
> sharing an address, an office network, a hostel. Without the graph, a
> simple same-device rule catches nine of fifteen. The graph is worth five
> more rings, measured against that floor, not just claimed."

## 4:00 — What didn't work (30s)

> "We also have to tell you what failed. The brief asked for a fusion model
> combining four risk signals. We built it, measured it, and it scored worse
> than the transaction model alone — 0.424 against 0.436 PR-AUC — because the
> four signals turned out to be correlated at 0.74 to 0.79. Not independent
> views, four measurements of the same thing. We shipped the honest number,
> not the one that looks better."

## 4:30 — Defense-only, by design (20s)

> "RazorShield never moves money and never blocks a payment on its own.
> Razorpay keys are test-mode only, enforced at import — a live key crashes
> the process before it can touch anything. Everything above five thousand
> rupees needs a person. This is a detector and an explainer, not an actor."

## 4:50 — Close (10s)

> "Individually boring transactions. A coordinated group hiding in plain
> sight. Numbers you can re-run yourself with one command: `make reproduce`.
> That's RazorShield."

[End on the Measurement tab or the repo README.]

---

# What broke, and how we got out

Three real bugs, kept rather than quietly fixed and forgotten, because a
submission that hides its failures is less credible on everything else it
claims.

**1. A silent detector kill switch, found by a floor value that was too big
by two orders of magnitude.**
The fraud-spike detector uses a modified z-score (median + MAD) to flag
abnormal windows, corroborated against the share of high-risk orders in that
window. The MAD floor — the minimum "noise" assumed in a metric — was a
single constant, `0.5`. That's reasonable for a count running in single
digits. It's enormous for a *proportion* between 0 and 1. It silently capped
every rate-based z-score at 1.35, which meant `failed_payment_rate` and
`high_risk_rate` could never register as anomalous — and since corroboration
requires both volume *and* risk-rate to spike together, the rule was
quietly suppressing every single alert. The detector looked like it was
working (it ran, it returned data) while doing nothing. We caught it by
checking the corroborated-alert count against the volume-only count and
finding it was zero when it should have been dozens. Fix: separate floors
for counts and for rates, with a regression test asserting the corroborated
count is nonzero on the known dataset.

**2. The audit trail's tamper-evident hash chain broke itself, on its own
first run.**
The audit log hashes each entry against the one before it, so any edit to
history is detectable. The very first time we ran the chain verifier, it
reported itself broken — which for a security feature is a heart-stopping
result. The cause: SQLite doesn't preserve timezone info. A UTC-aware
timestamp went in, came back out naive (no timezone), and re-hashed to a
different value than what was originally signed. Nothing was actually
tampered with — our own storage layer was the "attacker." Fix: normalize
every timestamp to UTC explicitly on both the write path and the verify
path, plus a regression test that writes, reads, and re-verifies in the same
run.

**3. A graph feature that was mathematically guaranteed to be useless, and
almost shipped anyway.**
We hypothesized that the *ratio of entities shared across a group* would
separate real rings from coincidental lookalikes. It scored a perfect 1.0
separating power in testing — which should have been exciting, and was
instead a red flag. A component is *defined* by its shared entities, so
every account in it touches at least one by construction — the feature was
tautological, not predictive. We removed it rather than leave it in to pad
the feature count, and kept the one graph feature that actually held up
(`bridging_entity_ratio`, 0.449 separating power) in the README with the
honest number next to the ones that didn't work.

**The pattern underneath all three:** a bug that fails loudly is annoying; a
bug that silently returns plausible-looking output is the dangerous one.
Two of these three shipped no error, no crash, no stack trace — just quietly
wrong behavior that looked correct until we went looking for the specific
number that should have been nonzero and wasn't. That's now the standing
rule for anything new we add: don't just test that it runs, test that the
number it produces is the number it should be.
