/**
 * The reconciliation ledger.
 *
 * This is the hero, and it is deliberately not a row of KPI cards. Most fraud
 * dashboards show one big saved-money number. Showing the subtraction instead
 * -- what we prevented, minus what our false alarms cost the merchant -- is the
 * whole argument of the product, so it gets the whole top of the page.
 *
 * The assumptions are draggable rather than printed. A number a merchant cannot
 * audit is a number they should not trust, and the honest way to say "we think
 * 70% of detected exposure was preventable" is to let someone disagree and see
 * what happens. Dragging re-runs the same sum over the same held-out decisions;
 * nothing is scaled from a summary.
 *
 * Push the lost-margin ratio high enough and net protected value goes negative.
 * That is left reachable on purpose.
 */

import { useEffect, useRef, useState } from "react";
import type { FinancialImpact } from "../types/api";
import { API_BASE, money, percent } from "../lib/format";
import { useAnimatedNumber } from "../lib/animate";
import { More } from "./More";

type Knobs = {
  recovery_rate: number;
  cost_per_false_review: number;
  cost_per_false_block_ratio: number;
};

function Line({
  label,
  value,
  sign,
  tone,
}: {
  label: string;
  value: number;
  sign: "+" | "−";
  tone: "signal" | "alert";
}) {
  const shown = useAnimatedNumber(value, 260);
  return (
    <div className="flex items-baseline justify-between gap-6 py-2.5">
      <span className="text-sm text-muted">{label}</span>
      <span className={`num text-lg ${tone === "signal" ? "text-signal" : "text-alert"}`}>
        <span className="mr-1.5 text-faint">{sign}</span>
        {money(shown)}
      </span>
    </div>
  );
}

function Knob({
  label,
  value,
  min,
  max,
  step,
  format,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  hint: string;
  onChange: (v: number) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <label className="text-sm text-muted">{label}</label>
        <span
          className={`num text-sm transition-colors ${
            dragging ? "text-signal" : "text-paper"
          }`}
        >
          {format(value)}
        </span>
      </div>
      <div className="relative mt-3 h-[15px]">
        <div className="absolute inset-y-0 left-0 right-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-ink-700" />
        <div
          className="absolute inset-y-0 left-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-signal transition-[width] duration-150 ease-out"
          style={{ width: `${pct}%` }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          onPointerDown={() => setDragging(true)}
          onPointerUp={() => setDragging(false)}
          onBlur={() => setDragging(false)}
          className="range-input absolute inset-0 h-full w-full cursor-pointer"
          aria-label={label}
        />
      </div>
      <p className="mt-2 text-xs leading-relaxed text-faint">{hint}</p>
    </div>
  );
}

export function Ledger({ financial }: { financial: FinancialImpact }) {
  const defaults: Knobs = {
    recovery_rate: 0.7,
    cost_per_false_review: 120,
    cost_per_false_block_ratio: 0.18,
  };
  for (const a of financial.assumptions) {
    if (a.key in defaults) defaults[a.key as keyof Knobs] = a.value;
  }

  const [knobs, setKnobs] = useState<Knobs>(defaults);
  const [impact, setImpact] = useState<FinancialImpact>(financial);
  const [pending, setPending] = useState(false);
  const timer = useRef<number | null>(null);
  const inFlight = useRef<AbortController | null>(null);

  const adjusted =
    knobs.recovery_rate !== defaults.recovery_rate ||
    knobs.cost_per_false_review !== defaults.cost_per_false_review ||
    knobs.cost_per_false_block_ratio !== defaults.cost_per_false_block_ratio;

  // A short debounce so a fast drag doesn't fire a request per pixel, but
  // short enough (well under one animation frame's worth of thinking time)
  // that the ledger reads as tracking the slider rather than catching up to
  // it. Each request cancels the one before it, so a slow response from an
  // earlier drag position can never land after a newer one and visibly snap
  // the figure backward.
  useEffect(() => {
    if (!adjusted) {
      setImpact(financial);
      return;
    }
    if (timer.current) window.clearTimeout(timer.current);
    setPending(true);
    timer.current = window.setTimeout(async () => {
      inFlight.current?.abort();
      const controller = new AbortController();
      inFlight.current = controller;
      try {
        const res = await fetch(`${API_BASE}/api/financial/recompute`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(knobs),
          signal: controller.signal,
        });
        if (res.ok) setImpact(await res.json());
      } catch (err) {
        if ((err as Error).name !== "AbortError") throw err;
        return;
      } finally {
        if (inFlight.current === controller) setPending(false);
      }
    }, 60);
  }, [knobs, adjusted, financial]);

  const positive = impact.net_protected_value >= 0;
  const netShown = useAnimatedNumber(impact.net_protected_value, 260);
  const exposureShown = useAnimatedNumber(impact.exposure_detected, 260);

  return (
    <section className="panel p-6 sm:p-8">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="eyebrow">Money saved</h2>
        <span className="num text-xs text-faint">
          fraud value we caught {money(exposureShown)}
        </span>
      </div>

      <div className="grid gap-8 lg:grid-cols-[1fr_18rem]">
        <div className="mt-5 max-w-xl">
          <Line
            label="Prevented loss"
            value={impact.prevented_loss}
            sign="+"
            tone="signal"
          />
          <Line
            label="Cost of our false alarms"
            value={impact.false_positive_cost}
            sign="−"
            tone="alert"
          />

          <div className="mt-1 border-t border-ink-600" />

          <div className="settle flex flex-wrap items-baseline justify-between gap-4 pt-5">
            <span className="eyebrow !text-muted">Net protected</span>
            <span
              className={`num text-4xl font-semibold transition-opacity duration-150 sm:text-5xl ${
                positive ? "text-signal" : "text-critical"
              } ${pending ? "opacity-80" : "opacity-100"}`}
            >
              {money(netShown)}
            </span>
          </div>

          {!positive && (
            <p className="mt-4 text-sm leading-relaxed text-critical">
              Under these assumptions RazorShield costs the merchant more than it
              saves. That outcome is reachable from this panel on purpose.
            </p>
          )}

          {adjusted && positive && (
            <p className="num mt-4 text-xs text-warn">
              recomputed from your assumptions, not the defaults
            </p>
          )}
        </div>

        <div className="mt-6 space-y-6 lg:mt-5">
          <div className="flex items-baseline justify-between gap-3">
            <h3 className="eyebrow">Assumptions</h3>
            {adjusted && (
              <button
                onClick={() => setKnobs(defaults)}
                className="num text-xs text-faint underline-offset-2 hover:text-muted hover:underline"
              >
                reset
              </button>
            )}
          </div>

          <Knob
            label="Recovery rate"
            value={knobs.recovery_rate}
            min={0}
            max={1}
            step={0.01}
            format={percent}
            hint="How much of the fraud we caught could really have been stopped. Some of it would have failed on its own anyway."
            onChange={(v) => setKnobs({ ...knobs, recovery_rate: v })}
          />
          <Knob
            label="Cost per false review"
            value={knobs.cost_per_false_review}
            min={0}
            max={1000}
            step={10}
            format={money}
            hint="What it costs us every time we stop a real customer's order for a human to check."
            onChange={(v) => setKnobs({ ...knobs, cost_per_false_review: v })}
          />
          <Knob
            label="Lost margin on a false block"
            value={knobs.cost_per_false_block_ratio}
            min={0}
            max={1}
            step={0.01}
            format={percent}
            hint="The profit we lose when a real customer gives up and doesn't buy."
            onChange={(v) => setKnobs({ ...knobs, cost_per_false_block_ratio: v })}
          />
        </div>
      </div>

      <More label="How this recalculates">
        Every figure above is recomputed over the same held-out decisions the
        model actually made — nothing is scaled from a summary. Disagree with a
        default and the answer changes in front of you.
      </More>
    </section>
  );
}
