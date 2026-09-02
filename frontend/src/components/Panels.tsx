/**
 * Measurement panels.
 *
 * Deliberate choices here:
 *  - Transaction-level and ring-level results sit in separate panels because
 *    they measure different problems. Merging them would be misleading.
 *  - Accuracy is shown with the do-nothing comparison beside it, so a reader
 *    can see immediately why we refuse to headline it.
 *  - "Legitimate lookalikes flagged" gets the largest type on the ring panel.
 *    It is the number that distinguishes this from a size-threshold rule.
 */

import { useState } from "react";
import type {
  ClassifierMetrics,
  DatasetInfo,
  RingLevelMetrics,
} from "../types/api";
import { count, percent, ratio, shortDate } from "../lib/format";
import { More } from "./More";

function Stat({
  label,
  value,
  hint,
  emphasis = false,
}: {
  label: string;
  value: string;
  hint?: string;
  emphasis?: boolean;
}) {
  return (
    <div>
      <div className="eyebrow">{label}</div>
      <div
        className={`num mt-1.5 ${
          emphasis ? "text-3xl font-semibold text-paper" : "text-xl text-paper"
        }`}
      >
        {value}
      </div>
      {hint && <div className="mt-1 text-xs text-faint">{hint}</div>}
    </div>
  );
}

export function DetectionQuality({
  primary,
  baseline,
}: {
  primary: ClassifierMetrics;
  baseline: ClassifierMetrics;
}) {
  const doNothingAccuracy = 1 - primary.positive_rate;
  const lift = primary.pr_auc / primary.positive_rate;

  return (
    <section className="panel p-6">
      <h2 className="eyebrow">Transaction scoring</h2>

      <div className="mt-5 grid grid-cols-2 gap-5 sm:grid-cols-4">
        <Stat
          label="PR-AUC"
          value={ratio(primary.pr_auc)}
          hint={`${lift.toFixed(1)}× the random floor`}
          emphasis
        />
        <Stat label="Precision" value={ratio(primary.precision)} />
        <Stat label="Recall" value={ratio(primary.recall)} />
        <Stat label="F1" value={ratio(primary.f1)} />
      </div>

      <table className="mt-7 w-full text-sm">
        <thead>
          <tr className="border-b border-ink-700 text-left">
            <th className="eyebrow pb-2 font-normal">Model</th>
            <th className="eyebrow pb-2 text-right font-normal">PR-AUC</th>
            <th className="eyebrow pb-2 text-right font-normal">Precision</th>
            <th className="eyebrow pb-2 text-right font-normal">Recall</th>
          </tr>
        </thead>
        <tbody>
          {[
            { row: primary, note: "primary" },
            { row: baseline, note: "baseline" },
          ].map(({ row, note }) => (
            <tr key={row.model_name} className="border-b border-ink-800">
              <td className="py-2.5">
                <span className="text-paper">
                  {row.model_name.replaceAll("_", " ")}
                </span>
                <span className="ml-2 text-xs text-faint">{note}</span>
              </td>
              <td className="num py-2.5 text-right text-paper">
                {ratio(row.pr_auc)}
              </td>
              <td className="num py-2.5 text-right text-muted">
                {ratio(row.precision)}
              </td>
              <td className="num py-2.5 text-right text-muted">
                {ratio(row.recall)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <More label="Why we don't headline accuracy">
        Accuracy is <span className="num text-muted">{ratio(primary.accuracy ?? 0)}</span>,
        but a model that flags nothing scores{" "}
        <span className="num text-muted">{ratio(doNothingAccuracy)}</span>. By
        accuracy, doing nothing wins. That is why it is displayed and never
        headlined.
      </More>
    </section>
  );
}

export function RingDetection({
  ring,
  floor,
}: {
  ring: RingLevelMetrics;
  floor?: RingLevelMetrics | null;
}) {
  const clean = ring.n_hard_negatives_flagged === 0;
  const gained = floor
    ? ring.n_true_rings_detected - floor.n_true_rings_detected
    : null;

  return (
    <section className="panel p-6">
      <h2 className="eyebrow">Coordinated abuse · cluster level</h2>

      <div className="mt-5 flex flex-wrap items-end gap-x-10 gap-y-6">
        <div>
          <div className="eyebrow">Rings caught</div>
          <div className="num mt-1.5 text-3xl font-semibold text-paper">
            {ring.n_true_rings_detected}
            <span className="text-xl text-faint">/{ring.n_true_rings}</span>
          </div>
        </div>
        <div>
          <div className="eyebrow">Legitimate lookalikes flagged for review</div>
          <div
            className={`num mt-1.5 text-3xl font-semibold ${
              clean ? "text-signal" : "text-alert"
            }`}
          >
            {ring.n_hard_negatives_flagged}
            <span className="text-xl text-faint">
              /{ring.n_hard_negative_clusters}
            </span>
          </div>
        </div>
        <div className="flex gap-8">
          <Stat label="Precision" value={ratio(ring.precision)} />
          <Stat label="Recall" value={ratio(ring.recall)} />
        </div>
      </div>

      {floor && (
        <div className="mt-7">
          <div className="eyebrow">Measured against detection without the graph</div>
          <table className="mt-3 w-full text-sm">
            <tbody>
              {[
                { name: "With risk graph", row: ring, lead: true },
                { name: "Shared device only", row: floor, lead: false },
              ].map(({ name, row, lead }) => (
                <tr key={name} className="border-b border-ink-800">
                  <td className={`py-2.5 ${lead ? "text-paper" : "text-muted"}`}>
                    {name}
                  </td>
                  <td className="num py-2.5 text-right text-paper">
                    {row.n_true_rings_detected}
                    <span className="text-faint">/{row.n_true_rings}</span>
                    <span className="ml-2 text-xs text-faint">rings</span>
                  </td>
                  <td className="num py-2.5 text-right text-muted">
                    {ratio(row.precision)}
                    <span className="ml-2 text-xs text-faint">prec</span>
                  </td>
                  <td className="num py-2.5 text-right text-muted">
                    {ratio(row.recall)}
                    <span className="ml-2 text-xs text-faint">recall</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {gained !== null && (
            <p className="mt-3 text-sm text-muted">
              The graph is worth{" "}
              <span className="num text-signal">{gained} additional rings</span>{" "}
              caught, at the cost of{" "}
              <span className="num text-alert">
                {ring.n_hard_negatives_flagged - floor.n_hard_negatives_flagged}
              </span>{" "}
              extra legitimate lookalike flagged for review
              {ring.n_hard_negatives_flagged - floor.n_hard_negatives_flagged === 1
                ? ""
                : "s"}
              .
            </p>
          )}
        </div>
      )}

      <More label="What the lookalikes are">
        The held-out split contains {ring.n_hard_negative_clusters} legitimate
        clusters built to resemble abuse rings — a family sharing one address, an
        office network, a hostel, a high-return reseller, a flash-sale cohort.
        Catching rings is easy. Not accusing these is the hard part.
      </More>
    </section>
  );
}

export function ConfusionMatrix({ model }: { model: ClassifierMetrics }) {
  const { tp, fp, fn, tn } = model.confusion_matrix;
  const cells = [
    { label: "Caught", value: tp, tone: "text-signal", note: "true positive" },
    { label: "False alarm", value: fp, tone: "text-alert", note: "false positive" },
    { label: "Missed", value: fn, tone: "text-warn", note: "false negative" },
    { label: "Cleared", value: tn, tone: "text-muted", note: "true negative" },
  ];

  return (
    <section className="panel p-6">
      <h2 className="eyebrow">Outcomes · {count(model.support)} test transactions</h2>
      <div className="mt-5 grid grid-cols-2 gap-px overflow-hidden rounded border border-ink-700 bg-ink-700">
        {cells.map((c) => (
          <div key={c.label} className="bg-ink-800 p-4">
            <div className={`num text-2xl font-semibold ${c.tone}`}>
              {count(c.value)}
            </div>
            <div className="mt-1 text-sm text-paper">{c.label}</div>
            <div className="text-xs text-faint">{c.note}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

export function DatasetPanel({ dataset }: { dataset: DatasetInfo }) {
  const rows = [
    ["train", dataset.splits.train],
    ["validation", dataset.splits.validation],
    ["test · held out", dataset.splits.test],
  ] as const;

  return (
    <section className="panel p-6">
      <h2 className="eyebrow">Dataset · seed {dataset.seed}</h2>

      <div className="mt-5 grid grid-cols-2 gap-5 sm:grid-cols-4">
        <Stat label="Transactions" value={count(dataset.n_transactions)} />
        <Stat label="Customers" value={count(dataset.n_customers)} />
        <Stat label="Rings injected" value={count(dataset.n_rings_injected)} />
        <Stat
          label="Lookalike clusters"
          value={count(dataset.n_hard_negative_clusters)}
        />
      </div>

      <table className="mt-7 w-full text-sm">
        <thead>
          <tr className="border-b border-ink-700 text-left">
            <th className="eyebrow pb-2 font-normal">Split</th>
            <th className="eyebrow pb-2 text-right font-normal">Rows</th>
            <th className="eyebrow pb-2 text-right font-normal">Positive</th>
            <th className="eyebrow pb-2 text-right font-normal">Period ends</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, split]) => (
            <tr key={name} className="border-b border-ink-800">
              <td className="py-2.5 text-paper">{name}</td>
              <td className="num py-2.5 text-right text-muted">
                {count(split.n_rows)}
              </td>
              <td className="num py-2.5 text-right text-muted">
                {percent(split.positive_rate)}
              </td>
              <td className="num py-2.5 text-right text-faint">
                {shortDate(split.end)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <More label="How the split works">
        Split by time, not at random, so the model never trains on transactions
        that happened after the ones it scores. The threshold was chosen on
        validation; the test split was scored once.
      </More>
    </section>
  );
}

export function Caveats({ caveats }: { caveats: string[] }) {
  const [open, setOpen] = useState(false);

  return (
    <section className="panel p-6">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 text-left"
      >
        <div>
          <h2 className="eyebrow">How to read these numbers</h2>
          {!open && (
            <p className="mt-1.5 text-sm text-faint">
              {caveats.length} things worth knowing before you trust this page —
              synthetic data, point-in-time features, a threshold fixed before
              the test split was touched.
            </p>
          )}
        </div>
        <span
          className={`num shrink-0 text-xs text-faint transition-transform duration-300 ${
            open ? "rotate-180" : ""
          }`}
          aria-hidden
        >
          ▾
        </span>
      </button>

      <div
        className="grid transition-[grid-template-rows] duration-300 ease-out"
        style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
      >
        <div className="overflow-hidden">
          <ul className="mt-4 space-y-3">
            {caveats.map((c, i) => (
              <li
                key={i}
                className="flex gap-3 text-sm leading-relaxed text-muted"
              >
                <span className="num mt-0.5 shrink-0 text-xs text-faint">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
