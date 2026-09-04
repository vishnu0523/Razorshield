/**
 * Recovery workflows.
 *
 * Drawn as an escalation board rather than a table, so the distribution is
 * visible at a glance instead of having to be counted down a column.
 *
 * Lanes are ordered by how much the action costs the merchant, cheapest
 * first. The honest reading of the current data is that the queue is
 * back-weighted -- roughly three quarters of detected groups score high
 * enough that policy sends them to a human. That is a cost, not a win, and
 * the board should show it rather than imply a tidy funnel that thins out to
 * the right.
 */

import type { RecoveryBoundedAction, RecoveryWorkflowListResponse } from "../types/api";
import { count, money, useApi } from "../lib/format";
import { ApiUnreachable } from "./ApiUnreachable";

const LOSS_COPY: Record<string, string> = {
  refund_abuse: "Refund abuse",
  return_abuse: "Return abuse",
  payment_failure_abuse: "Payment-failure abuse",
  promo_or_account_abuse: "Promo/account abuse",
  coordinated_abuse: "Coordinated abuse",
};

type Lane = {
  action: RecoveryBoundedAction;
  title: string;
  blurb: string;
  color: string;
};

/** Left to right = increasing cost to the merchant of taking the action. */
const LANES: Lane[] = [
  {
    action: "MONITOR_ONLY",
    title: "Watch",
    blurb: "No contact, no friction. Just keep counting.",
    color: "var(--color-teal-deep)",
  },
  {
    action: "REQUEST_VERIFICATION",
    title: "Verify",
    blurb: "Ask the customer to confirm. Reversible.",
    color: "var(--color-butter)",
  },
  {
    action: "PREPARE_EVIDENCE",
    title: "Build evidence",
    blurb: "Assemble the chargeback pack in advance.",
    color: "var(--color-coral)",
  },
  {
    action: "MANUAL_REVIEW",
    title: "Send to a person",
    blurb: "Costs analyst time. Always needs approval.",
    color: "var(--color-critical)",
  },
];

export function Recovery() {
  // The full set, not the top slice. Cases are ranked by risk, so a small
  // limit returns nothing but CRITICAL ones and every lane except the last
  // renders empty -- which misrepresents the distribution rather than
  // showing it.
  const workflows = useApi<RecoveryWorkflowListResponse>(
    "/api/recovery/workflows?limit=100",
  );

  if (workflows.state === "loading")
    return <p className="num px-5 py-16 text-sm text-faint">Loading workflows...</p>;
  if (workflows.state === "error")
    return <ApiUnreachable message={workflows.message} />;

  const all = workflows.data.workflows;
  const grandTotal = all.reduce((sum, w) => sum + w.expected_protected_value, 0);

  return (
    <div className="space-y-5">
      <section className="panel p-6">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 className="eyebrow">Recommended next steps</h2>
          <span className="num text-xs text-faint">
            {count(workflows.data.total)} risk cases · {money(grandTotal)} at stake
          </span>
        </div>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">
          Every detected group lands in exactly one lane, and further right
          costs the merchant more to act on. Most land on the right — these are
          all high-risk groups, and the policy engine sends anything scoring 90
          or above to a person. That analyst time is a real cost, and it is
          counted against us in the ledger rather than left out of it.
        </p>
      </section>

      <div className="grid gap-4 lg:grid-cols-4">
        {LANES.map((lane) => {
          const items = all.filter((w) => w.bounded_action === lane.action);
          const laneValue = items.reduce(
            (sum, w) => sum + w.expected_protected_value,
            0,
          );
          const share = all.length ? (items.length / all.length) * 100 : 0;

          return (
            <section key={lane.action} className="panel flex flex-col p-4">
              <div className="flex items-center gap-2">
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ backgroundColor: lane.color }}
                  aria-hidden
                />
                <h3 className="eyebrow !text-paper">{lane.title}</h3>
              </div>

              <div className="num mt-3 text-3xl font-semibold text-paper">
                {items.length}
                <span className="ml-1 text-sm font-normal text-faint">
                  {items.length === 1 ? "case" : "cases"}
                </span>
              </div>
              <div className="num mt-0.5 text-sm text-signal">
                {money(laneValue)}
              </div>

              {/* Share of the queue, as a bar -- the funnel shape at a glance. */}
              <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-ink-700">
                <div
                  className="h-full rounded-full transition-[width] duration-500 ease-out"
                  style={{ width: `${share}%`, backgroundColor: lane.color }}
                />
              </div>

              <p className="mt-3 text-xs leading-relaxed text-faint">
                {lane.blurb}
              </p>

              <div className="thin-scroll mt-4 max-h-72 space-y-2 overflow-y-auto pr-1">
                {items.length === 0 && (
                  <p className="text-xs text-faint">No cases in this lane.</p>
                )}
                {items.map((w) => (
                  <article
                    key={w.workflow_id}
                    className="rounded border border-ink-700 bg-ink-800 p-3 transition-colors duration-150 hover:border-ink-600"
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="num text-xs text-paper">{w.ring_id}</span>
                      <span className="num text-xs text-signal">
                        {money(w.expected_protected_value)}
                      </span>
                    </div>
                    <div className="mt-1 text-xs text-muted">
                      {LOSS_COPY[w.loss_type] ?? w.loss_type}
                    </div>
                    <div
                      className={`num mt-2 text-[0.6875rem] ${
                        w.requires_merchant_approval ? "text-warn" : "text-faint"
                      }`}
                    >
                      {w.requires_merchant_approval
                        ? "needs approval"
                        : "safe to run"}
                    </div>
                  </article>
                ))}
              </div>
            </section>
          );
        })}
      </div>

      <section className="panel p-6">
        <h3 className="eyebrow">Where each lane stops</h3>
        <ul className="mt-3 space-y-2">
          {LANES.map((lane) => {
            const sample = all.find((w) => w.bounded_action === lane.action);
            if (!sample) return null;
            return (
              <li key={lane.action} className="flex gap-3 text-sm leading-relaxed">
                <span
                  className="mt-1.5 h-2 w-2 shrink-0 rounded-full"
                  style={{ backgroundColor: lane.color }}
                  aria-hidden
                />
                <span className="text-muted">
                  <span className="text-paper">{lane.title}:</span>{" "}
                  {sample.stopping_rule}
                </span>
              </li>
            );
          })}
        </ul>
        <p className="mt-4 text-xs leading-relaxed text-faint">
          {workflows.data.caveat}
        </p>
      </section>
    </div>
  );
}
