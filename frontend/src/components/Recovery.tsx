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

const ACTION_COPY: Record<RecoveryBoundedAction, string> = {
  MONITOR_ONLY: "Monitor only",
  REQUEST_VERIFICATION: "Request verification",
  PREPARE_EVIDENCE: "Prepare evidence",
  MANUAL_REVIEW: "Manual review",
};

const ACTION_TONE: Record<RecoveryBoundedAction, string> = {
  MONITOR_ONLY: "text-muted",
  REQUEST_VERIFICATION: "text-warn",
  PREPARE_EVIDENCE: "text-signal",
  MANUAL_REVIEW: "text-alert",
};

export function Recovery() {
  const workflows = useApi<RecoveryWorkflowListResponse>("/api/recovery/workflows?limit=12");

  if (workflows.state === "loading")
    return <p className="num px-5 py-16 text-sm text-faint">Loading workflows...</p>;
  if (workflows.state === "error")
    return <ApiUnreachable message={workflows.message} />;

  return (
    <div className="space-y-5">
      <section className="panel p-6">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <h2 className="eyebrow">Bounded recovery workflows</h2>
            <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">
              Phase extension: detected risk becomes a merchant-controlled next
              step. No Razorpay production money action is executed from this demo.
            </p>
          </div>
          <span className="num text-xs text-faint">
            {count(workflows.data.total)} risk cases available
          </span>
        </div>
        <p className="mt-4 text-xs leading-relaxed text-faint">{workflows.data.caveat}</p>
      </section>

      <section className="panel overflow-hidden">
        <div className="grid grid-cols-[7.5rem_1fr_9rem] gap-4 border-b border-ink-700 px-4 py-3 text-left text-xs sm:grid-cols-[8rem_9rem_1fr_10rem_9rem]">
          <span className="eyebrow">Case</span>
          <span className="eyebrow hidden sm:block">Loss type</span>
          <span className="eyebrow">Next step</span>
          <span className="eyebrow text-right">Value</span>
          <span className="eyebrow hidden text-right sm:block">Control</span>
        </div>

        {workflows.data.workflows.map((workflow) => (
          <article
            key={workflow.workflow_id}
            className="grid grid-cols-[7.5rem_1fr_9rem] gap-4 border-b border-ink-800 px-4 py-4 text-sm transition-colors duration-150 last:border-b-0 hover:bg-ink-800 sm:grid-cols-[8rem_9rem_1fr_10rem_9rem]"
          >
            <div>
              <div className="num text-paper">{workflow.ring_id}</div>
              <div className="mt-1 text-xs text-faint sm:hidden">
                {LOSS_COPY[workflow.loss_type]}
              </div>
            </div>

            <div className="hidden text-muted sm:block">
              {LOSS_COPY[workflow.loss_type]}
            </div>

            <div>
              <div className={`text-paper ${ACTION_TONE[workflow.bounded_action]}`}>
                {ACTION_COPY[workflow.bounded_action]}
              </div>
              <p className="mt-1 leading-relaxed text-muted">
                {workflow.recommended_step}
              </p>
              <p className="mt-2 text-xs leading-relaxed text-faint">
                Stop rule: {workflow.stopping_rule}
              </p>
            </div>

            <div className="num text-right text-signal">
              {money(workflow.expected_protected_value)}
            </div>

            <div className="hidden text-right sm:block">
              <span
                className={`num text-xs ${
                  workflow.requires_merchant_approval ? "text-warn" : "text-faint"
                }`}
              >
                {workflow.requires_merchant_approval ? "approval" : "bounded"}
              </span>
            </div>
          </article>
        ))}
      </section>
    </div>
  );
}
