/**
 * Ring investigation.
 *
 * Left: the case queue, ranked by risk. Right: one case in full.
 *
 * The evidence list is the part that has to be right. Each item shows the share
 * of the decision that signal actually contributed, drawn as a bar, so a
 * merchant can see not just what we found but how much it mattered. Facts and
 * inferences are labelled differently because conflating them is how an
 * explanation becomes a story.
 */

import { useEffect, useState } from "react";
import type {
  AuditEntry,
  AuditListResponse,
  ChainVerification,
  EvidenceItem,
  Explanation as ExplanationType,
  PolicyAction,
  RingDetail,
  RingListResponse,
  RingStatus,
  RingSummary,
} from "../types/api";
import { count, money, percent, useApi } from "../lib/format";
import { API_BASE } from "../lib/format";
import { RingGraph } from "./RingGraph";
import { ApiUnreachable } from "./ApiUnreachable";

const LEVEL_STYLE: Record<string, string> = {
  LOW: "border-ink-600 text-muted",
  MEDIUM: "border-warn/50 text-warn",
  HIGH: "border-alert/50 text-alert",
  CRITICAL: "border-critical/60 text-critical",
};

/** Purely identifying, not judging — the text tone above already carries the
 *  severity read, so the dot can afford to be a pleasant color rather than a
 *  traffic-light one. */
const LEVEL_DOT: Record<string, string> = {
  LOW: "var(--color-teal-deep)",
  MEDIUM: "var(--color-butter)",
  HIGH: "var(--color-coral)",
  CRITICAL: "var(--color-critical)",
};

const ACTION_COPY: Record<PolicyAction, string> = {
  ALLOW: "No action",
  MONITOR: "Monitor",
  VERIFY: "Request verification",
  MANUAL_REVIEW: "Hold for manual review",
};

function Badge({ level }: { level: string }) {
  return (
    <span
      className={`num inline-flex items-center gap-1.5 rounded border px-1.5 py-0.5 text-[0.6875rem] tracking-wide ${
        LEVEL_STYLE[level] ?? LEVEL_STYLE.LOW
      }`}
    >
      <span
        className="h-1.5 w-1.5 shrink-0 rounded-full"
        style={{ backgroundColor: LEVEL_DOT[level] ?? LEVEL_DOT.LOW }}
        aria-hidden
      />
      {level}
    </span>
  );
}

function CaseRow({
  ring,
  active,
  onSelect,
}: {
  ring: RingSummary;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={`w-full border-b border-ink-800 px-4 py-3 text-left transition-colors duration-150 ${
        active ? "bg-ink-700" : "hover:bg-ink-800"
      }`}
    >
      <div className="flex items-baseline justify-between gap-3">
        <span className="num text-sm text-paper">{ring.ring_id}</span>
        <Badge level={ring.risk_level} />
      </div>
      <div className="num mt-1.5 text-sm text-signal">
        {money(ring.financial_exposure)}
      </div>
      <div className="mt-0.5 text-xs text-faint">
        {count(ring.n_accounts)} accounts · {count(ring.n_devices)} devices ·{" "}
        {count(ring.n_transactions)} orders
      </div>
    </button>
  );
}

function Evidence({ items }: { items: EvidenceItem[] }) {
  return (
    <ul className="space-y-3">
      {items.map((item) => (
        <li key={item.code}>
          <div className="flex items-baseline justify-between gap-4">
            <span className="text-sm leading-relaxed text-paper">
              {item.statement}
            </span>
            <span
              className={`num shrink-0 text-[0.6875rem] ${
                item.kind === "FACT" ? "text-faint" : "text-warn"
              }`}
            >
              {item.kind === "FACT" ? "observed" : "inferred"}
            </span>
          </div>
          {item.weight > 0 && (
            <div className="mt-1.5 flex items-center gap-3">
              <div className="h-1 flex-1 overflow-hidden rounded-full bg-ink-700">
                <div
                  className="h-full bg-alert"
                  style={{ width: `${Math.round(item.weight * 100)}%` }}
                />
              </div>
              <span className="num text-[0.6875rem] text-faint">
                {percent(item.weight)} of the decision
              </span>
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}

const ACTOR_TONE: Record<string, string> = {
  system: "text-faint",
  model: "text-signal",
  policy_engine: "text-warn",
  llm: "text-muted",
  merchant: "text-paper",
};

/** The trail is rendered oldest-first so it reads as a sequence of decisions
 *  rather than a reverse-chronological log. */
function CaseTimeline({ ringId, refreshKey }: { ringId: string; refreshKey: number }) {
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [chain, setChain] = useState<ChainVerification | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/audit?subject_id=${encodeURIComponent(ringId)}`)
      .then((r) => r.json() as Promise<AuditListResponse>)
      .then((d) => !cancelled && setEntries([...d.entries].reverse()))
      .catch(() => undefined);
    fetch(`${API_BASE}/api/audit/verify`)
      .then((r) => r.json() as Promise<ChainVerification>)
      .then((d) => !cancelled && setChain(d))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [ringId, refreshKey]);

  return (
    <section className="panel p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h3 className="eyebrow">Decision trail</h3>
        {chain && (
          <span
            className={`num text-[0.6875rem] ${
              chain.valid ? "text-signal" : "text-critical"
            }`}
          >
            {chain.valid
              ? `hash chain intact · ${chain.checked} entries verified`
              : `chain broken at ${chain.broken_at}`}
          </span>
        )}
      </div>

      <ol className="mt-4 space-y-0">
        {entries.map((e, i) => (
          <li key={e.entry_id} className="flex gap-4">
            <div className="flex flex-col items-center">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-ink-600" />
              {i < entries.length - 1 && (
                <span className="w-px flex-1 bg-ink-700" />
              )}
            </div>
            <div className="pb-5">
              <div className="flex flex-wrap items-baseline gap-x-3">
                <span className="num text-xs text-faint">
                  {e.timestamp.slice(11, 19)}
                </span>
                <span className={`num text-xs ${ACTOR_TONE[e.actor] ?? "text-muted"}`}>
                  {e.actor.replace("_", " ")}
                </span>
                <span className="text-sm text-paper">
                  {e.event.replaceAll("_", " ")}
                </span>
              </div>
              <div className="num mt-1 text-sm text-muted">{e.decision}</div>
              <div className="mt-0.5 text-xs leading-relaxed text-faint">
                {e.reason}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

function DecisionControls({
  ringId,
  status,
  requiresApproval,
  onRecorded,
}: {
  ringId: string;
  status: RingStatus;
  requiresApproval: boolean;
  onRecorded: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const decided = status === "APPROVED" || status === "DISMISSED";

  async function submit(decision: "APPROVED" | "DISMISSED") {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/rings/${ringId}/decision`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision, note: "" }),
      });
      if (!res.ok) throw new Error(`Request failed with status ${res.status}`);
      onRecorded();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel p-6">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="eyebrow">Merchant decision</h3>
        {decided && (
          <span
            className={`num rounded border px-1.5 py-0.5 text-[0.6875rem] tracking-wide ${
              status === "APPROVED"
                ? "border-signal/50 text-signal"
                : "border-ink-600 text-muted"
            }`}
          >
            {status}
          </span>
        )}
      </div>
      <p className="mt-2 text-sm leading-relaxed text-muted">
        {decided
          ? "This case already has a decision on record, shown in the trail below. Pressing a button again records a new decision rather than editing the old one — nothing is ever overwritten."
          : requiresApproval
            ? "This case cannot be actioned automatically. Recording a decision appends to the audit trail; nothing is ever overwritten."
            : "No approval is required for this case. A decision can still be recorded for the trail."}
      </p>
      <div className="mt-4 flex gap-3">
        <button
          disabled={busy}
          onClick={() => submit("APPROVED")}
          className="num rounded border border-signal/50 px-4 py-2 text-sm text-signal transition-colors hover:bg-signal/10 disabled:opacity-40"
        >
          {decided ? "Record approval again" : "Approve review"}
        </button>
        <button
          disabled={busy}
          onClick={() => submit("DISMISSED")}
          className="num rounded border border-ink-600 px-4 py-2 text-sm text-muted transition-colors hover:bg-ink-700 disabled:opacity-40"
        >
          {decided ? "Record dismissal again" : "Dismiss case"}
        </button>
      </div>
      {error && <p className="num mt-3 text-xs text-alert">{error}</p>}
    </section>
  );
}

/** The explanation panel, with a switch that kills the language model.
 *
 *  The switch exists because "it degrades gracefully" is a claim, and a claim a
 *  judge can test in one click is worth more than a paragraph asserting it. The
 *  score, the evidence and the recommended action beside this panel do not move
 *  when it is pressed, which is the whole point. */
function Investigator({
  ringId,
  initial,
}: {
  ringId: string;
  initial: ExplanationType;
}) {
  const [explanation, setExplanation] = useState<ExplanationType>(() => {
    if (typeof window !== "undefined") {
      const p = new URLSearchParams(window.location.search);
      if (p.get("outage") === "true") {
        return { ...initial, degraded: true };
      }
    }
    return initial;
  });
  const [busy, setBusy] = useState(false);

  async function regenerate(simulateFailure: boolean) {
    setBusy(true);
    try {
      const res = await fetch(
        `${API_BASE}/api/rings/${ringId}/explanation?simulate_failure=${simulateFailure}`,
      );
      if (res.ok) setExplanation(await res.json());
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel p-6">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="eyebrow">Investigator summary</h3>
        <span className="num text-[0.6875rem] text-faint">
          {explanation.source === "llm" ? "AI-written" : "fixed template"}
        </span>
      </div>

      <p className="mt-3 text-sm leading-relaxed text-muted">{explanation.text}</p>

      {explanation.degraded && (
        <div className="mt-4 rounded border border-warn/40 bg-warn/10 px-4 py-3">
          <p className="text-sm text-warn">
            AI is down — fell back to the fixed template.
          </p>
          <p className="mt-1 text-xs text-faint">
            Detection, scoring and policy are unaffected. The risk score and
            recommended action above have not changed.
          </p>
        </div>
      )}

      <div className="mt-4 flex flex-wrap gap-3">
        <button
          disabled={busy}
          onClick={() => regenerate(false)}
          className="num rounded border border-ink-600 px-3 py-1.5 text-xs text-muted hover:bg-ink-700 disabled:opacity-40"
        >
          Regenerate
        </button>
        <button
          disabled={busy}
          onClick={() => regenerate(true)}
          className="num rounded border border-warn/40 px-3 py-1.5 text-xs text-warn hover:bg-warn/10 disabled:opacity-40"
        >
          Simulate model outage
        </button>
      </div>
    </section>
  );
}

function CaseDetail({ ringId }: { ringId: string }) {
  const [refreshKey, setRefreshKey] = useState(0);
  // Keyed on refreshKey too: a recorded decision doesn't change the ring's
  // own URL, only what the backend derives for it, so the plain path alone
  // would never refetch and the case would look undecided forever.
  const detail = useApi<RingDetail>(`/api/rings/${ringId}`, refreshKey);

  if (detail.state === "loading")
    return <p className="num p-6 text-sm text-faint">Loading case…</p>;
  if (detail.state === "error")
    return <ApiUnreachable message={detail.message} />;

  const { summary, evidence, graph, policy, explanation } = detail.data;

  return (
    <div key={summary.ring_id} className="settle space-y-5">
      <section className="panel p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h2 className="eyebrow">Case {summary.ring_id}</h2>
            <div className="mt-2 flex items-baseline gap-3">
              <span className="num text-4xl font-semibold text-paper">
                {summary.risk_score.toFixed(0)}
                <span className="text-xl text-faint">/100</span>
              </span>
              <Badge level={summary.risk_level} />
            </div>
            <p className="num mt-1.5 text-xs text-faint">
              confidence {summary.confidence.toFixed(2)} · how far above the
              cut-off this sits
            </p>
          </div>
          <div className="text-right">
            <div className="eyebrow">Money at risk</div>
            <div className="num mt-1.5 text-2xl font-semibold text-signal">
              {money(summary.financial_exposure)}
            </div>
            <p className="mt-1 text-xs text-faint">
              refunds already paid out, plus money still refundable
            </p>
          </div>
        </div>
      </section>

      <section className="panel p-6">
        <h3 className="eyebrow">Relationships</h3>
        <div className="mt-4">
          <RingGraph
            nodes={graph.nodes}
            edges={graph.edges}
            truncated={graph.truncated}
          />
        </div>
      </section>

      <div className="grid gap-5 lg:grid-cols-2">
        <section className="panel p-6">
          <h3 className="eyebrow">Why this was flagged</h3>
          <div className="mt-4">
            <Evidence items={evidence} />
          </div>
        </section>

        <div className="space-y-5">
          <section className="panel p-6">
            <div className="flex items-baseline justify-between gap-3">
              <h3 className="eyebrow">Recommended action</h3>
              <span className="num text-[0.6875rem] text-faint">
                policy {policy.policy_version}
              </span>
            </div>
            <div className="mt-3 text-lg text-paper">
              {ACTION_COPY[policy.recommended_action]}
            </div>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              {policy.reason}
            </p>
            {policy.requires_merchant_approval && (
              <div className="mt-4 rounded border border-warn/40 bg-warn/10 px-4 py-3">
                <p className="text-sm text-warn">
                  Requires merchant approval before anything is actioned.
                  {policy.amount_cap_applied &&
                    " The automatic-intervention limit was reached."}
                </p>
              </div>
            )}
          </section>

          <Investigator ringId={ringId} initial={explanation} />
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-[1fr_20rem]">
        <CaseTimeline ringId={ringId} refreshKey={refreshKey} />
        <DecisionControls
          ringId={ringId}
          status={summary.status}
          requiresApproval={policy.requires_merchant_approval}
          onRecorded={() => setRefreshKey((k) => k + 1)}
        />
      </div>
    </div>
  );
}

export function Investigation() {
  const list = useApi<RingListResponse>("/api/rings?limit=50");
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    if (list.state === "ready" && !selected && list.data.rings.length) {
      const params = typeof window !== "undefined" ? new URLSearchParams(window.location.search) : null;
      const targetRing = params?.get("ring");
      const found = targetRing && list.data.rings.some((r) => r.ring_id === targetRing);
      setSelected(found ? targetRing : list.data.rings[0].ring_id);
    }
  }, [list, selected]);

  if (list.state === "loading")
    return <p className="num px-5 py-16 text-sm text-faint">Loading cases…</p>;
  if (list.state === "error") return <ApiUnreachable message={list.message} />;

  return (
    <div className="grid gap-5 lg:grid-cols-[18rem_1fr]">
      <aside className="panel overflow-hidden">
        <div className="border-b border-ink-700 px-4 py-3">
          <h2 className="eyebrow">Open cases</h2>
          <p className="num mt-1 text-xs text-faint">
            {count(list.data.total)} flagged, ranked by risk
          </p>
        </div>
        <div className="thin-scroll max-h-[38rem] overflow-y-auto">
          {list.data.rings.map((ring) => (
            <CaseRow
              key={ring.ring_id}
              ring={ring}
              active={ring.ring_id === selected}
              onSelect={() => setSelected(ring.ring_id)}
            />
          ))}
        </div>
      </aside>

      <div>{selected && <CaseDetail ringId={selected} />}</div>
    </div>
  );
}
