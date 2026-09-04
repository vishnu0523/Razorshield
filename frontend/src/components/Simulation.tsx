/**
 * Live demo.
 *
 * Replays a precomputed script built offline from real artifacts. Real
 * inference, real evidence, real policy, fixed sequence. A demo that can fail
 * live is a demo that will.
 *
 * The dramatic beat is phase 5, and it only lands because of phases 2 and 3:
 * the audience has to watch unremarkable orders go past first. The feed keeps
 * every earlier transaction on screen so the contrast between "nothing here
 * would have been stopped" and the component score is visible at once.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { AuditEntry, FeedItem, SimulationStep } from "../types/api";
import { API_BASE, money } from "../lib/format";
import { RingGraph } from "./RingGraph";
import { PipelineLog } from "./PipelineLog";

const STEP_MS = 2600;
const TOTAL = 9;

const LEVEL_TONE: Record<string, string> = {
  LOW: "text-faint",
  MEDIUM: "text-warn",
  HIGH: "text-alert",
  CRITICAL: "text-critical",
};

function FeedRow({ item }: { item: FeedItem }) {
  return (
    <div className="settle flex items-baseline justify-between gap-4 border-b border-ink-800 py-2">
      <span className="num text-sm text-paper">{money(item.amount)}</span>
      <span className="num text-xs text-faint">{item.product_category}</span>
      <span className={`num text-xs ${LEVEL_TONE[item.risk_level]}`}>
        {item.risk_score.toFixed(0)} {item.risk_level}
      </span>
    </div>
  );
}

export function Simulation() {
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState(0);
  const [steps, setSteps] = useState<SimulationStep[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const current = steps[steps.length - 1] ?? null;
  const feed: FeedItem[] = steps.flatMap((s) => s.feed_items);
  const trail: AuditEntry[] = steps.flatMap((s) => s.audit_entries);
  const ring = [...steps].reverse().find((s) => s.ring)?.ring ?? null;
  const protectedValue = steps.find((s) => s.financial_delta)?.financial_delta ?? null;

  const reset = useCallback(() => {
    if (timer.current) window.clearTimeout(timer.current);
    setRunning(false);
    setPhase(0);
    setSteps([]);
    setRunId(null);
  }, []);

  const start = useCallback(async () => {
    reset();
    const res = await fetch(`${API_BASE}/api/simulate/start`, { method: "POST" });
    const meta = await res.json();
    setRunId(meta.run_id);
    setRunning(true);
    setPhase(1);
  }, [reset]);

  useEffect(() => {
    if (!running || phase < 1 || phase > TOTAL) return;
    let cancelled = false;

    fetch(`${API_BASE}/api/simulate/step?phase=${phase}`)
      .then((r) => r.json() as Promise<SimulationStep>)
      .then((step) => {
        if (cancelled) return;
        setSteps((prev) => [...prev, step]);
        if (step.is_final) {
          setRunning(false);
          return;
        }
        timer.current = window.setTimeout(() => setPhase((p) => p + 1), STEP_MS);
      });

    return () => {
      cancelled = true;
    };
  }, [running, phase]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    const targetPhase = parseInt(params.get("phase") || "0", 10);
    if (targetPhase >= 1 && targetPhase <= TOTAL && phase === 0) {
      Promise.all(
        Array.from({ length: targetPhase }, (_, i) =>
          fetch(`${API_BASE}/api/simulate/step?phase=${i + 1}`).then((r) =>
            r.json() as Promise<SimulationStep>
          )
        )
      ).then((allSteps) => {
        setSteps(allSteps);
        setPhase(targetPhase);
      });
    }
  }, []);

  return (
    <div className="space-y-5">
      <section className="panel p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="eyebrow">Live simulation</h2>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted">
              A recorded run over real detections. Watch orders that no
              transaction model would stop, then watch the graph connect them.
            </p>
          </div>
          <div className="flex items-center gap-3">
            {steps.length > 0 && (
              <button
                onClick={reset}
                className="num rounded border border-ink-600 px-3 py-2 text-sm text-muted hover:bg-ink-700"
              >
                Reset
              </button>
            )}
            <button
              onClick={start}
              disabled={running}
              className="num rounded border border-signal/60 bg-signal/10 px-5 py-2 text-sm text-signal hover:bg-signal/20 disabled:opacity-40"
            >
              {running ? "Running…" : "Start simulation"}
            </button>
          </div>
        </div>

        {steps.length > 0 && (
          <>
            <div className="mt-5 flex gap-1">
              {Array.from({ length: TOTAL }, (_, i) => (
                <div
                  key={i}
                  className={`h-1 flex-1 rounded-full transition-colors ${
                    i < steps.length ? "bg-signal" : "bg-ink-700"
                  }`}
                />
              ))}
            </div>
            <div className="mt-4">
              <div className="num text-xs text-faint">
                phase {current?.phase} of {TOTAL} · {current?.phase_name}
                {runId && ` · ${runId}`}
              </div>
              <p className="settle mt-2 text-base leading-relaxed text-paper">
                {current?.narrative}
              </p>
            </div>
          </>
        )}
      </section>

      {protectedValue !== null && (
        <section className="panel p-6 sm:p-8">
          <h3 className="eyebrow">Net protected</h3>
          <div className="settle num mt-3 text-5xl font-semibold text-signal">
            {money(protectedValue)}
          </div>
        </section>
      )}

      {ring && (
        <section className="panel p-6">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h3 className="eyebrow">
              {ring.summary.ring_id} · {ring.summary.n_accounts} accounts ·{" "}
              {ring.summary.risk_score.toFixed(0)}/100
            </h3>
            <span className="num text-sm text-signal">
              {money(ring.summary.financial_exposure)} exposure
            </span>
          </div>
          <div className="mt-4">
            <RingGraph
              nodes={ring.graph.nodes}
              edges={ring.graph.edges}
              truncated={ring.graph.truncated}
            />
          </div>
        </section>
      )}

      {(feed.length > 0 || trail.length > 0) && (
        <div className="grid gap-5 lg:grid-cols-2">
          {feed.length > 0 && (
            <section className="panel p-6">
              <h3 className="eyebrow">Transaction feed</h3>
              <div className="mt-3 max-h-80 overflow-y-auto">
                {feed.map((item) => (
                  <FeedRow key={item.transaction_id} item={item} />
                ))}
              </div>
              <p className="mt-3 text-xs text-faint">
                Highest single score so far:{" "}
                <span className="num text-muted">
                  {Math.max(...feed.map((f) => f.risk_score)).toFixed(0)}
                </span>
                . Nothing here reaches the intervention threshold of 70.
              </p>
            </section>
          )}

          {trail.length > 0 && (
            <section className="panel p-6">
              <h3 className="eyebrow">Decision trail</h3>
              <ol className="mt-3 space-y-3">
                {trail.map((e) => (
                  <li key={e.entry_id} className="settle">
                    <div className="flex flex-wrap items-baseline gap-x-3">
                      <span className="num text-xs text-faint">
                        {e.timestamp.slice(11, 19)}
                      </span>
                      <span className="num text-xs text-muted">
                        {e.actor.replace("_", " ")}
                      </span>
                      <span className="text-sm text-paper">
                        {e.event.replaceAll("_", " ")}
                      </span>
                    </div>
                    <div className="num mt-0.5 text-sm text-muted">{e.decision}</div>
                  </li>
                ))}
              </ol>
            </section>
          )}
        </div>
      )}

      <PipelineLog />
    </div>
  );
}
