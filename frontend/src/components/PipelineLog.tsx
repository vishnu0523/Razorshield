/**
 * Pipeline log.
 *
 * The recorded stdout of the offline run that produced everything else on
 * screen -- generation, training, evaluation, graph build, ring detection,
 * and the test suite. Shown verbatim, after the fact.
 *
 * It is here because "these numbers came from a real run" is a claim, and the
 * cheapest way to support it is to show the run. Nothing is executed to
 * render this panel; it reads a text file the pipeline wrote.
 */

import { useState } from "react";
import type { PipelineLogResponse } from "../types/api";
import { shortDate, useApi } from "../lib/format";

/** Lines the pipeline emits that are worth spotting in a wall of output. */
function toneFor(line: string): string {
  if (/^\[\d+\/\d+\]/.test(line)) return "text-signal";
  if (/^\s*(VERDICT|Failed|FAILED|ERROR)/.test(line)) return "text-alert";
  if (/passed|Regenerated from seed/.test(line)) return "text-signal";
  if (/^={3,}|^-{3,}/.test(line)) return "text-faint";
  return "text-muted";
}

export function PipelineLog() {
  const log = useApi<PipelineLogResponse>("/api/pipeline/log");
  const [open, setOpen] = useState(false);

  if (log.state !== "ready" || !log.data.available) return null;

  const { lines, line_count, generated_at } = log.data;
  // Collapsed, the tail is the interesting part: the test summary and the
  // "regenerated from seed 42" line that closes a successful run.
  const shown = open ? lines : lines.slice(-14);

  return (
    <section className="panel p-6">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h3 className="eyebrow">Pipeline log</h3>
        <span className="num text-xs text-faint">
          {line_count} lines · {generated_at ? shortDate(generated_at) : "—"}
        </span>
      </div>

      <p className="mt-2 max-w-3xl text-sm leading-relaxed text-muted">
        Real output from the offline run that produced every number on this
        dashboard — data generation, training, evaluation, ring detection and
        the test suite. Recorded, not re-run: opening this page executes
        nothing.
      </p>

      <div className="thin-scroll mt-4 max-h-96 overflow-auto rounded border border-ink-700 bg-ink-800 p-4">
        <pre className="num text-[0.6875rem] leading-relaxed">
          {shown.map((line, i) => (
            <div key={i} className={toneFor(line)}>
              {line || " "}
            </div>
          ))}
        </pre>
      </div>

      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="num mt-3 text-xs text-faint underline-offset-2 transition-colors hover:text-signal hover:underline"
      >
        {open
          ? "show only the last few lines"
          : `show the full run — all ${line_count} lines`}
      </button>
    </section>
  );
}
