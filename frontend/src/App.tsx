import { useLayoutEffect, useRef, useState } from "react";
import type { HealthResponse, MetricsResponse } from "./types/api";
import { useApi, shortDate } from "./lib/format";
import { Investigation } from "./components/Investigation";
import { Ledger } from "./components/Ledger";
import { Recovery } from "./components/Recovery";
import { Simulation } from "./components/Simulation";
import {
  Caveats,
  ConfusionMatrix,
  DatasetPanel,
  DetectionQuality,
  RingDetection,
} from "./components/Panels";
import { ApiUnreachable } from "./components/ApiUnreachable";

function StatusBar({ health }: { health: HealthResponse | null }) {
  const artifacts = health?.artifacts_loaded ?? false;
  return (
    <header className="border-b border-[#e8dcc8] bg-[#11100d]">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-5 py-4 sm:px-8">
        <div className="flex items-baseline gap-3">
          <span className="text-lg font-semibold tracking-tight text-[#fffaf0]">
            RazorShield
          </span>
          <span className="hidden text-sm text-[#d8cbb7] sm:inline">
            Merchant Risk Command Center
          </span>
        </div>
        <div className="flex items-center gap-5 text-[15px] font-normal">
          <span className="flex items-center gap-2">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                artifacts ? "bg-signal" : "bg-warn"
              }`}
              aria-hidden
            />
            <span className="text-[#fffaf0]">
              {artifacts ? "evaluated" : "not evaluated"}
            </span>
          </span>
          <span className="text-[#d8cbb7]">
            explanations:{" "}
            {health?.llm_available ? "AI enabled" : "fixed template"}
          </span>
        </div>
      </div>
    </header>
  );
}

/** Shown whenever the API is serving an untrained state. Its job is to make it
 *  impossible to screenshot a zero and mistake it for a result. */
function PlaceholderBanner() {
  return (
    <div className="border border-warn/40 bg-warn/10 px-5 py-4">
      <p className="text-sm text-warn">
        No evaluation has run. Every figure below reads zero because nothing has
        been measured yet — these are not results.
      </p>
      <p className="num mt-2 text-xs text-faint">
        run: make generate && make train && make evaluate
      </p>
    </div>
  );
}

function EvaluationBanner() {
  return (
    <div className="border border-ink-700 bg-ink-850 px-5 py-4">
      <p className="text-sm text-muted">
        Made-up test data, scored honestly. These are benchmark results, not
        real Razorpay performance.
      </p>
    </div>
  );
}

function Loading() {
  return (
    <p className="num px-5 py-16 text-sm text-faint sm:px-8">
      Loading measurements…
    </p>
  );
}

type Tab = "measurement" | "details" | "cases" | "recovery" | "demo";

// Measurement is deliberately two pages, not one. The first answers "did this
// save money and did it catch the rings" -- the two questions a reader
// actually arrives with. Everything that supports those answers (model
// scores, confusion matrix, dataset, caveats) lives on the second, for the
// reader who wants to check the working rather than be handed all of it at
// once.
const TAB_ITEMS: [Tab, string][] = [
  ["measurement", "Measurement"],
  ["details", "Model details"],
  ["cases", "Investigations"],
  ["recovery", "Recovery"],
  ["demo", "Live demo"],
];

/** A pill that slides between tabs rather than a static underline, so
 *  switching sections reads as motion instead of a re-render. */
function Tabs({ tab, onChange }: { tab: Tab; onChange: (t: Tab) => void }) {
  const buttonRefs = useRef<Partial<Record<Tab, HTMLButtonElement | null>>>({});
  const [indicator, setIndicator] = useState<{ left: number; width: number } | null>(
    null,
  );

  useLayoutEffect(() => {
    const measure = () => {
      const el = buttonRefs.current[tab];
      if (el) setIndicator({ left: el.offsetLeft, width: el.offsetWidth });
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [tab]);

  return (
    <nav className="relative flex gap-1 border-b border-ink-700">
      {TAB_ITEMS.map(([key, label]) => (
        <button
          key={key}
          ref={(el) => {
            buttonRefs.current[key] = el;
          }}
          onClick={() => onChange(key)}
          className={`-mb-px px-4 py-2.5 text-sm transition-colors ${
            tab === key ? "text-paper" : "text-faint hover:text-muted"
          }`}
        >
          {label}
        </button>
      ))}
      {indicator && (
        <span
          aria-hidden
          className="absolute bottom-0 h-0.5 rounded-full bg-signal transition-[left,width] duration-200 ease-out"
          style={{ left: indicator.left, width: indicator.width }}
        />
      )}
    </nav>
  );
}

const TAB_ORDER: Tab[] = TAB_ITEMS.map(([key]) => key);

export default function App() {
  const health = useApi<HealthResponse>("/api/health");
  const metrics = useApi<MetricsResponse>("/api/metrics");
  const [tab, setTab] = useState<Tab>(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const qTab = params.get("tab") as Tab;
      if (qTab && TAB_ORDER.includes(qTab)) return qTab;
      const hash = window.location.hash.replace("#", "") as Tab;
      if (hash && TAB_ORDER.includes(hash)) return hash;
    }
    return "measurement";
  });
  const [dir, setDir] = useState(1);

  function changeTab(next: Tab) {
    if (next === tab) return;
    setDir(TAB_ORDER.indexOf(next) > TAB_ORDER.indexOf(tab) ? 1 : -1);
    setTab(next);
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      url.searchParams.set("tab", next);
      window.history.replaceState(null, "", url.toString());
    }
  }

  const pageStyle = { ["--dir" as string]: dir };

  return (
    <div className="min-h-screen">
      <StatusBar health={health.state === "ready" ? health.data : null} />

      <div className="mx-auto max-w-6xl px-5 sm:px-8">
        <Tabs tab={tab} onChange={changeTab} />
      </div>

      {tab === "demo" && (
        <main
          key="demo"
          style={pageStyle}
          className="page-transition mx-auto max-w-6xl px-5 py-6 sm:px-8 sm:py-8"
        >
          <Simulation />
        </main>
      )}

      {tab === "cases" && (
        <main
          key="cases"
          style={pageStyle}
          className="page-transition mx-auto max-w-6xl px-5 py-6 sm:px-8 sm:py-8"
        >
          <Investigation />
        </main>
      )}

      {tab === "recovery" && (
        <main
          key="recovery"
          style={pageStyle}
          className="page-transition mx-auto max-w-6xl px-5 py-6 sm:px-8 sm:py-8"
        >
          <Recovery />
        </main>
      )}

      {metrics.state === "loading" && tab === "measurement" && <Loading />}
      {metrics.state === "error" && tab === "measurement" && (
        <ApiUnreachable message={metrics.message} />
      )}

      {metrics.state === "ready" && tab === "measurement" && (
        <main
          key="measurement"
          style={pageStyle}
          className="page-transition mx-auto max-w-6xl space-y-5 px-5 py-6 sm:px-8 sm:py-8"
        >
          <div className="reveal" style={{ ["--i" as string]: 0 }}>
            {metrics.data.is_placeholder ? <PlaceholderBanner /> : <EvaluationBanner />}
          </div>

          <div className="reveal" style={{ ["--i" as string]: 1 }}>
            <Ledger financial={metrics.data.financial} />
          </div>

          <div className="reveal" style={{ ["--i" as string]: 2 }}>
            <RingDetection
              ring={metrics.data.ring_model}
              floor={metrics.data.ring_baseline_model}
            />
          </div>

          <footer className="num pt-2 pb-8 text-xs text-faint">
            Evaluated {shortDate(metrics.data.generated_at)} · the model scores,
            dataset and caveats behind these numbers are on{" "}
            <button
              onClick={() => changeTab("details")}
              className="text-signal underline underline-offset-2 hover:text-paper"
            >
              Model details
            </button>
          </footer>
        </main>
      )}

      {metrics.state === "loading" && tab === "details" && <Loading />}
      {metrics.state === "error" && tab === "details" && (
        <ApiUnreachable message={metrics.message} />
      )}

      {metrics.state === "ready" && tab === "details" && (
        <main
          key="details"
          style={pageStyle}
          className="page-transition mx-auto max-w-6xl space-y-5 px-5 py-6 sm:px-8 sm:py-8"
        >
          <div className="reveal" style={{ ["--i" as string]: 0 }}>
            <DetectionQuality
              primary={metrics.data.transaction_model}
              baseline={metrics.data.baseline_model}
            />
          </div>

          <div className="reveal grid gap-5 lg:grid-cols-2" style={{ ["--i" as string]: 1 }}>
            <ConfusionMatrix model={metrics.data.transaction_model} />
            <DatasetPanel dataset={metrics.data.dataset} />
          </div>

          <div className="reveal" style={{ ["--i" as string]: 2 }}>
            <Caveats caveats={metrics.data.caveats} />
          </div>

          <footer className="num pt-2 pb-8 text-xs text-faint">
            Evaluated {shortDate(metrics.data.generated_at)} · every figure on
            this page is regenerated by `make reproduce`
          </footer>
        </main>
      )}
    </div>
  );
}
