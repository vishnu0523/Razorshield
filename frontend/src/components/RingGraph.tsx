/**
 * The abuse ring graph.
 *
 * Hand-rolled d3-force simulation rendered to SVG rather than a graph library,
 * because the visual language matters here: node shape carries entity type,
 * fill carries risk, and shared entities are drawn larger than private ones.
 * Off-the-shelf components draw uniform circles and lose all of that.
 *
 * The reading the merchant should get in two seconds: a few hot entities in the
 * middle with many accounts hanging off them. That shape *is* the finding.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type Simulation,
} from "d3-force";
import type { GraphEdge, GraphNode, NodeType } from "../types/api";

type Positioned = GraphNode & { x: number; y: number };
type SimLink = { source: Positioned; target: Positioned; weight: number };

const WIDTH = 640;
const HEIGHT = 420;

const TYPE_LABEL: Record<NodeType, string> = {
  customer: "Account",
  device: "Device",
  ip: "IP address",
  address: "Delivery address",
  coupon: "Coupon code",
  order: "Order",
  payment: "Payment",
  refund: "Refund",
  return: "Return",
};

/** Entity nodes shared by more than one account are the evidence, so they are
 *  drawn largest. Accounts are uniform: no account is more important than
 *  another, only riskier. */
function radius(node: GraphNode): number {
  if (node.type === "customer") return 7;
  const shared = Number(node.attributes?.shared_by ?? 1);
  return 6 + Math.min(shared, 12) * 0.9;
}

/** One hue per entity type — shape already carries this, color reinforces it
 *  rather than replacing the risk read. Saturation still does the evidentiary
 *  work: a private entity (shared_by 1) fades toward gray, a hub entity comes
 *  in at full color, so "more color = more shared" survives the type tint. */
const TYPE_RGB: Partial<Record<NodeType, string>> = {
  device: "255,130,67", // coral
  address: "255,192,203", // blossom
  ip: "252,232,131", // butter
  coupon: "6,148,148", // teal
};

function fill(node: GraphNode): string {
  if (node.type === "customer") {
    if (node.risk_score >= 70) return "var(--color-critical)";
    if (node.risk_score >= 40) return "var(--color-warn)";
    return "var(--color-ink-600)";
  }
  const shared = Number(node.attributes?.shared_by ?? 1);
  const rgb = TYPE_RGB[node.type] ?? "205,191,174";
  const alpha = shared >= 3 ? 1 : shared === 2 ? 0.62 : 0.32;
  return `rgba(${rgb},${alpha})`;
}

/** Shape encodes entity type so the graph is readable without a legend lookup
 *  on every glance: accounts are circles, everything shared is angular. */
function Shape({ node, r }: { node: Positioned; r: number }) {
  const common = { fill: fill(node), stroke: "var(--color-ink-900)", strokeWidth: 1.5 };
  if (node.type === "customer") return <circle r={r} {...common} />;
  if (node.type === "device")
    return <rect x={-r} y={-r} width={r * 2} height={r * 2} rx={3} {...common} />;
  if (node.type === "address")
    return <rect x={-r} y={-r} width={r * 2} height={r * 2} transform="rotate(45)" {...common} />;
  if (node.type === "coupon")
    return <rect x={-r} y={-r * 0.7} width={r * 2} height={r * 1.4} rx={2} {...common} />;
  return (
    <polygon
      points={Array.from({ length: 6 }, (_, i) => {
        const a = (Math.PI / 3) * i - Math.PI / 6;
        return `${r * Math.cos(a)},${r * Math.sin(a)}`;
      }).join(" ")}
      {...common}
    />
  );
}

/** A small filled swatch rather than a plain colored glyph — the palette runs
 *  pale enough (butter, blossom) that raw colored text would wash out against
 *  the cream panel. A bordered chip stays legible at any hue. */
function LegendItem({
  color,
  label,
  square,
  diamond,
}: {
  color: string;
  label: string;
  square?: boolean;
  diamond?: boolean;
}) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className={`inline-block h-2.5 w-2.5 shrink-0 border border-ink-900/15 ${
          diamond ? "rotate-45" : square ? "rounded-[2px]" : "rounded-full"
        }`}
        style={{ backgroundColor: color }}
        aria-hidden
      />
      {label}
    </span>
  );
}

/** The strongest single link in the case — the entity shared by the most
 *  accounts. Selected by default so the graph opens already showing its own
 *  punchline instead of a hairball a viewer has to click around to decode. */
function findHub(nodes: GraphNode[]): GraphNode | null {
  let best: GraphNode | null = null;
  let bestShared = 1;
  for (const n of nodes) {
    if (n.type === "customer") continue;
    const shared = Number(n.attributes?.shared_by ?? 1);
    if (shared > bestShared) {
      bestShared = shared;
      best = n;
    }
  }
  return best;
}

export function RingGraph({
  nodes,
  edges,
  truncated,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
}) {
  const [tick, setTick] = useState(0);
  const hub = useMemo(() => findHub(nodes), [nodes]);
  const [selected, setSelected] = useState<string | null>(() => hub?.id ?? null);
  const [everClicked, setEverClicked] = useState(false);
  const simRef = useRef<Simulation<Positioned, undefined> | null>(null);

  const { simNodes, simLinks } = useMemo(() => {
    // Slight random jitter around center, not an exact shared point -- d3-force's
    // charge and collision forces have no gradient to act on when every node
    // starts at the identical coordinate, which stalls the first several ticks
    // before anything visibly separates.
    const copies: Positioned[] = nodes.map((n) => ({
      ...n,
      x: WIDTH / 2 + (Math.random() - 0.5) * 60,
      y: HEIGHT / 2 + (Math.random() - 0.5) * 60,
    }));
    const index = new Map(copies.map((n) => [n.id, n]));
    const links: SimLink[] = edges
      .map((e) => ({
        source: index.get(e.source)!,
        target: index.get(e.target)!,
        weight: e.weight,
      }))
      .filter((l) => l.source && l.target);
    return { simNodes: copies, simLinks: links };
  }, [nodes, edges]);

  useEffect(() => {
    const sim = forceSimulation(simNodes)
      .force(
        "link",
        forceLink<Positioned, SimLink>(simLinks)
          .id((d) => d.id)
          // Shared entities pull their accounts tight; private ones sit loose.
          .distance((l) => 70 - Math.min(l.weight, 8) * 4)
          .strength(0.7),
      )
      .force("charge", forceManyBody().strength(-220))
      .force("center", forceCenter(WIDTH / 2, HEIGHT / 2))
      .force("collide", forceCollide<Positioned>().radius((d) => radius(d) + 6))
      .force("x", forceX(WIDTH / 2).strength(0.04))
      .force("y", forceY(HEIGHT / 2).strength(0.06))
      .stop();

    // Pre-warm synchronously: run the layout to convergence before the first
    // paint, rather than animating it live from a standing start. Two things
    // depended on that animation finishing that shouldn't have to wait for
    // it -- a viewer's first glance, and a screenshot tool that has no way to
    // know "settled" from "still assembling."
    for (let i = 0; i < 300; i += 1) sim.tick();

    simRef.current = sim;
    setTick((t) => t + 1);
    return () => {
      sim.stop();
    };
  }, [simNodes, simLinks]);

  const neighbours = useMemo(() => {
    if (!selected) return new Set<string>();
    const out = new Set<string>([selected]);
    for (const e of edges) {
      if (e.source === selected) out.add(e.target);
      if (e.target === selected) out.add(e.source);
    }
    return out;
  }, [selected, edges]);

  const selectedNode = simNodes.find((n) => n.id === selected) ?? null;
  const dim = selected !== null;
  const isAutoHub = selected !== null && selected === hub?.id && !everClicked;

  function select(id: string | null) {
    setEverClicked(true);
    setSelected(id);
  }

  return (
    <div>
      <p className="max-w-2xl text-sm leading-relaxed text-muted">
        Every <strong className="text-paper">circle</strong> is one customer
        account. The other shapes are what several accounts share — a{" "}
        <strong className="text-paper">device</strong>,{" "}
        <strong className="text-paper">address</strong>,{" "}
        <strong className="text-paper">IP</strong>, or{" "}
        <strong className="text-paper">coupon</strong>. A shape drawn bigger and
        in fuller color is shared by more accounts — that's what turns a set of
        ordinary orders into a ring.
      </p>

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-faint">
        <span className="num">
          {nodes.length} nodes · {edges.length} links
        </span>
        <LegendItem color="var(--color-ink-600)" label="account" />
        <LegendItem color="var(--color-coral)" label="device" square />
        <LegendItem color="var(--color-blossom)" label="address" diamond />
        <LegendItem color="var(--color-butter)" label="IP" />
        <LegendItem color="var(--color-teal-deep)" label="coupon" square />
      </div>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="mt-3 w-full"
        role="img"
        aria-label={`Relationship graph: ${nodes.length} connected entities`}
        data-tick={tick}
        onClick={() => select(null)}
      >
        <g>
          {simLinks.map((l, i) => {
            const lit = !dim || (neighbours.has(l.source.id) && neighbours.has(l.target.id));
            return (
              <line
                key={i}
                x1={l.source.x}
                y1={l.source.y}
                x2={l.target.x}
                y2={l.target.y}
                stroke={lit ? "var(--color-ink-600)" : "var(--color-ink-800)"}
                strokeWidth={lit ? Math.min(1 + l.weight * 0.25, 2.5) : 0.6}
              />
            );
          })}
        </g>
        <g>
          {simNodes.map((n) => {
            const lit = !dim || neighbours.has(n.id);
            const shared = Number(n.attributes?.shared_by ?? 1);
            const r = radius(n);
            return (
              <g
                key={n.id}
                transform={`translate(${n.x},${n.y})`}
                opacity={lit ? 1 : 0.2}
                style={{ cursor: "pointer" }}
                onClick={(e) => {
                  e.stopPropagation();
                  select(n.id === selected ? null : n.id);
                }}
              >
                {n.id === selected && (
                  <circle
                    r={r + 5}
                    fill="none"
                    stroke={isAutoHub ? "var(--color-coral)" : "var(--color-paper)"}
                    strokeWidth={1.5}
                  />
                )}
                <Shape node={n} r={r} />
                {/* Always-on label for hub entities, so the "this is shared"
                    read doesn't depend on anyone clicking anything. */}
                {n.type !== "customer" && shared >= 2 && (
                  <text
                    y={r + 11}
                    textAnchor="middle"
                    className="num select-none"
                    style={{ fontSize: 8, fill: "var(--color-faint)" }}
                  >
                    ×{shared}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>

      {truncated && (
        <p className="mt-2 text-xs text-warn">
          Trimmed to stay readable. Every account is still shown — only the
          least-shared devices and addresses were hidden.
        </p>
      )}

      <div className="mt-4 min-h-[4.5rem] rounded border border-ink-700 bg-ink-800 p-4">
        {selectedNode ? (
          <>
            <div className="flex items-baseline justify-between gap-3">
              <div className="eyebrow">{TYPE_LABEL[selectedNode.type]}</div>
              {dim && (
                <button
                  type="button"
                  onClick={() => select(null)}
                  className="num text-[0.6875rem] text-faint underline-offset-2 hover:text-signal hover:underline"
                >
                  show every connection
                </button>
              )}
            </div>
            <div className="num mt-1 text-sm text-paper">{selectedNode.label}</div>
            <div className="mt-1.5 text-sm text-muted">
              {isAutoHub && (
                <span className="text-paper">
                  The strongest link in this case —{" "}
                </span>
              )}
              {selectedNode.type === "customer" ? (
                <>
                  Mean transaction risk{" "}
                  <span className="num text-paper">
                    {selectedNode.risk_score.toFixed(1)}
                  </span>
                  , connected to {neighbours.size - 1} entities.
                </>
              ) : (
                <>
                  Shared by{" "}
                  <span className="num text-paper">
                    {String(selectedNode.attributes?.shared_by ?? 1)}
                  </span>{" "}
                  accounts in this group. Every account touching it is lit up
                  below; everything else is dimmed.
                </>
              )}
            </div>
          </>
        ) : (
          <p className="text-sm text-faint">
            Click any shape above to see exactly what it connects to. Everything
            else dims so that one relationship stands alone.
          </p>
        )}
      </div>
    </div>
  );
}
