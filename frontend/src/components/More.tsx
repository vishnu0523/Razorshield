/**
 * Progressive disclosure for the methodology prose.
 *
 * The product's credibility rests on paragraphs like "the threshold was
 * chosen on validation, not on the held-out split" — that text has to stay on
 * the page somewhere. But printed open by default under every panel, five of
 * these read as a wall of caveats before a reader has even looked at a
 * number. Collapsed by default, one line, expandable on demand: the number
 * leads, the reasoning is one click away rather than unavoidable.
 */

import { useState, type ReactNode } from "react";

export function More({
  label = "Why this matters",
  defaultOpen = false,
  children,
}: {
  label?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="group inline-flex items-center gap-1.5 text-xs font-medium tracking-wide text-faint transition-colors hover:text-signal"
      >
        <span
          className={`inline-block text-signal transition-transform duration-300 ease-out ${
            open ? "rotate-90" : ""
          }`}
          aria-hidden
        >
          ›
        </span>
        {open ? "Hide" : label}
      </button>
      <div
        className="grid transition-[grid-template-rows] duration-300 ease-out"
        style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
      >
        <div className="overflow-hidden">
          <p className="mt-2 max-w-2xl text-sm leading-relaxed text-faint">
            {children}
          </p>
        </div>
      </div>
    </div>
  );
}
