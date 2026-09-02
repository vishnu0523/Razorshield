/**
 * Motion primitives.
 *
 * Kept dependency-free on purpose: this is a two-terminal judged project, and
 * a UI animation library is not worth the extra `npm install` surface. Two
 * small hooks cover everything the dashboard needs — a eased number roll for
 * money figures that change under the reader's hand, and a reduced-motion
 * check so neither animation fights an accessibility preference.
 */

import { useEffect, useRef, useState } from "react";

export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Eases a displayed number toward `target` instead of snapping to it. Used
 *  for the ledger figures so dragging an assumption slider reads as a live
 *  recalculation rather than a jump cut. */
export function useAnimatedNumber(target: number, duration = 500): number {
  const [value, setValue] = useState(target);
  const raf = useRef<number | null>(null);
  const from = useRef(target);
  const first = useRef(true);

  useEffect(() => {
    if (first.current) {
      first.current = false;
      from.current = target;
      setValue(target);
      return;
    }
    if (prefersReducedMotion()) {
      from.current = target;
      setValue(target);
      return;
    }

    const start = performance.now();
    const startValue = from.current;
    const delta = target - startValue;
    if (raf.current) cancelAnimationFrame(raf.current);
    if (delta === 0) return;

    function tick(now: number) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(startValue + delta * eased);
      if (t < 1) {
        raf.current = requestAnimationFrame(tick);
      } else {
        from.current = target;
        setValue(target);
      }
    }
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, duration]);

  return value;
}
