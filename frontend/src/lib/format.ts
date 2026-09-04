/**
 * Formatting and data access.
 *
 * Money uses Indian digit grouping (2,11,894 rather than 211,894) because the
 * merchant reading this reports in lakhs. Getting that wrong is the kind of
 * detail that tells a domain reader you did not build for them.
 */

import { useEffect, useState } from "react";

const inr = new Intl.NumberFormat("en-IN", {
  maximumFractionDigits: 0,
});

export function money(value: number): string {
  const sign = value < 0 ? "-" : "";
  return `${sign}₹${inr.format(Math.abs(Math.round(value)))}`;
}

export function count(value: number): string {
  return inr.format(value);
}

/** Metrics are shown to three decimals. Rounding them further would hide
 *  differences between models that are genuinely small. */
export function ratio(value: number): string {
  return value.toFixed(3);
}

export function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

export const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export type Fetched<T> =
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "ready"; data: T };

/** `reloadKey` forces a refetch of the same path without the URL changing --
 *  for data that the backend derives from something other than the resource's
 *  own address, like a ring's status, which moves when a decision is appended
 *  to the audit trail rather than when the ring itself is edited. Omit it and
 *  this behaves exactly as before: fetch once per distinct `path`. */
export function useApi<T>(path: string, reloadKey?: unknown): Fetched<T> {
  const [result, setResult] = useState<Fetched<T>>({ state: "loading" });

  useEffect(() => {
    let cancelled = false;

    fetch(`${API_BASE}${path}`)
      .then(async (res) => {
        if (!res.ok) {
          // Surface the API's own explanation rather than a status code. The
          // backend returns a directional message when an artifact is malformed.
          const body = await res.json().catch(() => null);
          throw new Error(body?.detail ?? `Request failed with status ${res.status}`);
        }
        return res.json() as Promise<T>;
      })
      .then((data) => {
        if (!cancelled) setResult({ state: "ready", data });
      })
      .catch((err: Error) => {
        if (!cancelled) setResult({ state: "error", message: err.message });
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, reloadKey]);

  return result;
}
