/**
 * Shared "the backend isn't running" screen.
 *
 * Every tab fetches independently, so every tab needs this — without it, a
 * stopped API surfaces as the browser's raw `TypeError: Failed to fetch`
 * dropped onto the page as a single line of red text. That's a confusing
 * thing to hand a judge who just wants to know what to do next.
 */
export function ApiUnreachable({ message }: { message: string }) {
  return (
    <div className="mx-auto max-w-2xl px-5 py-16 sm:px-8">
      <h2 className="text-lg font-semibold text-paper">The API is not responding</h2>
      <p className="mt-3 text-sm leading-relaxed text-muted">
        This screen reads from the RazorShield API on port 8000. Start it in a
        second terminal, then reload this page.
      </p>
      <p className="num mt-4 rounded border border-ink-700 bg-ink-850 px-4 py-3 text-sm text-signal">
        make api
      </p>
      <p className="num mt-4 text-xs text-faint">{message}</p>
    </div>
  );
}
