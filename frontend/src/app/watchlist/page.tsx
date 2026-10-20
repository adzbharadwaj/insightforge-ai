import { RequireAuth } from "@/components/auth";
import { WatchlistDashboard } from "@/components/watchlist";

export default function WatchlistPage() {
  return (
    <RequireAuth>
      <h1 className="mb-1 text-2xl font-semibold tracking-tight">Watchlist</h1>
      <p className="mb-6 text-sm text-muted">Topics re-researched on a schedule. You&apos;re alerted only when findings materially change.</p>
      <WatchlistDashboard />
    </RequireAuth>
  );
}
