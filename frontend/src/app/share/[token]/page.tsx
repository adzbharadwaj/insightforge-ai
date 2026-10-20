import { Suspense, use } from "react";
import { ReportLoader } from "@/components/report-loader";

// Public, read-only: the token itself is the credential (signed, 7-day expiry, revocable).
function Shared({ params }: { params: Promise<{ token: string }> }) {
  const { token } = use(params);
  return <ReportLoader path={`/share/${token}`} shareToken={token} />;
}

export default function SharePage({ params }: PageProps<"/share/[token]">) {
  return (
    <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
      <Shared params={params} />
    </Suspense>
  );
}
