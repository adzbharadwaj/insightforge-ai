import { Suspense, use } from "react";
import { RequireAuth } from "@/components/auth";
import { RunProgress } from "@/components/run-progress";

function Run({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  return <RunProgress runId={id} />;
}

export default function RunPage({ params }: PageProps<"/run/[id]">) {
  return (
    <RequireAuth>
      <h1 className="mb-6 text-2xl font-semibold tracking-tight">Research run</h1>
      <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
        <Run params={params} />
      </Suspense>
    </RequireAuth>
  );
}
