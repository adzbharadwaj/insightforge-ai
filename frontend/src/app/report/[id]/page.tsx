import { Suspense, use } from "react";
import { RequireAuth } from "@/components/auth";
import { ReportLoader } from "@/components/report-loader";

function Report({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  return <ReportLoader path={`/reports/${id}`} />;
}

export default function ReportPage({ params }: PageProps<"/report/[id]">) {
  return (
    <RequireAuth>
      <Suspense fallback={<p className="text-sm text-muted">Loading…</p>}>
        <Report params={params} />
      </Suspense>
    </RequireAuth>
  );
}
