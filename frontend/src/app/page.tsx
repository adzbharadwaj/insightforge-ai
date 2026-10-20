import { RequireAuth } from "@/components/auth";
import { BriefForm } from "@/components/brief-form";

export default function Home() {
  return (
    <RequireAuth>
      <div className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">New research brief</h1>
        <p className="mt-1 text-sm text-muted">
          Describe what to research. You get a cited report; every claim traces to a source and a sample is
          fact-checked before you see it.
        </p>
      </div>
      <BriefForm />
    </RequireAuth>
  );
}
