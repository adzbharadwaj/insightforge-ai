// Mirrors the backend response models (insightforge/api/views.py, research.py, watchlist.py).

export type RunStatus =
  | "QUEUED" | "PLANNING" | "AWAITING_CLARIFICATION" | "AWAITING_REVIEW" | "RESEARCHING" | "SYNTHESIZING"
  | "WRITING" | "GROUNDING" | "FACT_CHECKING" | "COMPLETE" | "FAILED" | "CANCELLED";

export type SubTaskStatus = "Queued" | "Running" | "Done" | "Failed";
export type Confidence = "HIGH" | "MEDIUM" | "LOW";

export interface SubTaskView {
  id: string;
  query: string;
  source_type: string;
  priority: number;
  status: SubTaskStatus;
  summary: string | null;
  error: string | null;
}

export interface RunSnapshot {
  run_id: string;
  status: RunStatus;
  brief: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  elapsed_s: number | null;
  sub_tasks: SubTaskView[];
  clarifying_question: string | null;
  review: { conflicts: ConflictReview[] } | null;
  error: string | null;
  report_id: string | null;
  tokens_used: number;
  attempts: number;
  eta_seconds: number;
}

export interface ConflictOption {
  option_id: string;
  value: string;
  evidence_ids: string[];
  claims: string[];
  sources: string[];
}

export interface ConflictReview {
  conflict_id: string;
  subject: string;
  predicate: string;
  qualifier: string | null;
  options: ConflictOption[];
}

export interface EntityNode {
  key: string;
  name: string;
  type: string;
  source_ids: string[];
}

export interface Relationship {
  subject: string;
  predicate: string;
  object: string;
  qualifier: string | null;
  source_id: string;
  source_url: string | null;
  date: string | null;
  conflict: boolean;
}

export interface EntityGraph {
  entities: EntityNode[];
  relationships: Relationship[];
  conflicts: { subject: string; predicate: string; qualifier: string | null; values: { object: string; source_ids: string[] }[] }[];
}

export interface LogEvent {
  id: string;
  agent: string;
  kind: string;
  message: string;
  at: string;
}

export interface WrittenSentence {
  text: string;
  evidence_ids: string[];
}

export interface WrittenSection {
  sub_task_id: string;
  heading: string;
  sentences: WrittenSentence[];
}

export interface WriterOutput {
  title: string;
  executive_summary: WrittenSentence[];
  sections: WrittenSection[];
  gaps: string[];
}

export interface ClaimRef {
  claim_id: string;
  section: string;
  text: string;
  evidence_ids: string[];
  loc: string;
}

export interface SourceMeta {
  source_id: string;
  kind: string;
  label: string;
  url: string | null;
  title: string | null;
  published_date: string | null;
  credibility: number | null;
}

export interface EvidenceView {
  evidence_id: string;
  claim: string;
  passage: string | null;
  source_ids: string[];
  confidence: Confidence;
  published_date: string | null;
  credibility: number;
  conflict: boolean;
  entity_keys: string[];
}

export interface ClaimCheck {
  claim_id: string;
  text: string;
  verified: boolean;
  similarity: number | null;
  source_id: string | null;
  checked_against: string;
  note: string;
}

export interface FactCheckSummary {
  total_checked: number;
  passed: number;
  failed: number;
  pass_rate: number;
  unverified_claims: string[];
  checks: ClaimCheck[];
}

export interface ReportView {
  id: string;
  title: string;
  brief: string;
  created_at: string;
  markdown: string;
  structure: { report: WriterOutput; claims: ClaimRef[] } | null;
  fact_check: FactCheckSummary | null;
  sections: { theme: string; sub_task_id: string; confidence: Confidence }[];
  sources: Record<string, SourceMeta>;
  evidence: Record<string, EvidenceView>;
  entity_graph: EntityGraph | null;
}

export interface ReportSummary {
  id: string;
  run_id: string;
  title: string;
  brief: string;
  created_at: string;
  pass_rate: number | null;
}

export interface ChangeSummary {
  added_findings: { text: string }[];
  removed_findings: { text: string }[];
  changed_findings: { before: string; after: string; reason: string }[];
  note?: string;
}

export interface AlertView {
  id: string;
  watchlist_item_id: string;
  topic: string;
  run_id: string;
  report_id: string | null;
  change_summary: ChangeSummary;
  read: boolean;
  webhook_status: string | null;
  created_at: string;
}

export interface WatchView {
  id: string;
  topic: string;
  cadence: "daily" | "weekly";
  active: boolean;
  webhook_url: string | null;
  last_run: string | null;
  next_run: string | null;
  created_at: string;
  latest_alert: AlertView | null;
}
