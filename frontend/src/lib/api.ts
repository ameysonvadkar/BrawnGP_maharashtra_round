// Typed client for the Black Box FastAPI backend (api/main.py).
// In development Vite proxies /api to http://127.0.0.1:8000 (see vite.config.ts).
import { queryOptions } from "@tanstack/react-query";

export type RunSummary = {
  run_id: string;
  question_id: string | null;
  question: string;
  gold: string | null;
  final_answer: string | null;
  success: boolean;
  fault_type: string | null;
  fault_step: number | null;
  parent_run_id: string | null;
  split: string;
  created_at: string | null;
  agent: string | null;
  n_reused?: number | null;
  n_reexecuted?: number | null;
};

export type Reason = { feature: string; value: number; shap_value: number };

export type StepRecord = {
  step_idx: number;
  type: "plan" | "retrieve" | "extract" | "calculate" | "answer" | string;
  input: unknown;
  output: unknown;
  state_before: Record<string, unknown>;
  parents: number[];
  cache_hit: boolean;
  latency_ms: number;
  error: string | null;
  blame_score: number;
  blame_rank: number | null;
  reasons: Reason[];
  direct_evidence: string[];
};

export type RunDetail = {
  run: RunSummary;
  top_blame_step: number | null;
  steps: StepRecord[];
};

export type StepDiff = {
  step_idx: number;
  type: string;
  output_a: unknown;
  output_b: unknown;
  input_changed: boolean;
  output_changed: boolean;
  cache_hit_a: number;
  cache_hit_b: number;
  changed: boolean;
};

export type RunDiff = {
  run_a_id: string;
  run_b_id: string;
  final_answer_a: string | null;
  final_answer_b: string | null;
  first_divergent_step: number | null;
  outcome_change: string;
  step_diffs: StepDiff[];
};

export type Savings = {
  tokens_full: number;
  tokens_spent: number;
  tokens_saved: number;
  pct_saved: number;
  usd_saved: number;
  inr_saved: number;
  inr_saved_per_1000: number;
};

export type Prices = { in_per_m: number; out_per_m: number; usd_inr: number };

export type ReplayResult = {
  run_id: string;
  parent_run_id: string;
  patched_steps: number[];
  final_answer: string | null;
  success: boolean;
  n_steps: number;
  n_reused: number;
  n_patched: number;
  n_reexecuted: number;
  n_prefix_reused: number;
  n_llm_reused: number;
  outcome_changed: boolean;
  diff: RunDiff;
  savings: Savings;
  prices: Prices;
};

export type VerifyRow = {
  rank: number;
  prediction: { step_idx: number; type: string; score: number };
  verification: { verified: boolean; root_cause_verified: boolean; input_matches_clean?: boolean };
};

export type Rate = { hits: number; rate: number; ci95: [number, number] };
export type MethodScores = { top1: number; top3: number; mrr: number };
export type SplitMetrics = {
  n_runs: number;
  metrics: Record<"random" | "last_step" | "first_anomaly" | "lightgbm", MethodScores>;
  replay_verified: {
    lightgbm_top1: number;
    lightgbm_top1_root_cause: number;
    root_cause_ci95?: [number, number];
    label_confirmed: number;
    per_fault?: Record<string, { n_runs: number; top1: Rate; root_cause_verified: Rate }>;
  };
};
export type Metrics = { seen_faults_test: SplitMetrics; heldout_faults: SplitMetrics };

export type Dashboard = {
  total_runs: number;
  passed: number;
  failed: number;
  by_split: Record<string, number>;
  by_fault: Record<string, number>;
  metrics: Metrics;
  cost_prices: Prices;
};

export type Health = { status: string; database: boolean; model: boolean; metrics: boolean };

export type Company = { id: string; name: string; [field: string]: unknown };

export type LiveInjection = {
  injection: {
    run_id: string;
    fault_type: string;
    fault_step: number;
    source_run_id: string;
    final_answer: string | null;
  };
  caught_top1: boolean;
};

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),
  dashboard: () => request<Dashboard>("/dashboard"),
  runs: (params: { success?: boolean | undefined; limit?: number | undefined }) => {
    const q = new URLSearchParams({ limit: String(params.limit ?? 500) });
    if (params.success !== undefined) q.set("success", String(params.success));
    return request<RunSummary[]>(`/runs?${q}`);
  },
  run: (runId: string) => request<RunDetail>(`/runs/${encodeURIComponent(runId)}`),
  replays: (runId: string) => request<RunSummary[]>(`/runs/${encodeURIComponent(runId)}/replays`),
  cleanReference: (runId: string) =>
    request<{ run: RunSummary | null }>(`/runs/${encodeURIComponent(runId)}/clean-reference`),
  oracle: (runId: string, step: number) =>
    request<{ output: unknown }>(`/runs/${encodeURIComponent(runId)}/oracle/${step}`),
  diff: (a: string, b: string) =>
    request<RunDiff>(`/runs/${encodeURIComponent(a)}/diff/${encodeURIComponent(b)}`),
  savings: (runId: string, replayId: string) =>
    request<{ savings: Savings; prices: Prices }>(
      `/runs/${encodeURIComponent(runId)}/replay-savings/${encodeURIComponent(replayId)}`,
    ),
  replay: (runId: string, step: number, patch: unknown) =>
    request<ReplayResult>(`/runs/${encodeURIComponent(runId)}/replays`, {
      method: "POST",
      body: JSON.stringify({ overrides: { [step]: patch } }),
    }),
  verifyTop: (runId: string) =>
    request<{ results: VerifyRow[] }>(`/runs/${encodeURIComponent(runId)}/verify-top`, {
      method: "POST",
      body: JSON.stringify({ top_k: 3 }),
    }),
  knowledgeBase: () => request<Company[]>("/knowledge-base"),
  injectFault: () =>
    request<LiveInjection>("/live-demo/injections", { method: "POST", body: "{}" }),
};

export const DEMO_RUN_ID = "r_9002";

export const queries = {
  health: () =>
    queryOptions({
      queryKey: ["health"],
      queryFn: api.health,
      refetchInterval: 10_000,
      retry: false,
    }),
  dashboard: () =>
    queryOptions({ queryKey: ["dashboard"], queryFn: api.dashboard, staleTime: 30_000 }),
  runs: (success?: boolean) =>
    queryOptions({ queryKey: ["runs", success ?? "all"], queryFn: () => api.runs({ success }) }),
  run: (runId: string) =>
    queryOptions({ queryKey: ["run", runId], queryFn: () => api.run(runId), staleTime: 60_000 }),
  replays: (runId: string) =>
    queryOptions({ queryKey: ["replays", runId], queryFn: () => api.replays(runId) }),
  cleanReference: (runId: string) =>
    queryOptions({
      queryKey: ["clean-ref", runId],
      queryFn: () => api.cleanReference(runId),
      staleTime: Infinity,
    }),
  diff: (a: string, b: string) =>
    queryOptions({ queryKey: ["diff", a, b], queryFn: () => api.diff(a, b), staleTime: Infinity }),
  knowledgeBase: () =>
    queryOptions({ queryKey: ["kb"], queryFn: api.knowledgeBase, staleTime: Infinity }),
};
