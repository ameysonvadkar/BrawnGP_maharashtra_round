import { useEffect, useMemo, useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  Check,
  FlaskConical,
  Play,
  RotateCcw,
  ShieldCheck,
  Wand2,
  X,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { StatusStrip, StepTimeline, type StepMark } from "@/components/blackbox";
import { useUtcClock } from "@/hooks/use-blackbox";
import {
  api,
  DEMO_RUN_ID,
  queries,
  type ReplayResult,
  type RunDetail,
  type RunSummary,
  type SplitMetrics,
  type StepRecord,
  type VerifyRow,
} from "@/lib/api";
import {
  describeInput,
  downstream,
  explainReason,
  FAULT_LABELS,
  isObj,
  pad2,
  pct,
  relativeTime,
  statusOf,
  summarize,
} from "@/lib/trace";

type Tab = "debugger" | "compare" | "metrics";
type DebuggerSearch = {
  run?: string | undefined;
  step?: number | undefined;
  tab?: Tab | undefined;
};

export const Route = createFileRoute("/debugger")({
  validateSearch: (search: Record<string, unknown>): DebuggerSearch => ({
    run:
      typeof search["run"] === "string" && /^[A-Za-z0-9_-]{1,80}$/.test(search["run"])
        ? search["run"]
        : undefined,
    step:
      typeof search["step"] === "number" && Number.isInteger(search["step"]) && search["step"] >= 0
        ? search["step"]
        : undefined,
    tab: search["tab"] === "compare" || search["tab"] === "metrics" ? search["tab"] : "debugger",
  }),
  head: () => ({
    meta: [
      { title: "Debugger | Black Box agent traces" },
      {
        name: "description",
        content:
          "Inspect agent run steps, evidence, replay checkpoints, trace differences, and evaluation metrics.",
      },
      { property: "og:title", content: "Debugger | Black Box agent traces" },
      {
        property: "og:description",
        content:
          "Inspect agent run steps, evidence, replay checkpoints, trace differences, and evaluation metrics.",
      },
    ],
  }),
  component: Debugger,
});

const json = (v: unknown) => JSON.stringify(v, null, 2);

// ── Run list ────────────────────────────────────────────────────────────────

function RunList({
  activeId,
  onSelect,
  onInjected,
}: {
  activeId: string;
  onSelect: (id: string) => void;
  onInjected: (id: string, note: string) => void;
}) {
  const [filter, setFilter] = useState<"failed" | "all" | "passed">("failed");
  const [term, setTerm] = useState("");
  const runs = useQuery(queries.runs(filter === "all" ? undefined : filter === "passed"));
  const queryClient = useQueryClient();
  const inject = useMutation({
    mutationFn: api.injectFault,
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["runs"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard"] });
      const i = res.injection;
      onInjected(
        i.run_id,
        `Injected ${FAULT_LABELS[i.fault_type] ?? i.fault_type} at step ${pad2(i.fault_step)} of a clean run · ranker ${res.caught_top1 ? "caught it at rank 1 ✓" : "did not rank it first"}`,
      );
    },
  });

  const visible = useMemo(() => {
    const t = term.trim().toLowerCase();
    const list = runs.data ?? [];
    const ordered = [...list].sort(
      (a, b) => order(a) - order(b) || (b.created_at ?? "").localeCompare(a.created_at ?? ""),
    );
    return t
      ? ordered.filter(
          (r) => r.run_id.toLowerCase().includes(t) || r.question.toLowerCase().includes(t),
        )
      : ordered;
  }, [runs.data, term]);

  return (
    <aside className="bb-run-list" aria-label="Run list">
      <div className="bb-run-heading">
        <span className="bb-mono">RUNS</span>
        <span className="bb-mono bb-meta">
          {runs.data ? String(visible.length).padStart(3, "0") : "…"}
        </span>
      </div>
      <Button
        className="bb-btn-shadcn bb-inject"
        onClick={() => inject.mutate()}
        disabled={inject.isPending}
      >
        <Zap size={15} /> {inject.isPending ? "Injecting…" : "Inject live fault"}
      </Button>
      {inject.isError && <p className="bb-inline-error">{(inject.error as Error).message}</p>}
      <div className="bb-run-filter" aria-label="Filter runs">
        {(["failed", "all", "passed"] as const).map((value) => (
          <Button
            variant="outline"
            key={value}
            type="button"
            onClick={() => setFilter(value)}
            className={`bb-filter-button bb-mono${filter === value ? " bb-filter-active" : ""}`}
            aria-pressed={filter === value}
          >
            {value}
          </Button>
        ))}
      </div>
      <input
        className="bb-field bb-run-search"
        placeholder="Search id or question"
        value={term}
        onChange={(e) => setTerm(e.target.value)}
        aria-label="Search runs"
      />
      <div className="bb-run-items">
        {runs.isPending && (
          <div className="bb-empty">
            <span className="bb-meta">Loading runs…</span>
          </div>
        )}
        {runs.isError && (
          <div className="bb-empty">
            <span className="bb-plate bb-mono">API OFFLINE</span>
            <p>
              Start the backend: <code>uvicorn api.main:app</code>
            </p>
          </div>
        )}
        {visible.map((r) => (
          <button
            key={r.run_id}
            type="button"
            onClick={() => onSelect(r.run_id)}
            className={`bb-run-item${r.run_id === activeId ? " bb-run-active" : ""}`}
          >
            <span className="bb-run-main">
              <span className="bb-mono bb-run-id">{r.run_id}</span>
              {(r.split === "demo" || r.split === "live_demo" || r.agent) && (
                <span className="bb-run-tags">
                  {r.split === "demo" && <span className="bb-tag">demo</span>}
                  {r.split === "live_demo" && (
                    <span className="bb-tag bb-tag-live">live injection</span>
                  )}
                  {r.agent && <span className="bb-tag">{r.agent} agent</span>}
                </span>
              )}
              <span className="bb-meta bb-run-q">{r.question}</span>
            </span>
            <span className={`bb-mono ${r.success ? "bb-run-pass" : "bb-run-fail"}`}>
              {r.success ? "pass" : "fail"}
            </span>
          </button>
        ))}
      </div>
    </aside>
  );
}

function order(r: RunSummary) {
  return r.split === "live_demo" ? 0 : r.split === "demo" ? 1 : r.split === "sdk" ? 2 : 3;
}

// ── Patch editor ────────────────────────────────────────────────────────────

function PatchEditor({
  detail,
  step,
  onReplayed,
}: {
  detail: RunDetail;
  step: StepRecord;
  onReplayed: (r: ReplayResult) => void;
}) {
  const runId = detail.run.run_id;
  const isCompanyRetrieve = step.type === "retrieve" && !detail.run.agent;
  const kb = useQuery({ ...queries.knowledgeBase(), enabled: isCompanyRetrieve });
  const queryClient = useQueryClient();
  const output = isObj(step.output) ? step.output : {};
  const [text, setText] = useState(json(step.output));
  const [docId, setDocId] = useState<string>(String(output.doc_id ?? ""));
  const [note, setNote] = useState("");

  // The document a clean run of the same question retrieved: the suggested fix.
  const suggestion = useQuery({
    queryKey: ["oracle", runId, step.step_idx],
    queryFn: () => api.oracle(runId, step.step_idx),
    enabled: isCompanyRetrieve,
    retry: false,
    staleTime: Infinity,
  });
  const suggestedDoc = isObj(suggestion.data?.output)
    ? String(suggestion.data.output.doc_id ?? "")
    : "";

  useEffect(() => {
    setText(json(step.output));
    setDocId(suggestedDoc || (isObj(step.output) ? String(step.output.doc_id ?? "") : ""));
    setNote("");
  }, [runId, step.step_idx, step.output, suggestedDoc]);

  const oracle = useMutation({
    mutationFn: () => api.oracle(runId, step.step_idx),
    onSuccess: (res) => {
      setText(json(res.output));
      const id = isObj(res.output) ? res.output.doc_id : undefined;
      if (typeof id === "string") setDocId(id);
      setNote("Loaded the output this step produced in a clean run of the same question.");
    },
    onError: (e) => setNote((e as Error).message),
  });

  const replay = useMutation({
    mutationFn: (patch: unknown) => api.replay(runId, step.step_idx, patch),
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["replays", runId] });
      onReplayed(res);
    },
  });

  function submit() {
    let patch: unknown;
    if (isCompanyRetrieve) {
      const company = kb.data?.find((c) => c.id === docId);
      if (!company) return setNote("Pick a document first.");
      patch = {
        doc_id: company.id,
        title: company.name,
        company,
        top_scores: output.top_scores ?? [1, 0],
      };
    } else {
      try {
        patch = JSON.parse(text);
      } catch (e) {
        return setNote(`Invalid JSON: ${(e as Error).message}`);
      }
    }
    setNote("");
    replay.mutate(patch);
  }

  return (
    <section className="bb-panel">
      <h3 className="bb-panel-heading">
        <Wand2 size={15} /> Patch step {pad2(step.step_idx)} and replay
      </h3>
      {isCompanyRetrieve ? (
        <>
          <label className="bb-field-label" htmlFor="doc-pick">
            Replace the retrieved document with
          </label>
          <select
            id="doc-pick"
            className="bb-field"
            value={docId}
            onChange={(e) => setDocId(e.target.value)}
          >
            {(kb.data ?? []).map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
                {c.id === suggestedDoc ? " (clean run)" : ""}
              </option>
            ))}
          </select>
          {suggestedDoc && (
            <p className="bb-meta">
              Suggested: the document a clean run of this question retrieved.
            </p>
          )}
        </>
      ) : (
        <>
          <label className="bb-field-label" htmlFor="patch-source">
            Patched output (JSON)
          </label>
          <textarea
            id="patch-source"
            className="bb-field bb-textarea bb-mono"
            value={text}
            onChange={(e) => setText(e.target.value)}
            spellCheck={false}
          />
        </>
      )}
      <div className="bb-patch-actions">
        <Button
          variant="outline"
          className="bb-btn-shadcn bb-btn-outline"
          onClick={() => oracle.mutate()}
          disabled={oracle.isPending}
        >
          <FlaskConical size={15} /> Load clean-run output
        </Button>
        <Button className="bb-btn-shadcn" onClick={submit} disabled={replay.isPending}>
          <Play size={15} /> {replay.isPending ? "Replaying…" : "Patch and replay"}
        </Button>
      </div>
      <p className="bb-meta">
        Steps before {pad2(step.step_idx)} are replayed from the recording; later steps re-run only
        if their inputs change.
      </p>
      {(note || replay.isError) && (
        <p className="bb-inline-status bb-mono" role="status">
          {note || (replay.error as Error).message}
        </p>
      )}
    </section>
  );
}

// ── Replay result ───────────────────────────────────────────────────────────

function ReplayPanel({ result }: { result: ReplayResult }) {
  const s = result.savings;
  return (
    <section className="bb-panel bb-replay-checkpoints" aria-live="polite">
      <h3 className="bb-panel-heading">Replay {result.run_id}</h3>
      <div
        className={`bb-verdict bb-mono ${result.success ? "bb-verdict-pass" : "bb-verdict-fail"}`}
      >
        {result.success && result.outcome_changed ? (
          <>
            <Check size={15} /> fail → pass · patching step {pad2(result.patched_steps[0] ?? 0)}{" "}
            alone fixes the run
          </>
        ) : result.success ? (
          <>
            <Check size={15} /> still passing
          </>
        ) : (
          <>
            <X size={15} /> still failing after the patch
          </>
        )}
        <span className="bb-verdict-answer">“{result.final_answer}”</span>
      </div>
      <div className="bb-counter-grid bb-counter-grid-4">
        {[
          [result.n_prefix_reused, "steps reused before the patch"],
          [result.n_patched, "patched"],
          [result.n_reexecuted, "re-executed"],
          [result.n_llm_reused, "LLM calls avoided"],
        ].map(([n, label]) => (
          <div className="bb-counter" key={label as string}>
            <div className="bb-numeral bb-mono">{n as number}</div>
            <div className="bb-counter-label">{label as string}</div>
          </div>
        ))}
      </div>
      {s.tokens_full > 0 && (
        <p className="bb-section-copy bb-mono bb-meta">
          {s.tokens_saved.toLocaleString()} of {s.tokens_full.toLocaleString()} est. LLM tokens not
          re-spent ({pct(s.pct_saved)}) · ≈ ₹{s.inr_saved_per_1000.toFixed(2)} saved per 1,000 fixes{" "}
          (${result.prices.in_per_m}/${result.prices.out_per_m} per 1M tokens, ₹
          {result.prices.usd_inr}/$)
        </p>
      )}
    </section>
  );
}

function VerifyPanel({ runId, disabled }: { runId: string; disabled: boolean }) {
  const [rows, setRows] = useState<VerifyRow[] | null>(null);
  const verify = useMutation({
    mutationFn: () => api.verifyTop(runId),
    onSuccess: (r) => setRows(r.results),
  });
  useEffect(() => setRows(null), [runId]);
  return (
    <section className="bb-panel">
      <h3 className="bb-panel-heading">
        <ShieldCheck size={15} /> Verify the top-3 suspects
      </h3>
      <p className="bb-meta">
        Patches each suspect alone with its clean-run output and replays. A root cause received
        correct input but produced wrong output.
      </p>
      <Button
        variant="outline"
        className="bb-btn-shadcn bb-btn-outline"
        onClick={() => verify.mutate()}
        disabled={disabled || verify.isPending}
      >
        {verify.isPending ? "Verifying…" : "Run auto-verify"}
      </Button>
      {verify.isError && <p className="bb-inline-error">{(verify.error as Error).message}</p>}
      {rows && (
        <table className="bb-table bb-mono">
          <thead>
            <tr>
              <th>step</th>
              <th>blame</th>
              <th>flips</th>
              <th>input ok</th>
              <th>root cause</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.rank} className={r.verification.root_cause_verified ? "bb-row-good" : ""}>
                <td>
                  {pad2(r.prediction.step_idx)} {r.prediction.type}
                </td>
                <td>{r.prediction.score.toFixed(2)}</td>
                <td>{r.verification.verified ? "✓" : "✕"}</td>
                <td>{r.verification.input_matches_clean ? "✓" : "✕"}</td>
                <td>{r.verification.root_cause_verified ? "✓ verified" : "✕"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

// ── Compare & metrics ───────────────────────────────────────────────────────

function ComparePanel({ runId, preferred }: { runId: string; preferred?: string | undefined }) {
  const replays = useQuery(queries.replays(runId));
  const clean = useQuery(queries.cleanReference(runId));
  const options = [
    ...(replays.data ?? [])
      .slice()
      .sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""))
      .map((r) => ({ id: r.run_id, label: `replay ${r.run_id} · ${r.success ? "pass" : "fail"}` })),
    ...(clean.data?.run
      ? [{ id: clean.data.run.run_id, label: `clean run ${clean.data.run.run_id}` }]
      : []),
  ];
  const [other, setOther] = useState<string | undefined>(preferred);
  const target = other && options.some((o) => o.id === other) ? other : options[0]?.id;
  const diffQuery = useQuery({ ...queries.diff(runId, target ?? ""), enabled: Boolean(target) });
  const d = diffQuery.data;

  if (!options.length)
    return (
      <div className="bb-empty">
        <span className="bb-plate bb-mono">NOTHING TO COMPARE</span>
        <p>Patch a step and replay, then compare the two traces here.</p>
      </div>
    );
  return (
    <section>
      <div className="bb-workbench-heading">
        <div>
          <p className="bb-mono bb-meta">TRACE COMPARISON</p>
          <h2 className="bb-app-title">
            {runId} vs {target}
          </h2>
        </div>
        <select
          className="bb-field bb-compare-select"
          value={target}
          onChange={(e) => setOther(e.target.value)}
          aria-label="Compare with"
        >
          {options.map((o) => (
            <option key={o.id} value={o.id}>
              {o.label}
            </option>
          ))}
        </select>
      </div>
      {d && (
        <>
          <p className="bb-mono bb-meta">
            FIRST DIVERGENCE · STEP{" "}
            {d.first_divergent_step === null ? "—" : pad2(d.first_divergent_step)} · OUTCOME{" "}
            {d.outcome_change.toUpperCase()}
          </p>
          <div className="bb-compare-grid">
            {(["a", "b"] as const).map((side) => (
              <div className="bb-compare-card" key={side}>
                <h3 className="bb-compare-heading bb-mono">
                  {side === "a" ? `ORIGINAL ${d.run_a_id}` : `COMPARED ${d.run_b_id}`}
                </h3>
                {d.step_diffs.map((row) => (
                  <div
                    key={row.step_idx}
                    className={`bb-compare-row${row.changed ? " bb-compare-row-changed" : ""}`}
                  >
                    <span className="bb-mono bb-meta">{pad2(row.step_idx)}</span>
                    <span>{summarize(row.type, side === "a" ? row.output_a : row.output_b)}</span>
                    {side === "b" && (
                      <span className="bb-mono bb-meta bb-source">
                        {row.cache_hit_b
                          ? "reused"
                          : !row.input_changed && row.output_changed
                            ? "patched"
                            : "re-run"}
                      </span>
                    )}
                  </div>
                ))}
                <div className="bb-diff-outcome bb-mono">
                  {side === "a" ? d.final_answer_a : d.final_answer_b}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function SplitTable({ title, split }: { title: string; split: SplitMetrics }) {
  const rows: [string, keyof SplitMetrics["metrics"]][] = [
    ["Random step", "random"],
    ["Last step", "last_step"],
    ["First anomaly", "first_anomaly"],
    ["LightGBM ranker", "lightgbm"],
  ];
  return (
    <div className="bb-panel">
      <h3 className="bb-panel-heading">
        {title} <span className="bb-meta">· {split.n_runs} failed runs</span>
      </h3>
      <table className="bb-table bb-mono">
        <thead>
          <tr>
            <th>method</th>
            <th>top-1</th>
            <th>top-3</th>
            <th>MRR</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, key]) => (
            <tr key={key} className={key === "lightgbm" ? "bb-row-good" : ""}>
              <td>{label}</td>
              <td>
                <span className="bb-inline-bar">
                  <span style={{ width: `${split.metrics[key].top1 * 100}%` }} />
                </span>
                {pct(split.metrics[key].top1, 1)}
              </td>
              <td>{pct(split.metrics[key].top3, 1)}</td>
              <td>{split.metrics[key].mrr.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {split.replay_verified.per_fault && (
        <>
          <p className="bb-mono bb-meta bb-subhead">
            ROOT CAUSE VERIFIED BY REPLAY, PER FAULT (95% CI)
          </p>
          {Object.entries(split.replay_verified.per_fault).map(([fault, v]) => (
            <div className="bb-fault-row" key={fault}>
              <span>
                {FAULT_LABELS[fault] ?? fault} <span className="bb-meta">n={v.n_runs}</span>
              </span>
              <span className="bb-metric-bar bb-ci-bar">
                <span style={{ width: `${v.root_cause_verified.rate * 100}%` }} />
                <i
                  style={{
                    left: `${v.root_cause_verified.ci95[0] * 100}%`,
                    width: `${(v.root_cause_verified.ci95[1] - v.root_cause_verified.ci95[0]) * 100}%`,
                  }}
                />
              </span>
              <span className="bb-mono">{pct(v.root_cause_verified.rate)}</span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

function MetricsPanel() {
  const dashboard = useQuery(queries.dashboard());
  const d = dashboard.data;
  if (!d)
    return (
      <div className="bb-empty">
        <span className="bb-meta">
          {dashboard.isError ? "Metrics unavailable: is the API running?" : "Loading metrics…"}
        </span>
      </div>
    );
  const m = d.metrics;
  return (
    <section>
      <div className="bb-workbench-heading">
        <div>
          <p className="bb-mono bb-meta">EVALUATION · data/metrics.json</p>
          <h2 className="bb-app-title">Fault localisation accuracy</h2>
        </div>
      </div>
      <div className="bb-metric-grid">
        {[
          ["TOP-1 · SEEN", m.seen_faults_test.metrics.lightgbm.top1],
          ["TOP-1 · HELD-OUT", m.heldout_faults.metrics.lightgbm.top1],
          [
            "ROOT CAUSE VERIFIED · HELD-OUT",
            m.heldout_faults.replay_verified.lightgbm_top1_root_cause,
          ],
        ].map(([label, value]) => (
          <article key={label as string} className="bb-metric-card">
            <div className="bb-mono bb-meta">{label as string}</div>
            <div className="bb-metric-value">{pct(value as number, 1)}</div>
            <div className="bb-metric-bar">
              <span style={{ width: `${(value as number) * 100}%` }} />
            </div>
          </article>
        ))}
      </div>
      <div className="bb-detail-grid bb-metrics-stack">
        <SplitTable title="Seen fault types · unseen questions" split={m.seen_faults_test} />
        <SplitTable title="Held-out fault types · never trained on" split={m.heldout_faults} />
      </div>
      <div className="bb-panel">
        <h3 className="bb-panel-heading">
          Recorded dataset{" "}
          <span className="bb-meta">
            · {d.total_runs} runs · {d.passed} passed · {d.failed} failed
          </span>
        </h3>
        <div className="bb-chip-row">
          {Object.entries(d.by_fault)
            .sort((a, b) => b[1] - a[1])
            .map(([f, n]) => (
              <span className="bb-chip bb-mono" key={f}>
                {FAULT_LABELS[f] ?? f} · {n}
              </span>
            ))}
        </div>
      </div>
    </section>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

function Debugger() {
  const search = Route.useSearch();
  const navigate = Route.useNavigate();
  const clock = useUtcClock();
  const runId = search.run ?? DEMO_RUN_ID;
  const tab: Tab = search.tab ?? "debugger";
  const detailQuery = useQuery(queries.run(runId));
  const detail = detailQuery.data;
  const [replayResult, setReplayResult] = useState<ReplayResult | null>(null);
  const [notice, setNotice] = useState("");

  useEffect(() => setReplayResult(null), [runId]);

  const selected = search.step ?? detail?.top_blame_step ?? 0;
  const step = detail?.steps.find((s) => s.step_idx === selected) ?? detail?.steps[0];
  const failedAt = detail && !detail.run.success ? (detail.steps.at(-1)?.step_idx ?? null) : null;

  const set = (patch: Partial<DebuggerSearch>) =>
    navigate({ search: (prev) => ({ ...prev, ...patch }), replace: true });

  const marks = useMemo(() => {
    if (!replayResult) return undefined;
    const out: Record<number, StepMark> = {};
    for (const d of replayResult.diff.step_diffs) {
      out[d.step_idx] = replayResult.patched_steps.includes(d.step_idx)
        ? "patched"
        : d.cache_hit_b
          ? "reused"
          : "rerun";
    }
    return out;
  }, [replayResult]);

  return (
    <div className="bb-app-shell">
      <header className="bb-app-heading">
        <div className="flex items-center gap-4">
          <Link to="/" aria-label="Back to Black Box home" className="bb-nav-link">
            <ArrowLeft size={18} />
          </Link>
          <h1 className="bb-app-title">
            Black Box <span className="bb-meta">/ {runId}</span>
          </h1>
        </div>
        <div className="bb-mono bb-meta">
          <span className="bb-rec-dot" aria-hidden /> LIVE · {clock} UTC
        </div>
      </header>
      <nav className="bb-tabs" aria-label="Debugger views">
        {(
          [
            ["debugger", "Debugger"],
            ["compare", "Compare"],
            ["metrics", "Metrics"],
          ] as const
        ).map(([value, label]) => (
          <Button
            variant="ghost"
            key={value}
            type="button"
            className={`bb-tab${tab === value ? " bb-tab-active" : ""}`}
            aria-current={tab === value ? "page" : undefined}
            onClick={() => set({ tab: value })}
          >
            {label}
          </Button>
        ))}
      </nav>
      {notice && (
        <div className="bb-toast bb-mono" role="status">
          {notice}
          <button type="button" onClick={() => setNotice("")} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}
      <div className="bb-workspace">
        <RunList
          activeId={runId}
          onSelect={(id) => {
            setNotice("");
            set({ run: id, step: undefined, tab: tab === "metrics" ? "debugger" : tab });
          }}
          onInjected={(id, note) => {
            setNotice(note);
            set({ run: id, step: undefined, tab: "debugger" });
          }}
        />
        <main className="bb-workbench">
          {tab === "metrics" && <MetricsPanel />}
          {tab === "compare" && <ComparePanel runId={runId} preferred={replayResult?.run_id} />}
          {tab === "debugger" && (
            <>
              {detailQuery.isPending && (
                <div className="bb-empty">
                  <span className="bb-meta">Loading trace…</span>
                </div>
              )}
              {detailQuery.isError && (
                <div className="bb-empty">
                  <span className="bb-plate bb-mono">COULD NOT LOAD RUN</span>
                  <p>{(detailQuery.error as Error).message}</p>
                </div>
              )}
              {detail && step && (
                <>
                  <div className="bb-workbench-heading">
                    <div>
                      <p className="bb-mono bb-meta">
                        TRACE TIMELINE · {detail.run.agent ?? "company-facts"} agent · recorded{" "}
                        {relativeTime(detail.run.created_at)}
                      </p>
                      <h2 className="bb-app-title">{detail.run.question}</h2>
                    </div>
                    <div className="bb-workbench-actions">
                      {replayResult && (
                        <Button
                          className="bb-btn-shadcn bb-btn-outline"
                          variant="outline"
                          onClick={() => setReplayResult(null)}
                        >
                          <RotateCcw size={15} /> Show original
                        </Button>
                      )}
                    </div>
                  </div>
                  <StepTimeline
                    steps={detail.steps}
                    blamed={detail.top_blame_step}
                    failedAt={failedAt}
                    selected={step.step_idx}
                    onSelect={(i) => set({ step: i })}
                    marks={marks}
                  />
                  <div className="bb-answer-row">
                    <div>
                      <span className="bb-mono bb-meta">
                        {marks ? "ORIGINAL RUN · " : ""}AGENT SAID
                      </span>
                      <p className={detail.run.success ? "" : "bb-run-fail"}>
                        {detail.run.final_answer}
                      </p>
                    </div>
                    <div>
                      <span className="bb-mono bb-meta">EXPECTED</span>
                      <p>{detail.run.gold}</p>
                    </div>
                    <div className="bb-diff-outcome bb-mono" role="status">
                      {detail.run.success ? (
                        <>
                          <Check size={14} /> pass
                        </>
                      ) : (
                        <>
                          <X size={14} /> fail
                        </>
                      )}
                    </div>
                  </div>
                  <div className="bb-detail-grid">
                    <section className="bb-panel">
                      <h3 className="bb-panel-heading">
                        Step {pad2(step.step_idx)} · {step.type}
                      </h3>
                      <p className="bb-meta">Asked to {describeInput(step.type, step.input)}</p>
                      <div className="bb-mono bb-meta">OUTPUT</div>
                      <pre className="bb-code">{json(step.output)}</pre>
                      <details className="bb-details">
                        <summary className="bb-mono bb-meta">
                          INPUT · STATE BEFORE (checkpoint)
                        </summary>
                        <pre className="bb-code">{json(step.input)}</pre>
                        <pre className="bb-code">{json(step.state_before)}</pre>
                      </details>
                      <p className="bb-mono bb-meta">
                        parents {step.parents.length ? step.parents.map(pad2).join(", ") : "none"} ·{" "}
                        {step.cache_hit ? "served from cache" : "executed"} · {step.latency_ms} ms
                      </p>
                    </section>
                    <section className="bb-panel">
                      <h3 className="bb-panel-heading">Evidence</h3>
                      <p className="bb-mono bb-meta">
                        {statusOf(step.blame_score).toUpperCase()} · BLAME{" "}
                        {step.blame_score.toFixed(2)} · RANK {step.blame_rank ?? "—"} OF{" "}
                        {detail.steps.length}
                      </p>
                      <ul className="bb-evidence-list">
                        {step.direct_evidence.map((t) => (
                          <li key={t}>
                            <span className="bb-evidence-dot">●</span>
                            <span>{t}</span>
                          </li>
                        ))}
                        {step.reasons.slice(0, 4).map((r) => (
                          <li key={r.feature}>
                            <span
                              className={`bb-evidence-dot${r.shap_value > 0 ? "" : " bb-dot-cool"}`}
                            >
                              ●
                            </span>
                            <span>
                              {explainReason(r, step)}{" "}
                              <span className="bb-meta bb-mono">
                                {r.shap_value > 0 ? "raises" : "lowers"} blame{" "}
                                {r.shap_value > 0 ? "+" : ""}
                                {r.shap_value.toFixed(2)}
                              </span>
                            </span>
                          </li>
                        ))}
                      </ul>
                      <div className="bb-callout">
                        <span className="bb-mono">DATA-FLOW PATH</span>
                        <p className="bb-mono">
                          {[step.step_idx, ...downstream(detail.steps, step.step_idx)]
                            .map(pad2)
                            .join(" → ")}
                        </p>
                      </div>
                      {detail.run.fault_type && detail.run.fault_step !== null && (
                        <details className="bb-details">
                          <summary className="bb-mono bb-meta">
                            GROUND TRUTH (injected label, never a model feature)
                          </summary>
                          <p>
                            {FAULT_LABELS[detail.run.fault_type] ?? detail.run.fault_type} injected
                            at step {pad2(detail.run.fault_step)} · top-1 blame{" "}
                            {detail.top_blame_step === detail.run.fault_step
                              ? "matches ✓"
                              : "differs ✕"}
                          </p>
                        </details>
                      )}
                    </section>
                  </div>
                  <div className="bb-detail-grid">
                    <PatchEditor detail={detail} step={step} onReplayed={setReplayResult} />
                    <VerifyPanel runId={runId} disabled={detail.run.success} />
                  </div>
                  {replayResult && <ReplayPanel result={replayResult} />}
                </>
              )}
            </>
          )}
        </main>
      </div>
      <StatusStrip />
    </div>
  );
}
