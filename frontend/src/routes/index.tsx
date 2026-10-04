import type { ReactNode } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowRight, ArrowUpRight, Check, Crosshair, X } from "lucide-react";
import { DEMO_RUN_ID, queries, type RunDetail } from "@/lib/api";
import { describeInput, downstream, explainReason, pad2, pct, summarize } from "@/lib/trace";
import { HeroVideo, StatusStrip, StepTimeline } from "@/components/blackbox";
import { useUtcClock } from "@/hooks/use-blackbox";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Black Box | Find the step that broke your agent" },
      {
        name: "description",
        content:
          "Black Box records every run, ranks the step most likely at fault, then replays only the fix.",
      },
      { property: "og:title", content: "Black Box | Find the step that broke your agent" },
      {
        property: "og:description",
        content:
          "Black Box records every run, ranks the step most likely at fault, then replays only the fix.",
      },
    ],
  }),
  component: Index,
});

function Section({
  id,
  label,
  kicker,
  children,
}: {
  id?: string;
  label: string;
  kicker?: string;
  children: ReactNode;
}) {
  return (
    <section className="bb-section" id={id}>
      <div className="bb-section-label">
        <span className="bb-mono">{label}</span>
      </div>
      <div className="bb-section-content">
        {kicker && <p className="bb-kicker bb-mono">// {kicker}</p>}
        {children}
      </div>
      <Crosshair aria-hidden className="bb-crosshair" size={14} strokeWidth={1} />
    </section>
  );
}

function Plate({ children }: { children: ReactNode }) {
  return <div className="bb-plate bb-mono">{children}</div>;
}

function Skeleton({ h = 122 }: { h?: number }) {
  return <div className="bb-skeleton" style={{ height: h }} aria-hidden />;
}

function lastStep(detail: RunDetail) {
  return detail.steps.at(-1)?.step_idx ?? null;
}

function DemoTimeline({ detail, compact = false }: { detail: RunDetail; compact?: boolean }) {
  return (
    <StepTimeline
      steps={detail.steps}
      blamed={detail.top_blame_step}
      failedAt={detail.run.success ? null : lastStep(detail)}
      compact={compact}
      renderLink={(step, children, className, style) => (
        <Link
          className={className}
          style={style}
          to="/debugger"
          search={{ run: detail.run.run_id, step: step.step_idx, tab: "debugger" }}
          aria-label={`Step ${step.step_idx}, ${step.type}, blame score ${step.blame_score.toFixed(2)}`}
        >
          {children}
        </Link>
      )}
    />
  );
}

function RecorderCard({ detail }: { detail: RunDetail | undefined }) {
  const blamed = detail?.steps.find((s) => s.step_idx === detail.top_blame_step);
  return (
    <aside className="bb-recorder-card" aria-label="Latest diagnosed run">
      <div className="bb-recorder-head bb-mono">
        <span>
          <span className="bb-rec-dot" aria-hidden /> REC · {detail?.run.run_id ?? "…"}
        </span>
        <span className="bb-meta">{detail?.run.agent ?? "company-facts"} agent</span>
      </div>
      {detail ? (
        <>
          <p className="bb-recorder-q">“{detail.run.question}”</p>
          <dl className="bb-recorder-dl">
            <dt className="bb-mono bb-meta">AGENT SAID</dt>
            <dd className="bb-run-fail">{detail.run.final_answer}</dd>
            <dt className="bb-mono bb-meta">EXPECTED</dt>
            <dd>{detail.run.gold}</dd>
            {blamed && (
              <>
                <dt className="bb-mono bb-meta">BLAMED</dt>
                <dd>
                  step {pad2(blamed.step_idx)} · {blamed.type} ·{" "}
                  <span className="bb-mono">{blamed.blame_score.toFixed(2)}</span>
                  <div className="bb-meta bb-recorder-why">
                    {blamed.reasons[0]
                      ? explainReason(blamed.reasons[0], blamed)
                      : summarize(blamed.type, blamed.output)}
                  </div>
                </dd>
              </>
            )}
          </dl>
        </>
      ) : (
        <Skeleton h={160} />
      )}
    </aside>
  );
}

function Index() {
  const clock = useUtcClock();
  const demo = useQuery(queries.run(DEMO_RUN_ID));
  const dashboard = useQuery(queries.dashboard());
  const replays = useQuery(queries.replays(DEMO_RUN_ID));
  const cleanRef = useQuery(queries.cleanReference(DEMO_RUN_ID));
  // Only user patch replays (rp_*), not automatic verification runs
  const latestReplay = replays.data
    ?.filter((r) => r.run_id.startsWith("rp_"))
    .sort((a, b) => (b.created_at ?? "").localeCompare(a.created_at ?? ""))[0];
  const compareWith = latestReplay?.run_id ?? cleanRef.data?.run?.run_id;
  const diffQuery = useQuery({
    ...queries.diff(DEMO_RUN_ID, compareWith ?? ""),
    enabled: Boolean(compareWith),
  });

  const detail = demo.data;
  const blamedIdx = detail?.top_blame_step ?? null;
  const blamed = detail?.steps.find((s) => s.step_idx === blamedIdx);
  const failedAt = detail ? lastStep(detail) : null;
  const flow = detail && blamedIdx !== null ? downstream(detail.steps, blamedIdx) : [];
  const metrics = dashboard.data?.metrics;
  const seen = metrics?.seen_faults_test;
  const heldout = metrics?.heldout_faults;
  const bestBaseline = seen
    ? Math.max(
        seen.metrics.random.top1,
        seen.metrics.last_step.top1,
        seen.metrics.first_anomaly.top1,
      )
    : 0;
  const prefix = blamedIdx ?? 0;
  const suffix = detail && blamedIdx !== null ? detail.steps.length - blamedIdx - 1 : 0;

  return (
    <div>
      <header className="bb-nav bb-nav-overlay">
        <Link to="/" className="bb-brand">
          Black Box<span className="bb-brand-mark">_</span>
        </Link>
        <nav className="bb-nav-links" aria-label="Main navigation">
          <a className="bb-nav-link" href="#diagnosis">
            How it works
          </a>
          <a className="bb-nav-link" href="#evaluation">
            Evaluation
          </a>
          <Link className="bb-nav-link" to="/debugger">
            Debugger <ArrowUpRight size={13} aria-hidden />
          </Link>
        </nav>
      </header>
      <main className="bb-main">
        <section className="bb-hero bb-hero-video" aria-labelledby="hero-heading">
          <HeroVideo src="/media/hero-servers.mp4" poster="/media/hero-poster.jpg" />
          <div className="bb-hero-cap bb-mono">
            <span className="bb-meta">SYS.TIME</span>
            <br />
            {clock} UTC
          </div>
          <div className="bb-hero-grid">
            <div className="bb-hero-copy">
              <Plate>
                {detail
                  ? `RUN ${detail.run.run_id} · ${detail.run.success ? "PASSED" : `FAILED AT STEP ${pad2(failedAt ?? 0)}`}`
                  : "CONNECTING TO RECORDER…"}
              </Plate>
              <h1 id="hero-heading">
                Find the step
                <br />
                <span className="bb-headline-line">
                  <strong>that broke</strong> your agent.
                </span>
              </h1>
              <p className="bb-hero-description">
                Black Box records every run, ranks the step most likely at fault with a trained
                model, then proves it by replaying only the fix.
              </p>
              <div className="bb-hero-actions">
                <Link
                  className="bb-button"
                  to="/debugger"
                  search={{ run: DEMO_RUN_ID, step: blamedIdx ?? undefined, tab: "debugger" }}
                >
                  Open the failed run <ArrowRight size={16} aria-hidden />
                </Link>
                <a className="bb-button-secondary bb-button-glass" href="#evaluation">
                  See the benchmark
                </a>
              </div>
            </div>
            <RecorderCard detail={detail} />
          </div>
          <div className="bb-hero-timeline">
            {detail ? <DemoTimeline detail={detail} /> : <Skeleton />}
          </div>
        </section>

        <Section label="LATE FAILURE">
          <h2>The first wrong step is easy to miss.</h2>
          {detail ? <DemoTimeline detail={detail} compact /> : <Skeleton h={96} />}
          {blamed && failedAt !== null && (
            <p className="bb-section-copy">
              Step {pad2(blamed.step_idx)} was asked to {describeInput(blamed.type, blamed.input)}{" "}
              and returned <strong>{summarize(blamed.type, blamed.output)}</strong>. Nothing errors.
              The run looks healthy until step {pad2(failedAt)} answers “{detail?.run.final_answer}
              ”.
            </p>
          )}
        </Section>

        <Section id="diagnosis" label="DIAGNOSIS" kicker="DIAGNOSIS">
          <h2>See why the trace went off course.</h2>
          <div className="bb-diagnosis-grid">
            <div className="bb-schematic" aria-label="Recorder cutaway schematic">
              <svg
                viewBox="0 0 1000 420"
                role="img"
                aria-label="Wireframe cutaway, data path, and highlighted cache core"
              >
                <g fill="none" stroke="var(--carbon)" strokeWidth="1.4" opacity="0.82">
                  <path d="M278 144 505 67l223 78v148l-223 79-227-79z" />
                  <path d="m278 144 227 83 223-82M505 227v145M352 119v160l153 64M652 117v164l-147 91M278 292l227-65 223 65" />
                  <path
                    d="m325 128 180 66 177-66M325 309l180-51 175 49M352 119l-74 25m374-29 76 30"
                    strokeDasharray="4 7"
                  />
                  <path
                    d="M505 195c35-23 50-23 84-2m-81 15c37-26 61-26 96-4m-98 26c38-28 66-29 105-5"
                    opacity="0.7"
                  />
                  <path d="m234 211 44-7m444 26 73 5m-312-179V29m-176 84-30-22m399 15 28-33" />
                </g>
                <path
                  d="m469 173 42-14 46 16v43l-46 15-42-17z"
                  fill="var(--signal)"
                  stroke="var(--carbon)"
                  strokeWidth="1.5"
                />
                <path
                  d="m469 173 42 17 46-15m-46 15v43"
                  fill="none"
                  stroke="var(--deck)"
                  strokeWidth="1"
                />
                <g fill="var(--carbon)">
                  <circle cx="278" cy="144" r="3" />
                  <circle cx="728" cy="145" r="3" />
                  <circle cx="505" cy="372" r="3" />
                </g>
                <g stroke="var(--hairline)" strokeWidth="1">
                  <path d="M35 35h18m-9-9v18M947 35h18m-9-9v18M35 385h18m-9-9v18M947 385h18m-9-9v18" />
                </g>
              </svg>
              <span className="bb-schematic-label bb-label-ground bb-mono">grounding gap</span>
              <span className="bb-schematic-label bb-label-flow bb-mono">data-flow path</span>
              <span className="bb-schematic-label bb-label-cache bb-mono">cache boundary</span>
            </div>
            <div className="bb-panel">
              <h3 className="bb-panel-heading">
                Evidence for step {blamed ? pad2(blamed.step_idx) : "—"}
              </h3>
              <p className="bb-mono bb-meta">
                LIGHTGBM BLAME {blamed ? blamed.blame_score.toFixed(2) : "—"} · TOP SHAP REASONS
              </p>
              <ul className="bb-evidence-list">
                {blamed?.direct_evidence.map((t) => (
                  <li key={t}>
                    <span className="bb-evidence-dot">●</span>
                    <span>{t}</span>
                  </li>
                ))}
                {blamed?.reasons.slice(0, 3).map((r) => (
                  <li key={r.feature}>
                    <span className="bb-evidence-dot">●</span>
                    <span>
                      {explainReason(r, blamed)}{" "}
                      <span className="bb-meta bb-mono">
                        ({r.shap_value > 0 ? "+" : ""}
                        {r.shap_value.toFixed(2)})
                      </span>
                    </span>
                  </li>
                ))}
              </ul>
              {flow.length > 0 && (
                <div className="bb-callout">
                  <span className="bb-mono">DATA-FLOW PATH</span>
                  <p className="bb-mono">
                    {[blamedIdx, ...flow].map((i) => pad2(i ?? 0)).join(" → ")}
                  </p>
                </div>
              )}
            </div>
          </div>
        </Section>

        <Section label="REPLAY">
          <h2>Keep the good work. Re-run the fix.</h2>
          <div className="bb-counter-grid">
            <div className="bb-counter">
              <div className="bb-numeral bb-mono">{detail ? prefix : "—"}</div>
              <div className="bb-counter-label">steps reused from the recording</div>
            </div>
            <div className="bb-counter">
              <div className="bb-numeral bb-mono">{detail ? 1 : "—"}</div>
              <div className="bb-counter-label">step patched</div>
            </div>
            <div className="bb-counter">
              <div className="bb-numeral bb-mono">{detail ? suffix : "—"}</div>
              <div className="bb-counter-label">downstream steps re-checked</div>
            </div>
          </div>
          <p className="bb-section-copy">
            Steps before the patch are replayed from the trace and the content-hashed cache, so
            testing a fix costs only what comes after it.
            {latestReplay && (
              <>
                {" "}
                Last replay <span className="bb-mono">{latestReplay.run_id}</span>:{" "}
                {latestReplay.success ? "fail → pass" : "still failing"}.
              </>
            )}
          </p>
        </Section>

        <Section id="evaluation" label="EVALUATION" kicker="EVALUATION">
          <h2>Find the fault, then prove the fix.</h2>
          <div className="bb-metric-grid">
            {[
              [
                "TOP-1 · SEEN FAULTS",
                seen?.metrics.lightgbm.top1,
                seen ? `${seen.n_runs} failed runs, unseen questions` : "",
              ],
              [
                "TOP-1 · HELD-OUT FAULTS",
                heldout?.metrics.lightgbm.top1,
                heldout ? `${heldout.n_runs} runs, fault types never trained on` : "",
              ],
              ["BEST BASELINE", bestBaseline || undefined, "random / last step / first anomaly"],
            ].map(([label, value, note]) => (
              <article className="bb-metric-card" key={label as string}>
                <div className="bb-mono bb-meta">{label as string}</div>
                <div className="bb-metric-value">
                  {typeof value === "number" ? pct(value, 1) : "—"}
                </div>
                <div className="bb-metric-bar">
                  <span style={{ width: `${typeof value === "number" ? value * 100 : 0}%` }} />
                </div>
                <div className="bb-meta bb-metric-note">{note as string}</div>
              </article>
            ))}
          </div>
          {heldout && (
            <p className="bb-section-copy bb-mono bb-meta">
              Root cause verified by replay on held-out faults:{" "}
              {pct(heldout.replay_verified.lightgbm_top1_root_cause, 1)}
              {heldout.replay_verified.root_cause_ci95 &&
                ` (95% CI ${pct(heldout.replay_verified.root_cause_ci95[0])}–${pct(heldout.replay_verified.root_cause_ci95[1])})`}{" "}
              · source: data/metrics.json
            </p>
          )}
        </Section>

        <Section label="TRACE DIFF">
          <h2>Verify the patch against the original.</h2>
          {diffQuery.data ? (
            <div className="bb-diff">
              {(["a", "b"] as const).map((side) => (
                <div className="bb-diff-column" key={side}>
                  <div className="bb-diff-title bb-mono">
                    {side === "a"
                      ? `ORIGINAL ${diffQuery.data.run_a_id}`
                      : `${latestReplay ? "PATCHED" : "CLEAN"} ${diffQuery.data.run_b_id}`}
                  </div>
                  {diffQuery.data.step_diffs.map((d) => (
                    <div
                      key={d.step_idx}
                      className={`bb-diff-step bb-mono${d.changed ? " bb-diff-changed" : ""}`}
                    >
                      {pad2(d.step_idx)} &nbsp;{" "}
                      {summarize(d.type, side === "a" ? d.output_a : d.output_b)}
                    </div>
                  ))}
                  <div className="bb-diff-outcome bb-mono">
                    {side === "a" ? (
                      <>
                        <X size={14} /> fail
                      </>
                    ) : (
                      <>
                        <Check size={14} /> pass{" "}
                        <span className="bb-verified">
                          first divergence: step {pad2(diffQuery.data.first_divergent_step ?? 0)}
                        </span>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <Skeleton h={260} />
          )}
        </Section>

        <div className="bb-footer-cta">
          <Link className="bb-button" to="/debugger" search={{ run: DEMO_RUN_ID, tab: "debugger" }}>
            Open the debugger <ArrowRight size={16} aria-hidden />
          </Link>
        </div>
        <StatusStrip />
      </main>
    </div>
  );
}
