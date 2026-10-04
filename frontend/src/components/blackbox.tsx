// Shared Black Box UI pieces: live clock, live system status, step timeline.
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  Calculator,
  Check,
  Database,
  FileSearch,
  ListTree,
  MessageSquareText,
  X,
} from "lucide-react";
import { type StepRecord } from "@/lib/api";
import { useLiveStatus } from "@/hooks/use-blackbox";
import { queries } from "@/lib/api";
import { pad2, statusOf } from "@/lib/trace";

export function StatusStrip({ extra }: { extra?: ReactNode }) {
  const { online, loading, latency, health } = useLiveStatus();
  const dashboard = useQuery({ ...queries.dashboard(), enabled: online });
  return (
    <footer className="bb-status-strip bb-mono" aria-live="polite">
      <span className={online ? "bb-status" : loading ? "bb-meta" : "bb-status-down"}>
        <span className={online ? "bb-pulse-dot" : "bb-dead-dot"} aria-hidden />{" "}
        {online
          ? "RECORDER ONLINE"
          : loading
            ? "CONNECTING…"
            : "API OFFLINE — start: uvicorn api.main:app"}
      </span>
      {online && <span>API {latency ?? "—"} ms</span>}
      {online && dashboard.data && (
        <span>
          {dashboard.data.total_runs} runs recorded · {dashboard.data.failed} failed
        </span>
      )}
      {online && health && <span>model {health.model ? "loaded" : "missing"}</span>}
      {extra}
      <span>blackbox v1.0</span>
    </footer>
  );
}

const GLYPHS: Record<string, ReactNode> = {
  plan: <ListTree size={13} />,
  retrieve: <FileSearch size={13} />,
  extract: <Database size={13} />,
  calculate: <Calculator size={13} />,
  answer: <MessageSquareText size={13} />,
};

export type StepMark = "reused" | "patched" | "rerun" | undefined;

export function StepTimeline({
  steps,
  blamed,
  failedAt,
  selected,
  onSelect,
  marks,
  compact = false,
  renderLink,
}: {
  steps: StepRecord[];
  blamed: number | null;
  failedAt: number | null;
  selected?: number | undefined;
  onSelect?: ((idx: number) => void) | undefined;
  marks?: Record<number, StepMark> | undefined;
  compact?: boolean;
  renderLink?: (
    step: StepRecord,
    children: ReactNode,
    className: string,
    style: React.CSSProperties,
  ) => ReactNode;
}) {
  return (
    <div
      className={`bb-timeline${compact ? " bb-timeline-wide" : ""}`}
      style={{ gridTemplateColumns: `repeat(${Math.max(steps.length, 1)}, minmax(0, 1fr))` }}
      aria-label={`${steps.length}-step recorded run`}
    >
      {steps.map((step, index) => {
        const mark = marks?.[step.step_idx];
        const isBlamed = step.step_idx === blamed && !marks;
        const isFailed = step.step_idx === failedAt && !marks;
        const status = statusOf(step.blame_score);
        const className = `bb-step bb-replay-flow${isBlamed ? " bb-step-blamed" : ""}${isFailed ? " bb-step-failed" : ""}${
          !isBlamed && status === "watch" && !marks ? " bb-step-watch" : ""
        }${selected === step.step_idx ? " bb-step-selected" : ""}${
          mark === "reused" ? " bb-step-reused" : ""
        }${mark === "rerun" ? " bb-step-rerun" : ""}${mark === "patched" ? " bb-step-patched" : ""}`;
        const label =
          mark ??
          (isFailed
            ? "fail"
            : isBlamed
              ? `blamed ${step.blame_score.toFixed(2)}`
              : step.blame_score.toFixed(2));
        const content = (
          <>
            {isFailed ? (
              <X size={13} />
            ) : mark === "patched" ? (
              <Check size={13} />
            ) : (
              (GLYPHS[step.type] ?? <Activity size={13} />)
            )}
            <span className="bb-step-index bb-mono">{pad2(step.step_idx)}</span>
            {!compact && <span className="bb-step-label">{step.type}</span>}
            <span className="bb-score bb-mono">{label}</span>
            {isBlamed && (
              <span className="bb-heat" aria-label="Blame confidence">
                <span className="bb-heat-fill" style={{ width: `${step.blame_score * 100}%` }} />
              </span>
            )}
          </>
        );
        const style = { animationDelay: `${index * 80}ms` };
        if (renderLink)
          return (
            <span key={step.step_idx} style={{ display: "contents" }}>
              {renderLink(step, content, className, style)}
            </span>
          );
        return (
          <button
            key={step.step_idx}
            type="button"
            className={className}
            style={style}
            onClick={() => onSelect?.(step.step_idx)}
            aria-pressed={selected === step.step_idx}
            aria-label={`Step ${step.step_idx}, ${step.type}, blame score ${step.blame_score.toFixed(2)}${mark ? `, ${mark}` : ""}`}
          >
            {content}
          </button>
        );
      })}
    </div>
  );
}

/** Background hero video. React does not render `muted` during SSR, so set it before play(). */
export function HeroVideo({ src, poster }: { src: string; poster: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    video.muted = true;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      video.pause();
      return;
    }
    video.play().catch(() => {
      /* autoplay blocked: the poster frame stays visible */
    });
  }, []);
  return (
    <div className="bb-hero-media" aria-hidden>
      <video ref={ref} autoPlay muted loop playsInline preload="auto" poster={poster}>
        <source src={src} type="video/mp4" />
      </video>
      <div className="bb-hero-scrim" />
      <div className="bb-hero-scanlines" />
    </div>
  );
}
