// Presentation helpers for recorded traces: one-line step summaries, readable
// SHAP reasons and data-flow paths. Mirrors the wording of the Streamlit debugger.
import type { Reason, StepRecord } from "./api";

/** Known fields of recorded step inputs/outputs (all optional; payloads are agent-defined JSON). */
export type Payload = {
  actions?: unknown;
  action?: unknown;
  name?: unknown;
  query?: unknown;
  field?: unknown;
  expr?: unknown;
  title?: unknown;
  value?: unknown;
  result?: unknown;
  answer?: unknown;
  question?: unknown;
  company?: unknown;
  source?: unknown;
  doc_id?: unknown;
  top_scores?: unknown;
};
export const isObj = (v: unknown): v is Payload =>
  typeof v === "object" && v !== null && !Array.isArray(v);
const nameOf = (v: unknown, fallback: string) =>
  isObj(v) && v.name !== undefined ? String(v.name) : fallback;

export type StepStatus = "suspect" | "watch" | "ok";
export const statusOf = (score: number): StepStatus =>
  score >= 0.5 ? "suspect" : score >= 0.2 ? "watch" : "ok";

export const STEP_LABELS: Record<string, string> = {
  plan: "Plan actions",
  retrieve: "Retrieve document",
  extract: "Extract field",
  calculate: "Calculate",
  answer: "Compose answer",
};

export function summarize(type: string, out: unknown): string {
  if (out === null || out === undefined) return "no output";
  if (!isObj(out)) return String(out).slice(0, 90);
  if (type === "plan" && Array.isArray(out.actions)) {
    return (out.actions as Payload[])
      .map((a) =>
        a.action === "retrieve"
          ? `retrieve(${a.name})`
          : a.action === "extract"
            ? `extract(${a.field})`
            : a.action === "calculate"
              ? `calc[${a.expr}]`
              : String(a.action),
      )
      .join(" → ");
  }
  if (type === "retrieve") return String(out.title ?? "?");
  if (type === "extract") return `${String(out.field)} = ${String(out.value)}`;
  if (type === "calculate") return `${String(out.expr)} = ${String(out.result)}`;
  if (type === "answer") return String(out.answer ?? "");
  return JSON.stringify(out).slice(0, 90);
}

/** What the step was asked to do, in a few words. */
export function describeInput(type: string, inp: unknown): string {
  if (!isObj(inp)) return "";
  if (type === "plan") return String(inp.question ?? "");
  if (type === "retrieve") return `look up "${String(inp.name ?? inp.query ?? "")}"`;
  if (type === "extract")
    return `read "${String(inp.field)}" from ${nameOf(inp.company, nameOf(inp.source, "the document"))}`;
  if (type === "calculate") return String(inp.expr ?? "");
  if (type === "answer") return "write the final answer from collected facts";
  return "";
}

const ONLY_FOR: Record<string, string> = {
  grounding_match: "retrieve",
  top_score_gap: "retrieve",
  extracted_in_source: "extract",
  calc_args_traceable: "calculate",
  calc_result_recheckable: "calculate",
  calc_result_consistent: "calculate",
  plan_action_coverage: "plan",
  plan_question_consistent: "plan",
};

/** Turn a SHAP reason into a sentence grounded in the recorded step. */
export function explainReason(reason: Reason, step: StepRecord): string {
  const { feature: f, value: v } = reason;
  const inp = isObj(step.input) ? step.input : {};
  const out = isObj(step.output) ? step.output : {};
  const only = ONLY_FOR[f];
  if (only && step.type !== only)
    return `not ${/^[aeiou]/.test(only) ? "an" : "a"} ${only} step, so ${f} does not apply`;
  switch (f) {
    case "grounding_match":
      return `retrieved “${String(out.title)}” for requested “${String(inp.name ?? inp.query)}” (name match ${v.toFixed(2)})`;
    case "top_score_gap":
      return `rank-1 vs rank-2 retrieval score gap is ${v.toFixed(2)}`;
    case "extracted_in_source":
      return v
        ? `extracted value ${String(out.value)} appears in the source document`
        : `extracted value ${String(out.value)} is not in the source document`;
    case "calc_args_traceable":
      return v
        ? "every variable in the expression came from an earlier step"
        : "the expression uses values with no recorded source";
    case "calc_result_recheckable":
      return v
        ? "the calculation can be re-checked from its recorded inputs"
        : "the calculation cannot be re-checked";
    case "calc_result_consistent":
      return v
        ? "re-running the expression reproduces the recorded result"
        : "re-running the expression gives a different result";
    case "plan_action_coverage":
      return `${Math.round(v * 100)}% of planned actions were executed`;
    case "plan_question_consistent":
      return v
        ? "the planned calculation matches what the question asks"
        : "the planned calculation does not match the question";
    case "num_downstream_consumers":
      return `output consumed by ${v} later step${v === 1 ? "" : "s"}`;
    case "on_path_to_final":
      return v
        ? "on the data-flow path to the final answer"
        : "not on the path to the final answer";
    case "pos_norm":
      return `sits ${Math.round(v * 100)}% of the way through the run`;
    case "steps_remaining":
      return `${v} step${v === 1 ? "" : "s"} remain after it`;
    case "has_error":
      return v ? "the step raised an error" : "the step ran without error";
    case "is_empty_output":
      return v ? "the output is empty" : "the output is non-empty";
    case "output_len":
      return `output is ${v} characters long`;
    case "cosine_sim_question":
      return `output text similarity to the question is ${v.toFixed(2)}`;
    case "valid_json":
      return v ? "output is valid structured data" : "output is not valid structured data";
    default:
      if (f.startsWith("type_")) return `step type ${v ? "is" : "is not"} ${f.slice(5)}`;
      return `${f} = ${v}`;
  }
}

/** Steps that (transitively) consumed `start`'s output, in order. */
export function downstream(steps: StepRecord[], start: number): number[] {
  const children = new Map<number, number[]>();
  for (const s of steps)
    for (const p of s.parents ?? []) children.set(p, [...(children.get(p) ?? []), s.step_idx]);
  const seen = new Set<number>();
  const stack = [start];
  while (stack.length) {
    for (const c of children.get(stack.pop() as number) ?? []) {
      if (!seen.has(c)) {
        seen.add(c);
        stack.push(c);
      }
    }
  }
  return [...seen].sort((a, b) => a - b);
}

export const pad2 = (n: number) => String(n).padStart(2, "0");
export const pct = (x: number, digits = 0) => `${(x * 100).toFixed(digits)}%`;

export function relativeTime(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "—";
  const t = Date.parse(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  if (Number.isNaN(t)) return "—";
  const s = Math.max(0, Math.round((now - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export const FAULT_LABELS: Record<string, string> = {
  wrong_retrieval: "Wrong retrieval",
  bad_tool_arg: "Bad tool argument",
  corrupted_extract: "Corrupted extract",
  dropped_context: "Dropped context",
  bad_plan: "Bad plan",
  benign: "Benign fault",
};
